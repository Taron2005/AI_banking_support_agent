from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterator, Literal, Protocol

from .evidence_pack import strip_navigation_lines
from .evidence_select import dedupe_urls, filter_chunks_to_bank_keys, normalize_http_url, query_term_overlap
from .models import RetrievedChunk
from .prompts import LLM_USER_ANSWER_INSTRUCTIONS, STANDARD_AI_FOOTNOTE_LINE
from .rag_prompts import answer_mode_supplement

logger = logging.getLogger(__name__)

AnswerSynthesis = Literal["llm", "extractive_fallback", "extractive_only"]
AnswerMode = Literal["single_bank", "multi_bank", "comparison"]


def _answer_mode_instructions(mode: AnswerMode) -> str:
    return answer_mode_supplement(mode)

_URL_RE = re.compile(r"https?://[^\s\)\]\>\"']+", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]{0,240})\]\((https?://[^\s)]+)\)", re.IGNORECASE)

# Lines the model sometimes echoes from system/policy text — remove from customer-facing output.
_LEAKY_LINE_MARKERS = (
    "արտաքին գիտելիք",
    "մի օգտագործիր",
    "մի ավելացրու",
    "համակարգի հրահանգ",
    "համակարգի կամ ներքին",
    "numbered evidence",
    "evidence blocks",
    "user template",
    "system instruction",
    "system prompt",
    "outside knowledge",
    "do not use external",
    "do not add facts",
    "no outside knowledge",
    "only use facts",
)


def _line_looks_leaky(line: str) -> bool:
    low = line.lower()
    for m in _LEAKY_LINE_MARKERS:
        if all(ord(ch) < 128 for ch in m):
            if m.lower() in low:
                return True
        elif m in line:
            return True
    return False


def _strip_leaked_meta_lines(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        if _line_looks_leaky(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


@dataclass(frozen=True)
class AnswerGeneratorConfig:
    max_evidence_chunks: int = 4
    answer_language: str = "hy"
    max_snippet_chars: int = 520
    max_chars_per_evidence: int = 560


@dataclass(frozen=True)
class AnswerResult:
    """Structured outcome for orchestrator / API (synthesis path + optional LLM failure hint)."""

    text: str
    answer_synthesis: AnswerSynthesis
    llm_error: str | None = None


@dataclass(frozen=True)
class LlmStreamPiece:
    """One step of streamed LLM synthesis: token delta and/or terminal ``AnswerResult``."""

    delta: str | None = None
    result: AnswerResult | None = None


class AnswerBackend(Protocol):
    def generate(
        self,
        query: str,
        topic: str,
        chunks: list[RetrievedChunk],
        bank_keys: frozenset[str] | None,
        *,
        context: str | None = None,
    ) -> str: ...


def _split_sentences(text: str) -> list[str]:
    t = text.replace("\n", " ").strip()
    parts = re.split(r"(?<=[.!?\u0589])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _first_sentences(text: str, *, max_sentences: int = 2, max_chars: int = 340) -> str:
    parts = _split_sentences(text)
    out: list[str] = []
    for p in parts[:max_sentences]:
        out.append(p)
    s = " ".join(out).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0].strip() + "…"
    return s


def _query_focused_snippet(
    query: str,
    text: str,
    *,
    max_sentences: int = 3,
    max_chars: int = 560,
) -> str:
    """Pick sentences that overlap the query most, then restore reading order."""

    parts = _split_sentences(text)
    if not parts:
        return ""
    if len(parts) <= max_sentences:
        base = " ".join(parts).strip()
    else:
        ranked = sorted(
            range(len(parts)),
            key=lambda i: query_term_overlap(query, parts[i]),
            reverse=True,
        )
        take = sorted(set(ranked[:max_sentences]))
        base = " ".join(parts[i] for i in take).strip()
    if len(base) > max_chars:
        base = base[:max_chars].rsplit(" ", 1)[0].strip() + "…"
    return base


def _allowed_urls(chunks: list[RetrievedChunk]) -> set[str]:
    return {normalize_http_url(c.chunk.source_url) for c in chunks if c.chunk.source_url}


def _url_in_evidence(raw: str, allowed_norm: set[str]) -> bool:
    n = normalize_http_url(raw.rstrip(".,;։）»"))
    if not n:
        return False
    if n in allowed_norm:
        return True
    for a in allowed_norm:
        if not a:
            continue
        if n.startswith(a.rstrip("/") + "/") or a.startswith(n.rstrip("/")):
            return True
    return False


def _scrub_unknown_urls(text: str, allowed_norm: set[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        raw = m.group(0)
        return raw if _url_in_evidence(raw, allowed_norm) else ""

    return _URL_RE.sub(repl, text)


def _scrub_disallowed_markdown_links(text: str, allowed_norm: set[str]) -> str:
    """Remove [label](url) when url is not from evidence; keep label text if present."""

    def repl(m: re.Match[str]) -> str:
        url = m.group(2)
        if _url_in_evidence(url, allowed_norm):
            return m.group(0)
        label = (m.group(1) or "").strip()
        return label

    return _MD_LINK_RE.sub(repl, text)


def _normalize_llm_error(exc: BaseException) -> str:
    """Stable, short API-facing codes for known Gemini failures."""

    raw = str(exc).strip()
    if not raw:
        return type(exc).__name__
    low = raw.lower()
    for prefix in (
        "gemini_missing_api_key",
        "gemini_resource_exhausted",
        "gemini_blocked:",
        "gemini_empty_response:",
        "gemini_api_error:",
        "gemini_error:",
    ):
        if low.startswith(prefix):
            return raw[:200]
    return f"{type(exc).__name__}: {raw}"[:220]


def _append_ai_footnote_if_missing(text: str) -> str:
    """Ensure the required Armenian AI disclaimer is present after successful LLM synthesis."""

    t = text.strip()
    if STANDARD_AI_FOOTNOTE_LINE in t or "արհեստական բանականությամբ՝ բացառապես" in t:
        return t
    return f"{t}\n\n{STANDARD_AI_FOOTNOTE_LINE}"


def _tidy_whitespace(text: str) -> str:
    t = text.replace("\r\n", "\n").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


class GroundedAnswerGenerator:
    """
    Deterministic grounded answers when Gemini (or LLM client) is unavailable or declines.

    Tighter Armenian wording than raw chunks; grouped by bank when needed.
    """

    def __init__(self, cfg: AnswerGeneratorConfig | None = None) -> None:
        self._cfg = cfg or AnswerGeneratorConfig()

    def generate(
        self,
        query: str,
        topic: str,
        chunks: list[RetrievedChunk],
        bank_keys: frozenset[str] | None,
        *,
        context: str | None = None,
    ) -> str:
        used = chunks[: self._cfg.max_evidence_chunks]
        lines: list[str] = []
        lines.append(
            "Ներքևում կարճ պատասխանն ըստ պաշտոնական կայքից վերցված հատվածների է "
            "(առցանց մոդելը հասանելի չէ կամ պատասխան չի տվել)։"
        )
        if bank_keys and len(bank_keys) > 1:
            lines.append(
                f"Հարցման շրջանակում նշված են միայն հետևյալ բանկերը՝ {', '.join(sorted(bank_keys))}։"
            )
        elif bank_keys and len(bank_keys) == 1:
            only = next(iter(bank_keys))
            lines.append(f"Նշված է բանկը՝ {only}։")

        by_label: dict[str, list[RetrievedChunk]] = {}
        order: list[str] = []
        for c in used:
            label = (c.chunk.bank_name or c.chunk.bank_key or "").strip() or "Բանկ"
            if label not in by_label:
                order.append(label)
                by_label[label] = []
            by_label[label].append(c)

        multi_display = len(order) > 1 and not (bank_keys and len(bank_keys) == 1)
        if multi_display:
            lines.append("Ըստ բանկերի՝")
            for label in order:
                lines.append(f"{label}․ Նշված տվյալների համաձայն՝")
                for c in by_label[label][:2]:
                    snippet = _query_focused_snippet(
                        query, c.chunk.cleaned_text, max_sentences=3, max_chars=520
                    )
                    if snippet:
                        lines.append(f"  • {snippet}")
        else:
            lines.append("Հիմնական կետեր՝")
            for c in used:
                snippet = _query_focused_snippet(query, c.chunk.cleaned_text, max_sentences=3, max_chars=560)
                if snippet:
                    lines.append(f"• {snippet}")
        urls = dedupe_urls([c.chunk.source_url for c in used], max_n=6)
        if urls:
            lines.append("Աղբյուրներ՝")
            for u in urls:
                lines.append(u)
        return "\n".join(lines)


def _trim_evidence_text(text: str, limit: int) -> str:
    t = text.replace("\r", " ").replace("\n", " ").strip()
    if len(t) <= limit:
        return t
    return t[: limit].rsplit(" ", 1)[0].strip() + "…"


def _is_payload_too_large(exc: BaseException) -> bool:
    """Some APIs return 413 when the RAG prompt exceeds provider limits."""

    s = str(exc).lower()
    return "413" in s or "payload too large" in s or ("too large" in s and "request" in s)


def _truncate_context_block(context: str | None, max_chars: int) -> str | None:
    if not context or not context.strip():
        return None
    c = context.strip()
    if len(c) <= max_chars:
        return c
    cut = c[:max_chars].rsplit("\n", 1)[0].strip()
    return cut + "\n…"


class LLMAnswerGenerator:
    """
    Gemini-backed answer synthesis using only post-evidence chunks (grounded by construction).
    """

    def __init__(
        self,
        llm_client: object | None,
        fallback: GroundedAnswerGenerator,
        cfg: AnswerGeneratorConfig | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._fallback = fallback
        self._cfg = cfg or AnswerGeneratorConfig()

    def _build_llm_prompt(
        self,
        query: str,
        topic: str,
        scoped: list[RetrievedChunk],
        bank_keys: frozenset[str] | None,
        *,
        context: str | None,
        max_n: int,
        max_chars: int,
        answer_mode: AnswerMode,
    ) -> tuple[str, list[RetrievedChunk]] | None:
        evidence_blocks: list[str] = []
        used_for_prompt: list[RetrievedChunk] = []
        n_take = max(1, min(max_n, self._cfg.max_evidence_chunks))
        lim = max(180, min(max_chars, self._cfg.max_chars_per_evidence))
        for i, c in enumerate(scoped[:n_take], start=1):
            raw_body = strip_navigation_lines(c.chunk.cleaned_text or "")
            body = _trim_evidence_text(raw_body, lim)
            if not body:
                continue
            used_for_prompt.append(c)
            evidence_blocks.append(
                f"[{i}] Բանկ՝ {c.chunk.bank_name} | Թեմա՝ {c.chunk.topic} | URL՝ {c.chunk.source_url}\n{body}"
            )
        if not evidence_blocks:
            return None

        distinct_banks = {
            (c.chunk.bank_key or c.chunk.bank_name or "").strip().lower()
            for c in used_for_prompt
            if (c.chunk.bank_key or c.chunk.bank_name)
        }
        distinct_banks.discard("")
        multi_bank_evidence = (bank_keys is None and len(distinct_banks) > 1) or (
            bank_keys is not None and len(bank_keys) > 1
        )

        if bank_keys and len(bank_keys) > 1:
            bank_line = f"Ֆիլտր՝ միայն այս բանկերի ապացույցները՝ {', '.join(sorted(bank_keys))}։"
        elif bank_keys and len(bank_keys) == 1:
            only = next(iter(bank_keys))
            bank_line = f"Ընտրած բանկի ֆիլտր՝ {only}։"
        else:
            bank_line = "Բանկի ֆիլտր՝ բոլորը (ըստ հարցման)։"
        if multi_bank_evidence:
            bank_line += (
                " Ապացույցում կան մի քանի բանկ․ ներկայացրու յուրաքանչյուրին առանձին բաժնով՝ "
                "մանրամասն, քանի դեռ հիմնվում ես միայն այդ բանկի ապացույցի վրա։"
            )
        ctx_block = f"Զրույցի կոնտեքստ (նախորդ հարցումներ, կարճ)\n{context}\n\n" if context else ""
        mode_block = _answer_mode_instructions(answer_mode)
        prompt = (
            f"{LLM_USER_ANSWER_INSTRUCTIONS}\n\n"
            f"{mode_block}\n\n"
            f"{ctx_block}"
            f"{bank_line}\n"
            f"Հարց՝ {query}\n"
            f"Թեմա՝ {topic}\n\n"
            "Ապացույցներ (միայն որպես աղբյուր, չպատճենել մենյու-տողերն ամբողջությամբ)՝\n"
            + "\n\n".join(evidence_blocks)
        )
        return prompt, used_for_prompt

    def generate_answer_result(
        self,
        query: str,
        topic: str,
        chunks: list[RetrievedChunk],
        bank_keys: frozenset[str] | None,
        *,
        context: str | None = None,
        answer_mode: AnswerMode = "single_bank",
    ) -> AnswerResult:
        if self._llm_client is None:
            logger.warning("LLM client is not configured; using extractive fallback.")
            return AnswerResult(
                text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                answer_synthesis="extractive_fallback",
                llm_error="llm_client_not_configured",
            )
        if not chunks:
            return AnswerResult(
                text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                answer_synthesis="extractive_fallback",
                llm_error="no_evidence_chunks",
            )

        scoped = filter_chunks_to_bank_keys(chunks, bank_keys)
        base_n = self._cfg.max_evidence_chunks
        base_lim = self._cfg.max_chars_per_evidence
        # Shrink plans when the provider rejects oversized prompts (HTTP 413, etc.).
        shrink_plans: list[tuple[int, int, str | None]] = [
            (base_n, base_lim, context),
            (min(3, base_n), min(480, base_lim), context),
            (2, min(380, base_lim), context),
            (2, 300, _truncate_context_block(context, 1400)),
            (2, 260, None),
        ]
        seen: set[tuple[int, int, str | None]] = set()
        deduped: list[tuple[int, int, str | None]] = []
        for n, lim, ctx in shrink_plans:
            key = (n, lim, ctx)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((n, lim, ctx))

        fn = getattr(self._llm_client, "generate")
        last_413: Exception | None = None
        for plan_idx, (max_n, lim, ctx_use) in enumerate(deduped):
            built = self._build_llm_prompt(
                query,
                topic,
                scoped,
                bank_keys,
                context=ctx_use,
                max_n=max_n,
                max_chars=lim,
                answer_mode=answer_mode,
            )
            if built is None:
                if plan_idx == 0:
                    logger.warning("After filtering, no evidence blocks left for LLM; extractive fallback.")
                    return AnswerResult(
                        text=self._fallback.generate(query, topic, scoped, bank_keys, context=context),
                        answer_synthesis="extractive_fallback",
                        llm_error="evidence_filtered_empty",
                    )
                continue
            prompt, used_for_prompt = built
            try:
                out = fn(prompt)
            except Exception as exc:
                if _is_payload_too_large(exc) and plan_idx < len(deduped) - 1:
                    last_413 = exc
                    logger.warning(
                        "LLM payload too large; retrying with smaller evidence "
                        "(chunks_cap=%s chars/chunk=%s context=%s)",
                        max_n,
                        lim,
                        "trimmed" if ctx_use and ctx_use != context else ("off" if not ctx_use else "full"),
                    )
                    continue
                err = _normalize_llm_error(exc)
                logger.exception("Gemini/LLM call failed; using extractive fallback (%s)", err)
                return AnswerResult(
                    text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                    answer_synthesis="extractive_fallback",
                    llm_error=err,
                )
            if isinstance(out, str) and out.strip():
                candidate = _tidy_whitespace(out.strip())
                if len(candidate) < 8:
                    logger.warning("LLM returned very short text; extractive fallback.")
                    return AnswerResult(
                        text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                        answer_synthesis="extractive_fallback",
                        llm_error="llm_output_too_short",
                    )
                low = candidate.lower()
                if "ignore previous" in low or "system prompt" in low:
                    logger.warning("LLM output rejected (prompt-injection echo); extractive fallback.")
                    return AnswerResult(
                        text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                        answer_synthesis="extractive_fallback",
                        llm_error="llm_output_rejected_policy_echo",
                    )
                allowed = _allowed_urls(used_for_prompt)
                candidate = _scrub_disallowed_markdown_links(candidate, allowed)
                candidate = _scrub_unknown_urls(candidate, allowed)
                candidate = _strip_leaked_meta_lines(candidate)
                candidate = _tidy_whitespace(candidate)
                candidate = _append_ai_footnote_if_missing(candidate)
                candidate = _tidy_whitespace(candidate)
                return AnswerResult(text=candidate, answer_synthesis="llm", llm_error=None)
            logger.warning("LLM returned empty content; extractive fallback.")
            return AnswerResult(
                text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                answer_synthesis="extractive_fallback",
                llm_error="llm_empty_response",
            )

        if last_413 is not None:
            err = _normalize_llm_error(last_413)
            logger.error("LLM still rejected prompt size after shrinking; extractive fallback.")
            return AnswerResult(
                text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                answer_synthesis="extractive_fallback",
                llm_error=err,
            )
        logger.warning("LLM returned empty content; extractive fallback.")
        return AnswerResult(
            text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
            answer_synthesis="extractive_fallback",
            llm_error="llm_empty_response",
        )

    def _finalize_streamed_llm_candidate(
        self,
        raw: str,
        *,
        used_for_prompt: list[RetrievedChunk],
        chunks: list[RetrievedChunk],
        query: str,
        topic: str,
        bank_keys: frozenset[str] | None,
        context: str | None,
    ) -> AnswerResult:
        candidate = _tidy_whitespace(raw.strip())
        if len(candidate) < 8:
            logger.warning("LLM stream returned very short text; extractive fallback.")
            return AnswerResult(
                text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                answer_synthesis="extractive_fallback",
                llm_error="llm_output_too_short",
            )
        low = candidate.lower()
        if "ignore previous" in low or "system prompt" in low:
            logger.warning("LLM stream output rejected (prompt-injection echo); extractive fallback.")
            return AnswerResult(
                text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                answer_synthesis="extractive_fallback",
                llm_error="llm_output_rejected_policy_echo",
            )
        allowed = _allowed_urls(used_for_prompt)
        candidate = _scrub_disallowed_markdown_links(candidate, allowed)
        candidate = _scrub_unknown_urls(candidate, allowed)
        candidate = _strip_leaked_meta_lines(candidate)
        candidate = _tidy_whitespace(candidate)
        candidate = _append_ai_footnote_if_missing(candidate)
        candidate = _tidy_whitespace(candidate)
        return AnswerResult(text=candidate, answer_synthesis="llm", llm_error=None)

    def generate_answer_result_stream(
        self,
        query: str,
        topic: str,
        chunks: list[RetrievedChunk],
        bank_keys: frozenset[str] | None,
        *,
        context: str | None = None,
        answer_mode: AnswerMode = "single_bank",
    ) -> Iterator[LlmStreamPiece]:
        """
        Stream raw token deltas from the LLM, then yield a single terminal ``AnswerResult``
        (same post-processing as the non-streaming path).
        """

        if self._llm_client is None:
            yield LlmStreamPiece(
                result=AnswerResult(
                    text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                    answer_synthesis="extractive_fallback",
                    llm_error="llm_client_not_configured",
                )
            )
            return
        if not chunks:
            yield LlmStreamPiece(
                result=AnswerResult(
                    text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                    answer_synthesis="extractive_fallback",
                    llm_error="no_evidence_chunks",
                )
            )
            return

        scoped = filter_chunks_to_bank_keys(chunks, bank_keys)
        base_n = self._cfg.max_evidence_chunks
        base_lim = self._cfg.max_chars_per_evidence
        shrink_plans: list[tuple[int, int, str | None]] = [
            (base_n, base_lim, context),
            (min(3, base_n), min(480, base_lim), context),
            (2, min(380, base_lim), context),
            (2, 300, _truncate_context_block(context, 1400)),
            (2, 260, None),
        ]
        seen: set[tuple[int, int, str | None]] = set()
        deduped: list[tuple[int, int, str | None]] = []
        for n, lim, ctx in shrink_plans:
            key = (n, lim, ctx)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((n, lim, ctx))

        fn = getattr(self._llm_client, "generate")
        stream_fn = getattr(self._llm_client, "generate_stream", None)
        last_413: Exception | None = None
        for plan_idx, (max_n, lim, ctx_use) in enumerate(deduped):
            built = self._build_llm_prompt(
                query,
                topic,
                scoped,
                bank_keys,
                context=ctx_use,
                max_n=max_n,
                max_chars=lim,
                answer_mode=answer_mode,
            )
            if built is None:
                if plan_idx == 0:
                    logger.warning("After filtering, no evidence blocks left for LLM; extractive fallback.")
                    yield LlmStreamPiece(
                        result=AnswerResult(
                            text=self._fallback.generate(query, topic, scoped, bank_keys, context=context),
                            answer_synthesis="extractive_fallback",
                            llm_error="evidence_filtered_empty",
                        )
                    )
                    return
                continue
            prompt, used_for_prompt = built
            try:
                if stream_fn is None:
                    out = fn(prompt)
                    if isinstance(out, str) and out.strip():
                        yield LlmStreamPiece(
                            result=self._finalize_streamed_llm_candidate(
                                out,
                                used_for_prompt=used_for_prompt,
                                chunks=chunks,
                                query=query,
                                topic=topic,
                                bank_keys=bank_keys,
                                context=context,
                            )
                        )
                    else:
                        yield LlmStreamPiece(
                            result=AnswerResult(
                                text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                                answer_synthesis="extractive_fallback",
                                llm_error="llm_empty_response",
                            )
                        )
                    return

                acc: list[str] = []
                for delta in stream_fn(prompt):
                    if delta:
                        acc.append(delta)
                        yield LlmStreamPiece(delta=delta)
                full = "".join(acc)
                if not full.strip():
                    logger.warning("LLM stream returned no text; extractive fallback.")
                    yield LlmStreamPiece(
                        result=AnswerResult(
                            text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                            answer_synthesis="extractive_fallback",
                            llm_error="llm_empty_response",
                        )
                    )
                    return
                yield LlmStreamPiece(
                    result=self._finalize_streamed_llm_candidate(
                        full,
                        used_for_prompt=used_for_prompt,
                        chunks=chunks,
                        query=query,
                        topic=topic,
                        bank_keys=bank_keys,
                        context=context,
                    )
                )
                return
            except Exception as exc:
                if _is_payload_too_large(exc) and plan_idx < len(deduped) - 1:
                    last_413 = exc
                    logger.warning(
                        "LLM stream payload too large; retrying with smaller evidence "
                        "(chunks_cap=%s chars/chunk=%s context=%s)",
                        max_n,
                        lim,
                        "trimmed" if ctx_use and ctx_use != context else ("off" if not ctx_use else "full"),
                    )
                    continue
                err = _normalize_llm_error(exc)
                logger.exception("Gemini/LLM stream failed; using extractive fallback (%s)", err)
                yield LlmStreamPiece(
                    result=AnswerResult(
                        text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                        answer_synthesis="extractive_fallback",
                        llm_error=err,
                    )
                )
                return

        if last_413 is not None:
            err = _normalize_llm_error(last_413)
            logger.error("LLM stream still rejected prompt size after shrinking; extractive fallback.")
            yield LlmStreamPiece(
                result=AnswerResult(
                    text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                    answer_synthesis="extractive_fallback",
                    llm_error=err,
                )
            )
            return
        yield LlmStreamPiece(
            result=AnswerResult(
                text=self._fallback.generate(query, topic, chunks, bank_keys, context=context),
                answer_synthesis="extractive_fallback",
                llm_error="llm_empty_response",
            )
        )

    def generate(
        self,
        query: str,
        topic: str,
        chunks: list[RetrievedChunk],
        bank_keys: frozenset[str] | None,
        *,
        context: str | None = None,
    ) -> str:
        return self.generate_answer_result(
            query, topic, chunks, bank_keys, context=context, answer_mode="single_bank"
        ).text
