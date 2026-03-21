from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import RetrievedChunk


@dataclass(frozen=True)
class AnswerGeneratorConfig:
    max_evidence_chunks: int = 4
    answer_language: str = "hy"
    max_snippet_chars: int = 520
    max_chars_per_evidence: int = 560


class AnswerBackend(Protocol):
    def generate(self, query: str, topic: str, chunks: list[RetrievedChunk], bank: str | None) -> str: ...


class GroundedAnswerGenerator:
    """
    Lightweight grounded answer generator.

    Deterministic extractive fallback when Groq is unavailable or declines.
    """

    def __init__(self, cfg: AnswerGeneratorConfig | None = None) -> None:
        self._cfg = cfg or AnswerGeneratorConfig()

    def generate(self, query: str, topic: str, chunks: list[RetrievedChunk], bank: str | None) -> str:
        used = chunks[: self._cfg.max_evidence_chunks]
        lines: list[str] = []
        if bank:
            lines.append(f"Հարցը մշակվել է `{bank}` բանկի տվյալներով։")
        else:
            banks = sorted({c.chunk.bank_name for c in used})
            lines.append(f"Հարցը մշակվել է հետևյալ բանկերի տվյալներով՝ {', '.join(banks)}։")
        lines.append("Հիմնական տվյալներ՝")
        for c in used:
            snippet = c.chunk.cleaned_text.replace("\n", " ").strip()
            if len(snippet) > self._cfg.max_snippet_chars:
                snippet = snippet[: self._cfg.max_snippet_chars] + "..."
            lines.append(f"- {snippet}")
        lines.append("Եթե պետք է, կարող եմ նեղացնել պատասխանը ըստ կոնկրետ բանկի կամ պայմանի։")
        return "\n".join(lines)


def _trim_evidence_text(text: str, limit: int) -> str:
    t = text.replace("\r", " ").replace("\n", " ").strip()
    if len(t) <= limit:
        return t
    return t[: limit].rsplit(" ", 1)[0].strip() + "…"


class LLMAnswerGenerator:
    """
    Groq-backed answer synthesis using only post-evidence chunks (grounded by construction).
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

    def generate(self, query: str, topic: str, chunks: list[RetrievedChunk], bank: str | None) -> str:
        if self._llm_client is None:
            return self._fallback.generate(query, topic, chunks, bank)
        if not chunks:
            return self._fallback.generate(query, topic, chunks, bank)

        max_n = self._cfg.max_evidence_chunks
        lim = self._cfg.max_chars_per_evidence
        evidence_blocks: list[str] = []
        for i, c in enumerate(chunks[:max_n], start=1):
            body = _trim_evidence_text(c.chunk.cleaned_text, lim)
            if not body:
                continue
            evidence_blocks.append(
                f"[{i}] Բանկ՝ {c.chunk.bank_name} | Թեմա՝ {c.chunk.topic} | URL՝ {c.chunk.source_url}\n{body}"
            )
        if not evidence_blocks:
            return self._fallback.generate(query, topic, chunks, bank)

        bank_line = f"Ընտրած բանկի ֆիլտր՝ {bank}։" if bank else "Բանկի ֆիլտր՝ բոլորը (ըստ հարցման)։"
        prompt = (
            "Պատասխանիր միայն ապացույցների հիման վրա։ Մի պատճենիր կամ մի թվարկիր հում չանկների մասնակի տեքստը։ "
            "Սինթեզիր մեկ հստակ, բնական պատասխան հայերենով։\n"
            "Ձևաչափ՝ 2–6 կարճ նախադասություն, առանց կրկնությունների, առանց URL-ների, առանց համարակալված ցուցակների։\n"
            "Եթե տվյալները թերի են՝ մեկ նախադասությամբ ասա, որ բավարար չեն, առանց գուշակելու։\n\n"
            f"{bank_line}\n"
            f"Հարց՝ {query}\n"
            f"Թեմա՝ {topic}\n\n"
            "Ապացույցներ (միայն որպես աղբյուր, չպատճենել որպես պատասխան)՝\n"
            + "\n\n".join(evidence_blocks)
        )
        try:
            fn = getattr(self._llm_client, "generate")
            out = fn(prompt)
            if isinstance(out, str) and out.strip():
                candidate = out.strip()
                if len(candidate) < 12:
                    return self._fallback.generate(query, topic, chunks, bank)
                if len(candidate) > 4000:
                    return self._fallback.generate(query, topic, chunks, bank)
                low = candidate.lower()
                if "ignore previous" in low or "system prompt" in low:
                    return self._fallback.generate(query, topic, chunks, bank)
                return candidate
        except Exception:
            return self._fallback.generate(query, topic, chunks, bank)
        return self._fallback.generate(query, topic, chunks, bank)
