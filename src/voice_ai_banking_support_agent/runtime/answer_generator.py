from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import RetrievedChunk


@dataclass(frozen=True)
class AnswerGeneratorConfig:
    max_evidence_chunks: int = 3
    answer_language: str = "hy"
    max_snippet_chars: int = 260


class AnswerBackend(Protocol):
    def generate(self, query: str, topic: str, chunks: list[RetrievedChunk], bank: str | None) -> str: ...


class GroundedAnswerGenerator:
    """
    Lightweight grounded answer generator.

    This is intentionally extractive and deterministic for production safety.
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


class LLMAnswerGenerator:
    """
    Optional LLM backend that remains grounded by construction.

    It receives only query/topic/retrieved chunks from post-evidence stage.
    """

    def __init__(self, llm_client: object | None, fallback: GroundedAnswerGenerator) -> None:
        self._llm_client = llm_client
        self._fallback = fallback

    def generate(self, query: str, topic: str, chunks: list[RetrievedChunk], bank: str | None) -> str:
        if self._llm_client is None:
            return self._fallback.generate(query, topic, chunks, bank)
        # Provider-agnostic placeholder; if client exists it should expose `.generate(prompt)`.
        evidence_lines = []
        for c in chunks[:4]:
            evidence_lines.append(
                f"[{c.chunk.bank_name} | {c.chunk.topic} | {c.chunk.source_url}] {c.chunk.cleaned_text[:320]}"
            )
        prompt = (
            "Պատասխանիր միայն տրված ապացույցներով, հայերեն, կարճ ու հստակ։ "
            "Եթե ապացույցը թերի է՝ նշիր դա։\n"
            f"Հարց: {query}\nԹեմա: {topic}\nԱպացույցներ:\n" + "\n".join(evidence_lines)
        )
        try:
            fn = getattr(self._llm_client, "generate")
            out = fn(prompt)
            if isinstance(out, str) and out.strip():
                candidate = out.strip()
                # Keep voice/runtime stable on broken long/noisy outputs.
                if len(candidate) > 3500:
                    return self._fallback.generate(query, topic, chunks, bank)
                if "ignore previous" in candidate.lower():
                    return self._fallback.generate(query, topic, chunks, bank)
                return candidate
        except Exception:
            return self._fallback.generate(query, topic, chunks, bank)
        return self._fallback.generate(query, topic, chunks, bank)

