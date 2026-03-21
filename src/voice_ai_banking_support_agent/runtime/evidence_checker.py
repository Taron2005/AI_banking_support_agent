from __future__ import annotations

from dataclasses import dataclass

from .models import EvidenceDecision, RetrievedChunk


@dataclass(frozen=True)
class EvidenceConfig:
    min_chunks: int = 1
    min_top_score: float = 0.15
    branch_address_patterns: tuple[str, ...] = ("հասցե", "address")


class EvidenceChecker:
    """Heuristic sufficiency checks before answer generation."""

    def __init__(self, cfg: EvidenceConfig | None = None) -> None:
        self._cfg = cfg or EvidenceConfig()

    def assess(self, query: str, topic: str, chunks: list[RetrievedChunk]) -> EvidenceDecision:
        if not chunks:
            return EvidenceDecision(sufficient=False, reason="no_retrieval_hits")

        scores = [c.score for c in chunks]
        max_score = max(scores)
        if len(chunks) < self._cfg.min_chunks:
            return EvidenceDecision(
                sufficient=False,
                reason="not_enough_chunks",
                matched_chunk_count=len(chunks),
                max_score=max_score,
                min_score=min(scores),
            )
        if max_score < self._cfg.min_top_score:
            return EvidenceDecision(
                sufficient=False,
                reason="low_retrieval_score",
                matched_chunk_count=len(chunks),
                max_score=max_score,
                min_score=min(scores),
            )

        lower_q = query.lower()
        if topic == "branch" and any(t in lower_q for t in ("where", "որտե", "address", "հասցե")):
            has_address_signal = any(
                any(p in c.chunk.cleaned_text.lower() for p in self._cfg.branch_address_patterns) for c in chunks
            )
            if not has_address_signal:
                return EvidenceDecision(
                    sufficient=False,
                    reason="branch_question_without_address_evidence",
                    matched_chunk_count=len(chunks),
                    max_score=max_score,
                    min_score=min(scores),
                )

        return EvidenceDecision(
            sufficient=True,
            reason="sufficient",
            matched_chunk_count=len(chunks),
            max_score=max_score,
            min_score=min(scores),
        )

