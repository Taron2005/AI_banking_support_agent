from __future__ import annotations

from .models import RetrievedChunk


def dominant_bank_key_from_chunks(chunks: list[RetrievedChunk]) -> str | None:
    """Pick the bank with the largest sum of retrieval scores (for single-bank evidence restriction)."""

    scores: dict[str, float] = {}
    for c in chunks:
        k = (c.chunk.bank_key or "").strip().lower()
        if not k:
            continue
        scores[k] = scores.get(k, 0.0) + float(c.score)
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def format_bank_catalog_for_prompt(bank_aliases: dict[str, list[str]]) -> str:
    """Stable, config-driven list for optional LLM intent prompt (English keys)."""

    return ", ".join(sorted(bank_aliases.keys()))


def supported_bank_keys_csv(bank_aliases: dict[str, list[str]]) -> str:
    return ", ".join(sorted(bank_aliases.keys()))
