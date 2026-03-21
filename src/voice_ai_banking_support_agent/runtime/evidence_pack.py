from __future__ import annotations

import re

from .evidence_select import _tokens, content_substance_score
from .models import RetrievedChunk


def _normalize_for_near_dup(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").lower().strip())
    return t[:2400]


def _token_set_for_dedupe(text: str) -> frozenset[str]:
    return frozenset(_tokens(_normalize_for_near_dup(text)))


def jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / u


def strip_navigation_lines(text: str) -> str:
    """
    Drop lines that look like menu crumbs / nav-only rows before sending text to the LLM.
    """

    if not (text or "").strip():
        return text
    kept: list[str] = []
    for raw in text.replace("\r", "\n").split("\n"):
        ln = raw.strip()
        if not ln:
            continue
        low = ln.lower()
        # Single-token or breadcrumb-style rows (hy/en).
        if len(ln) < 22:
            sub = content_substance_score(ln)
            has_digit = any(ch.isdigit() for ch in ln)
            if sub < 0.22 and not has_digit:
                continue
            if low in (
                "home",
                "menu",
                "search",
                "գլխավոր",
                "մենյու",
                "հետ",
                "back",
                "next",
                "»",
                "«",
            ):
                continue
        kept.append(ln)
    return "\n".join(kept) if kept else text.strip()


def dedupe_retrieved_chunks(
    chunks: list[RetrievedChunk],
    *,
    max_keep: int,
    near_duplicate_jaccard: float = 0.88,
) -> list[RetrievedChunk]:
    """
    Remove exact duplicate chunk_ids and near-duplicate bodies (common when pages overlap).
    Preserves input order (retrieval score order).
    """

    if not chunks or max_keep <= 0:
        return []
    seen_ids: set[str] = set()
    sigs: list[tuple[frozenset[str], RetrievedChunk]] = []
    out: list[RetrievedChunk] = []
    for c in chunks:
        cid = (c.chunk.chunk_id or "").strip()
        if cid and cid in seen_ids:
            continue
        body = strip_navigation_lines(c.chunk.cleaned_text or "")
        if not body.strip():
            continue
        sig = _token_set_for_dedupe(body)
        if sig and any(jaccard_similarity(sig, prev) >= near_duplicate_jaccard for prev, _ in sigs):
            continue
        if cid:
            seen_ids.add(cid)
        sigs.append((sig, c))
        out.append(c)
        if len(out) >= max_keep:
            break
    return out


def prepare_evidence_for_answer(
    chunks: list[RetrievedChunk],
    *,
    max_chunks: int,
) -> list[RetrievedChunk]:
    """Dedupe and trim pool for LLM / extractive paths (max_chunks final cap)."""

    pool = max(max_chunks * 2, max_chunks + 2)
    slim = dedupe_retrieved_chunks(chunks, max_keep=pool)
    return slim[:max_chunks]
