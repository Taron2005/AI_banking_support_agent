from __future__ import annotations

import logging
from typing import ClassVar

from .models import RetrievedChunk

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}


def cross_encoder_rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    model_name: str,
    top_k: int,
    max_doc_chars: int = 520,
) -> list[RetrievedChunk]:
    """
    Rerank retrieved chunks with a sentence-transformers CrossEncoder.

    First load can download model weights (~hundreds of MB).
    """

    if not chunks or top_k <= 0:
        return []
    if len(chunks) <= top_k:
        return chunks[:top_k]

    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Cross-encoder reranking requires sentence-transformers (already a project dependency)."
        ) from exc

    if model_name not in _MODEL_CACHE:
        logger.info("Loading cross-encoder reranker: %s", model_name)
        _MODEL_CACHE[model_name] = CrossEncoder(model_name)

    model = _MODEL_CACHE[model_name]
    pairs = [(query, (c.chunk.cleaned_text or "")[:max_doc_chars]) for c in chunks]
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    out: list[RetrievedChunk] = []
    for sc, ch in ranked[:top_k]:
        out.append(RetrievedChunk(score=float(sc), chunk=ch.chunk))
    return out
