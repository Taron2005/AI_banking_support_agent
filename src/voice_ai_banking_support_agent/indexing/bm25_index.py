from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\w\u0561-\u0587\u0531-\u0556]{2,}", re.UNICODE)


def tokenize_for_bm25(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class ChunkBM25Index:
    """
    In-memory Okapi BM25 over chunk `cleaned_text`, aligned with metadata row order.

    Built lazily at runtime from `metadata.jsonl` (same order as FAISS row ids).
    """

    def __init__(self, *, metadata_path: Path) -> None:
        self._metadata_path = metadata_path
        self._chunk_ids: list[str] = []
        self._corpus: list[list[str]] = []
        self._bm25 = None

    def _ensure_built(self) -> None:
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Hybrid BM25 retrieval requires the `rank-bm25` package. "
                "Install dependencies (pip install -r requirements.txt)."
            ) from exc

        docs: list[str] = []
        with self._metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                title = str(obj.get("page_title") or "").strip()
                section = str(obj.get("section_title") or "").strip()
                body = str(obj.get("cleaned_text") or "")
                # Titles carry product names and head terms often missing from chunk bodies alone.
                composite = "\n".join(x for x in (title, section, body) if x)
                docs.append(composite)
                self._chunk_ids.append(str(obj.get("chunk_id") or ""))

        self._corpus = [tokenize_for_bm25(t) for t in docs]
        if not self._corpus:
            self._bm25 = None
            logger.warning("BM25: no documents in %s", self._metadata_path)
            return
        self._bm25 = BM25Okapi(self._corpus)
        logger.info("BM25 index ready: %d chunks from %s", len(self._corpus), self._metadata_path)

    def scores_for_query_tokens(self, query_tokens: list[str]) -> dict[str, float]:
        self._ensure_built()
        if self._bm25 is None or not query_tokens:
            return {}
        raw = self._bm25.get_scores(query_tokens)
        out: dict[str, float] = {}
        for i, sc in enumerate(raw):
            if i < len(self._chunk_ids):
                cid = self._chunk_ids[i]
                if cid:
                    out[cid] = float(sc)
        return out

    def score_dict_for_query(self, query: str) -> dict[str, float]:
        return self.scores_for_query_tokens(tokenize_for_bm25(query))
