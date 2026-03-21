from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import DocumentMetadata, TopicLabel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieval hit containing score and chunk metadata."""

    score: float
    doc: DocumentMetadata


class FaissVectorStore:
    """
    Local FAISS vector index with persisted metadata.

    Design:
    - vectors are stored in `faiss.index`
    - metadata mapping is stored in JSONL (`metadata.jsonl`)
    - FAISS IDs map 1:1 to metadata row order

    Retrieval filtering (topic/bank) is done after FAISS search so we can add
    hybrid/BM25 later without changing the index format.
    """

    def __init__(self, *, index_path: Path, metadata_path: Path) -> None:
        self._index_path = index_path
        self._metadata_path = metadata_path

        self._index = None
        self._metadata: list[DocumentMetadata] | None = None

    @staticmethod
    def _faiss():
        import faiss  # type: ignore[import-not-found]

        return faiss

    def _load_index(self):
        if self._index is not None:
            return self._index
        faiss = self._faiss()
        logger.info("Loading FAISS index: %s", self._index_path)
        self._index = faiss.read_index(str(self._index_path))
        return self._index

    def _load_metadata(self) -> list[DocumentMetadata]:
        if self._metadata is not None:
            return self._metadata

        logger.info("Loading metadata: %s", self._metadata_path)
        docs: list[DocumentMetadata] = []
        with self._metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                docs.append(DocumentMetadata.model_validate(obj))
        self._metadata = docs
        return docs

    @staticmethod
    def build_and_save(
        *,
        embeddings: np.ndarray,
        docs: list[DocumentMetadata],
        index_dir: Path,
        index_name: str,
    ) -> Path:
        """
        Build a FAISS index and persist it to disk.

        Returns:
            Path to the saved FAISS index file.
        """

        if embeddings.shape[0] != len(docs):
            raise ValueError("Embeddings row count must match docs count.")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D: (n_docs, dim).")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        faiss = FaissVectorStore._faiss()
        dim = int(embeddings.shape[1])

        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "faiss.index"
        metadata_path = index_dir / "metadata.jsonl"
        tmp_index_path = index_dir / "faiss.index.tmp"
        tmp_metadata_path = index_dir / "metadata.jsonl.tmp"

        # With normalized embeddings, IndexFlatIP approximates cosine similarity via inner product.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(tmp_index_path))

        with tmp_metadata_path.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(d.model_dump_json(ensure_ascii=False) + "\n")
        tmp_index_path.replace(index_path)
        tmp_metadata_path.replace(metadata_path)

        index_info = {
            "index_name": index_name,
            "embedding_dim": dim,
            "vector_count": len(docs),
        }
        (index_dir / "index_info.json").write_text(
            json.dumps(index_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return index_path

    def search(
        self,
        *,
        query_embedding: np.ndarray,
        top_k: int,
        topic_filter: TopicLabel | None = None,
        bank_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Search vectors and return top hits.

        Filtering:
        - FAISS returns nearest vectors; we apply topic/bank filtering by reading metadata.
        """

        if top_k <= 0:
            return []
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("query_embedding must have shape (1, dim).")

        query_embedding = query_embedding.astype(np.float32)

        index = self._load_index()
        metadata = self._load_metadata()
        if hasattr(index, "ntotal") and int(index.ntotal) != len(metadata):
            raise RuntimeError(
                f"Index/metadata mismatch: vectors={int(index.ntotal)} metadata_rows={len(metadata)}. "
                "Rebuild index to restore consistency."
            )

        # Fetch extra candidates to compensate for metadata filtering (topic/bank).
        candidates = max(top_k * 8, top_k)
        scores, ids = index.search(query_embedding, candidates)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            doc = metadata[int(idx)]

            if topic_filter is not None and doc.topic != topic_filter:
                continue
            if bank_filter is not None:
                bank_filter_lower = bank_filter.lower()
                if doc.bank_name.lower() != bank_filter_lower and doc.bank_key.lower() != bank_filter_lower:
                    continue

            results.append(RetrievalResult(score=float(score), doc=doc))
            if len(results) >= top_k:
                break

        return results

