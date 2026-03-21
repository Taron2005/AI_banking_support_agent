from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig
from ..indexing.embedder import EmbedderConfig, EmbeddingModel
from ..indexing.vector_store import FaissVectorStore
from ..models import TopicLabel
from .models import RetrievedChunk


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    index_name: str
    top_k: int = 6
    topic: TopicLabel | None = None
    bank_filter: str | None = None


class RuntimeRetriever:
    """Retrieval wrapper used by runtime orchestration."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._embedder = EmbeddingModel(
            EmbedderConfig(
                model_name=config.embedding_model_name,
                device="cpu",
                batch_size=32,
                normalize=True,
            )
        )

    def _store(self, index_name: str) -> FaissVectorStore:
        index_dir = Path(self._config.index_dir) / index_name
        return FaissVectorStore(
            index_path=index_dir / "faiss.index",
            metadata_path=index_dir / "metadata.jsonl",
        )

    def retrieve(self, req: RetrievalRequest) -> list[RetrievedChunk]:
        query_emb = self._embedder.embed_query(req.query)
        hits = self._store(req.index_name).search(
            query_embedding=query_emb,
            top_k=req.top_k,
            topic_filter=req.topic,
            bank_filter=req.bank_filter,
        )
        return [RetrievedChunk(score=h.score, chunk=h.doc) for h in hits]

