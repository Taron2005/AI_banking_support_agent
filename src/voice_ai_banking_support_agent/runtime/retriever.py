from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from ..config import AppConfig
from ..indexing.bm25_index import ChunkBM25Index
from ..indexing.embedder import EmbedderConfig, EmbeddingModel
from ..indexing.vector_store import FaissVectorStore
from ..models import TopicLabel
from .bank_scope import should_diversify_across_banks
from .cross_encoder_rerank import cross_encoder_rerank
from .evidence_select import rerank_and_select
from .models import RetrievedChunk
from .runtime_config import RetrievalSettings

logger = logging.getLogger(__name__)


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    index_name: str
    top_k: int = 6
    topic: TopicLabel | None = None
    # None = all banks; one or more keys = restrict to those banks (OR).
    bank_keys: frozenset[str] | None = None


class RuntimeRetriever:
    """Retrieval wrapper used by runtime orchestration."""

    _embedders: ClassVar[dict[str, EmbeddingModel]] = {}
    _bm25_indexes: ClassVar[dict[str, ChunkBM25Index]] = {}

    def __init__(self, config: AppConfig, retrieval: RetrievalSettings | None = None) -> None:
        self._config = config
        self._retrieval = retrieval or RetrievalSettings()
        self._validated_index_names: set[str] = set()

    def _embedder(self) -> EmbeddingModel:
        key = self._config.embedding_model_name
        cached = RuntimeRetriever._embedders.get(key)
        if cached is None:
            cached = EmbeddingModel(
                EmbedderConfig(
                    model_name=key,
                    device="cpu",
                    batch_size=32,
                    normalize=True,
                )
            )
            RuntimeRetriever._embedders[key] = cached
        return cached

    def _assert_index_compatible(self, index_name: str, query_emb) -> None:  # noqa: ANN001
        if index_name in self._validated_index_names:
            return
        index_dir = Path(self._config.index_dir) / index_name
        info_path = index_dir / "index_info.json"
        if not info_path.is_file():
            logger.warning(
                "Missing %s — cannot verify embedding_dim / embedding_model_name against config.",
                info_path,
            )
            self._validated_index_names.add(index_name)
            return
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in %s: %s", info_path, exc)
            self._validated_index_names.add(index_name)
            return
        dim = int(info.get("embedding_dim") or 0)
        qdim = int(query_emb.shape[1])
        if dim and qdim != dim:
            raise RuntimeError(
                f"Embedding dimension mismatch for index {index_name!r}: query vectors have dim={qdim} "
                f"but index_info.json declares embedding_dim={dim}. Rebuild the index with "
                f"embedding model {self._config.embedding_model_name!r} or fix EMBEDDING_MODEL_NAME."
            )
        built = str(info.get("embedding_model_name") or "").strip()
        cfg_name = (self._config.embedding_model_name or "").strip()
        if built and cfg_name and built != cfg_name:
            logger.warning(
                "Embedding model mismatch for index %r: index built with %r but runtime config uses %r. "
                "Retrieval scores may be meaningless — rebuild the index or align EMBEDDING_MODEL_NAME.",
                index_name,
                built,
                cfg_name,
            )
        self._validated_index_names.add(index_name)

    def _store(self, index_name: str) -> FaissVectorStore:
        index_dir = Path(self._config.index_dir) / index_name
        return FaissVectorStore(
            index_path=index_dir / "faiss.index",
            metadata_path=index_dir / "metadata.jsonl",
        )

    def _bm25_for_index(self, index_name: str) -> ChunkBM25Index:
        cached = RuntimeRetriever._bm25_indexes.get(index_name)
        if cached is None:
            meta = (Path(self._config.index_dir) / index_name) / "metadata.jsonl"
            cached = ChunkBM25Index(metadata_path=meta)
            RuntimeRetriever._bm25_indexes[index_name] = cached
        return cached

    def retrieve(self, req: RetrievalRequest) -> list[RetrievedChunk]:
        query_emb = self._embedder().embed_query(req.query)
        self._assert_index_compatible(req.index_name, query_emb)
        factor = max(4, int(self._retrieval.faiss_candidate_pool_factor))
        pool_k = max(48, req.top_k * factor)
        hits = self._store(req.index_name).search(
            query_embedding=query_emb,
            top_k=pool_k,
            topic_filter=req.topic,
            bank_keys=req.bank_keys,
        )
        rough = [RetrievedChunk(score=h.score, chunk=h.doc) for h in hits]

        if self._retrieval.hybrid_bm25 and rough:
            try:
                bm25_map = self._bm25_for_index(req.index_name).score_dict_for_query(req.query)
                dscores = [float(c.score) for c in rough]
                bscores = [bm25_map.get(c.chunk.chunk_id, 0.0) for c in rough]
                dn = _min_max_normalize(dscores)
                bn = _min_max_normalize(bscores)
                w = float(self._retrieval.hybrid_dense_weight)
                fused: list[RetrievedChunk] = []
                for i, c in enumerate(rough):
                    ns = w * dn[i] + (1.0 - w) * bn[i]
                    fused.append(RetrievedChunk(score=ns, chunk=c.chunk))
                rough = fused
            except Exception:
                logger.exception("BM25 hybrid fusion failed; using dense scores only.")

        rerank_k = req.top_k
        if self._retrieval.cross_encoder_enabled and self._retrieval.cross_encoder_model:
            rerank_k = max(rerank_k, min(self._retrieval.cross_encoder_top_n, len(rough)))

        diversify = should_diversify_across_banks(req.bank_keys)
        per_url = (
            int(self._retrieval.diversify_max_per_source_url)
            if diversify and self._retrieval.diversify_max_per_source_url > 0
            else None
        )
        picked = rerank_and_select(
            rough,
            req.query,
            rerank_k,
            diversify_banks=diversify,
            max_per_bank=max(1, int(self._retrieval.diversify_max_per_bank)),
            max_per_source_url=per_url,
        )

        if (
            self._retrieval.cross_encoder_enabled
            and self._retrieval.cross_encoder_model
            and len(picked) > req.top_k
        ):
            try:
                return cross_encoder_rerank(
                    req.query,
                    picked,
                    model_name=self._retrieval.cross_encoder_model,
                    top_k=req.top_k,
                )
            except Exception:
                logger.exception("Cross-encoder rerank failed; using lexical rerank only.")
                return picked[: req.top_k]

        if len(picked) > req.top_k:
            return picked[: req.top_k]
        return picked
