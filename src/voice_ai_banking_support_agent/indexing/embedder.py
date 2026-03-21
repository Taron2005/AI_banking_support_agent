from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedderConfig:
    """Embedding model settings."""

    model_name: str
    batch_size: int = 32
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    normalize: bool = True


class EmbeddingModel:
    """Configurable embedding model wrapper."""

    def __init__(self, config: EmbedderConfig) -> None:
        self._config = config
        self._model = None

    def _get_model(self):
        """Lazy import + lazy model loading."""

        if self._model is not None:
            return self._model

        # IMPORTANT:
        # Some environments have TensorFlow installed, and `transformers` can attempt
        # to import TF during `sentence_transformers` initialization (even if we only
        # intend to run on CPU for embeddings).
        #
        # Your terminal log shows TF crashing due to protobuf descriptor issues.
        # Disabling TF imports makes embeddings robust and avoids that failure mode.
        #
        # We set these environment variables *before* importing sentence_transformers.
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

        # Fallback: if protobuf-related issues still surface via transitive imports,
        # use pure-Python protobuf implementation.
        # This is slower, but prevents hard crashes.
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

        from sentence_transformers import SentenceTransformer  # lazy import

        logger.info("Loading embedding model: %s", self._config.model_name)
        self._model = SentenceTransformer(self._config.model_name, device=self._config.device)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors.

        Returns:
            Float32 numpy array of shape (n_texts, embedding_dim).
        """

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        return arr

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query into shape (1, embedding_dim)."""

        vec = self.embed_texts([query])
        return vec

