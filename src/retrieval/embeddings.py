"""
Embedding generation, isolated from both the UI and the vector store.

Wraps a sentence-transformers model behind a small, stable interface
(`EmbeddingModel.encode`) so that:
  * the UI never talks to the embedding library directly, and
  * swapping the underlying model (e.g. to a larger or domain-specific
    model) only requires changing the EMBEDDING_MODEL env var -- no other
    module needs to change.
"""

from __future__ import annotations

import logging
import threading
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

_model_cache: dict[str, "EmbeddingModel"] = {}
_cache_lock = threading.Lock()


class EmbeddingError(Exception):
    """Raised when embedding generation fails (e.g. model fails to load)."""


class EmbeddingModel:
    """Thin wrapper around a sentence-transformers model.

    `all-MiniLM-L6-v2` is used by default: it is small (~80MB), fast on
    CPU, and produces strong general-purpose sentence embeddings (384-dim)
    for semantic search -- a good fit for a local, dependency-light RAG app.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = self._load_model(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    @staticmethod
    def _load_model(model_name: str):
        try:
            # Imported lazily so environments that only run non-embedding
            # code paths (e.g. some unit tests) don't pay the import cost.
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Run "
                "`pip install -r requirements.txt`."
            ) from exc

        try:
            return SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{model_name}'. Check your internet "
                f"connection (first run downloads the model) and the model name."
            ) from exc

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings into a (n, dim) float32 numpy array.

        Returns L2-normalized vectors so that inner-product search in FAISS
        is equivalent to cosine similarity.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

        return embeddings.astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        """Convenience helper to encode a single string (e.g. a query)."""
        return self.encode([text])[0]


def get_embedding_model(model_name: str) -> EmbeddingModel:
    """Return a cached EmbeddingModel for `model_name`, creating it if needed.

    Loading a sentence-transformers model is relatively expensive, so we
    cache one instance per model name for the lifetime of the process
    (Streamlit's session reruns would otherwise reload the model constantly).
    """
    with _cache_lock:
        if model_name not in _model_cache:
            logger.info("Loading embedding model '%s'...", model_name)
            _model_cache[model_name] = EmbeddingModel(model_name)
        return _model_cache[model_name]
