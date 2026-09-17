"""
FAISS-backed vector store.

Stores chunk embeddings plus their associated text and metadata, and
supports similarity search across chunks from multiple uploaded PDFs.
Kept fully independent of Streamlit so it can be unit tested and reused.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised for FAISS index build/search failures."""


@dataclass
class StoredChunk:
    """A chunk of text plus its metadata, as retained by the vector store."""

    text: str
    source: str
    page: int
    chunk_id: str


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity search, with its relevance score."""

    chunk: StoredChunk
    score: float  # cosine similarity in [-1, 1] (typically [0, 1] for text)


class VectorStore:
    """A local, in-memory FAISS index over document chunks.

    Uses an inner-product index (IndexFlatIP) over L2-normalized vectors,
    which is mathematically equivalent to cosine-similarity search and is
    exact (no approximation) -- appropriate for the modest corpus sizes of
    a few uploaded PDFs.
    """

    def __init__(self, dimension: int):
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError(
                "faiss-cpu is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self._faiss = faiss
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: List[StoredChunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[StoredChunk]) -> None:
        """Add a batch of embeddings and their corresponding chunks.

        Args:
            embeddings: (n, dimension) float32 array of L2-normalized vectors.
            chunks: List of StoredChunk, one per row of `embeddings`.
        """
        if embeddings.shape[0] != len(chunks):
            raise VectorStoreError(
                f"Embedding count ({embeddings.shape[0]}) does not match "
                f"chunk count ({len(chunks)})."
            )
        if embeddings.shape[0] == 0:
            return
        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.dimension}."
            )

        try:
            self._index.add(embeddings)
        except Exception as exc:
            raise VectorStoreError(f"Failed to add vectors to FAISS index: {exc}") from exc

        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        """Return the top_k most similar chunks to `query_embedding`.

        Args:
            query_embedding: A single (dimension,) L2-normalized vector.
            top_k: Maximum number of results to return.

        Returns:
            A list of RetrievedChunk, sorted by descending similarity.
            Empty if the index has no vectors yet.
        """
        if self.is_empty:
            return []

        top_k = min(top_k, len(self._chunks))
        query = np.expand_dims(query_embedding.astype("float32"), axis=0)

        try:
            scores, indices = self._index.search(query, top_k)
        except Exception as exc:
            raise VectorStoreError(f"FAISS search failed: {exc}") from exc

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score)))
        return results

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def sources(self) -> List[str]:
        """Return the distinct document names currently indexed."""
        seen = []
        for c in self._chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen

    def pages_per_source(self) -> dict:
        """Return {source_name: number_of_distinct_pages_indexed}."""
        pages: dict[str, set] = {}
        for c in self._chunks:
            pages.setdefault(c.source, set()).add(c.page)
        return {source: len(page_set) for source, page_set in pages.items()}
