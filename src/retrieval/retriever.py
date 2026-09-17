"""
Semantic retrieval: turns a user question into relevant document chunks.

This is the only module the rest of the app should call for retrieval --
it composes `embeddings.py` and `vector_store.py` so the UI never needs to
know how similarity search is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import RetrievedChunk, VectorStore


@dataclass
class RetrievalResult:
    """The outcome of running retrieval for one user query."""

    query: str
    chunks: List[RetrievedChunk]

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0


class Retriever:
    """Retrieves the most relevant chunks for a query from a VectorStore."""

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore, top_k: int, min_relevance_score: float = 0.0):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not -1.0 <= min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score must be between -1 and 1")

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Embed `query` and return the most relevant chunks across all docs.

        Only the top-k relevant chunks are returned -- the full document
        text is never sent to the LLM, keeping prompts small and answers
        grounded in the most relevant evidence.
        """
        k = top_k or self.top_k
        query = query.strip()

        if not query:
            return RetrievalResult(query=query, chunks=[])

        if self.vector_store.is_empty:
            return RetrievalResult(query=query, chunks=[])

        query_embedding = self.embedding_model.encode_one(query)
        results = self.vector_store.search(query_embedding, top_k=k)
        if self.min_relevance_score > 0.0:
            results = [r for r in results if r.score >= self.min_relevance_score]
        return RetrievalResult(query=query, chunks=results)
