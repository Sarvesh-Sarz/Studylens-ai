"""Tests for src.retrieval.retriever.

A tiny fake embedding model is used in place of sentence-transformers so
these tests run instantly and without downloading any model weights.
"""

from __future__ import annotations

import numpy as np

from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import StoredChunk, VectorStore


class FakeEmbeddingModel:
    """Deterministic 'embedding' model: maps known strings to fixed vectors."""

    dimension = 3

    _lookup = {
        "deadlock": [1.0, 0.0, 0.0],
        "networking": [0.0, 1.0, 0.0],
        "database": [0.0, 0.0, 1.0],
    }

    def _vec(self, text: str):
        for key, vec in self._lookup.items():
            if key in text.lower():
                arr = np.array(vec, dtype="float32")
                return arr / np.linalg.norm(arr)
        # default: neutral vector
        arr = np.array([1.0, 1.0, 1.0], dtype="float32")
        return arr / np.linalg.norm(arr)

    def encode(self, texts):
        return np.array([self._vec(t) for t in texts], dtype="float32")

    def encode_one(self, text):
        return self._vec(text)


def _build_store():
    store = VectorStore(dimension=3)
    model = FakeEmbeddingModel()
    chunks = [
        StoredChunk(text="Deadlock happens when processes wait forever.", source="OS.pdf", page=41, chunk_id="1"),
        StoredChunk(text="Networking protocols like TCP and UDP.", source="CN.pdf", page=10, chunk_id="2"),
        StoredChunk(text="Database transactions ensure ACID properties.", source="DBMS.pdf", page=12, chunk_id="3"),
    ]
    embeddings = model.encode([c.text for c in chunks])
    store.add(embeddings, chunks)
    return model, store


def test_retrieve_returns_relevant_chunk_for_matching_query():
    model, store = _build_store()
    retriever = Retriever(model, store, top_k=2)

    result = retriever.retrieve("What is a deadlock?")

    assert not result.is_empty
    assert result.chunks[0].chunk.source == "OS.pdf"


def test_retrieve_on_empty_store_returns_empty_result():
    model = FakeEmbeddingModel()
    store = VectorStore(dimension=3)
    retriever = Retriever(model, store, top_k=5)

    result = retriever.retrieve("What is a deadlock?")

    assert result.is_empty
    assert result.chunks == []


def test_retrieve_on_empty_query_returns_empty_result():
    model, store = _build_store()
    retriever = Retriever(model, store, top_k=5)

    result = retriever.retrieve("   ")

    assert result.is_empty


def test_retrieve_respects_top_k_override():
    model, store = _build_store()
    retriever = Retriever(model, store, top_k=5)

    result = retriever.retrieve("database networking deadlock", top_k=1)

    assert len(result.chunks) == 1
