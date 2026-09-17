"""Tests for src.retrieval.vector_store.

Uses hand-constructed vectors (no real embedding model) so these tests run
fast and without any external model download.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.vector_store import StoredChunk, VectorStore, VectorStoreError


def _unit(vec):
    arr = np.array(vec, dtype="float32")
    return arr / np.linalg.norm(arr)


def test_empty_store_returns_no_results():
    store = VectorStore(dimension=3)
    assert store.is_empty
    results = store.search(_unit([1, 0, 0]), top_k=5)
    assert results == []


def test_add_and_search_returns_most_similar_chunk_first():
    store = VectorStore(dimension=3)
    chunks = [
        StoredChunk(text="about deadlocks", source="OS.pdf", page=1, chunk_id="a"),
        StoredChunk(text="about networking", source="CN.pdf", page=1, chunk_id="b"),
        StoredChunk(text="about databases", source="DBMS.pdf", page=1, chunk_id="c"),
    ]
    embeddings = np.array([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])
    store.add(embeddings, chunks)

    results = store.search(_unit([0.95, 0.05, 0]), top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "a"  # closest to [1,0,0]
    assert results[0].score > results[1].score


def test_search_respects_top_k_and_caps_at_available_chunks():
    store = VectorStore(dimension=2)
    chunks = [StoredChunk(text="x", source="doc.pdf", page=1, chunk_id="only")]
    store.add(np.array([_unit([1, 0])]), chunks)

    results = store.search(_unit([1, 0]), top_k=10)
    assert len(results) == 1


def test_add_raises_on_mismatched_embedding_and_chunk_counts():
    store = VectorStore(dimension=3)
    with pytest.raises(VectorStoreError):
        store.add(np.zeros((2, 3), dtype="float32"), [StoredChunk("t", "d.pdf", 1, "id")])


def test_add_raises_on_wrong_dimension():
    store = VectorStore(dimension=3)
    with pytest.raises(VectorStoreError):
        store.add(np.zeros((1, 4), dtype="float32"), [StoredChunk("t", "d.pdf", 1, "id")])


def test_sources_and_pages_per_source_across_multiple_documents():
    store = VectorStore(dimension=2)
    chunks = [
        StoredChunk(text="a", source="OS.pdf", page=1, chunk_id="1"),
        StoredChunk(text="b", source="OS.pdf", page=2, chunk_id="2"),
        StoredChunk(text="c", source="DBMS.pdf", page=1, chunk_id="3"),
    ]
    embeddings = np.array([_unit([1, 0]), _unit([0.9, 0.1]), _unit([0, 1])])
    store.add(embeddings, chunks)

    assert set(store.sources()) == {"OS.pdf", "DBMS.pdf"}
    assert store.pages_per_source() == {"OS.pdf": 2, "DBMS.pdf": 1}
    assert store.num_chunks == 3
