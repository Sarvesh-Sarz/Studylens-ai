"""Tests for src.ingestion.chunker."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_document, chunk_page_text
from src.ingestion.pdf_loader import PageText


def test_short_page_produces_single_chunk():
    page = PageText(source="doc.pdf", page=1, text="Short text that fits in one chunk.")
    chunks = chunk_page_text(page, chunk_size=1000, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0].text == page.text
    assert chunks[0].source == "doc.pdf"
    assert chunks[0].page == 1
    assert chunks[0].chunk_id  # non-empty


def test_long_page_is_split_into_multiple_chunks_with_metadata_preserved():
    long_text = ("Deadlock occurs when processes wait on each other. " * 50).strip()
    page = PageText(source="Operating Systems.pdf", page=42, text=long_text)

    chunks = chunk_page_text(page, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    for c in chunks:
        assert c.source == "Operating Systems.pdf"
        assert c.page == 42
        assert c.chunk_id
        assert len(c.text) > 0


def test_chunk_ids_are_unique():
    long_text = ("Sentence about networking protocols. " * 60).strip()
    page = PageText(source="Computer Networks.pdf", page=5, text=long_text)

    chunks = chunk_page_text(page, chunk_size=150, chunk_overlap=30)
    ids = [c.chunk_id for c in chunks]

    assert len(ids) == len(set(ids))


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    page = PageText(source="doc.pdf", page=1, text="some text")
    with pytest.raises(ValueError):
        chunk_page_text(page, chunk_size=100, chunk_overlap=100)


def test_chunk_document_preserves_page_numbers_across_multiple_pages():
    pages = [
        PageText(source="DBMS.pdf", page=1, text="Transactions are units of work."),
        PageText(source="DBMS.pdf", page=2, text="ACID properties ensure reliability."),
    ]

    chunks = chunk_document(pages, chunk_size=1000, chunk_overlap=100)

    assert {c.page for c in chunks} == {1, 2}
    assert all(c.source == "DBMS.pdf" for c in chunks)


def test_empty_page_produces_no_chunks():
    page = PageText(source="doc.pdf", page=1, text="   ")
    chunks = chunk_page_text(page, chunk_size=100, chunk_overlap=10)
    assert chunks == []


def test_to_metadata_contains_expected_keys():
    page = PageText(source="doc.pdf", page=3, text="content")
    chunk = chunk_page_text(page, chunk_size=100, chunk_overlap=10)[0]
    meta = chunk.to_metadata()

    assert set(meta.keys()) == {"source", "page", "chunk_id"}
    assert meta["source"] == "doc.pdf"
    assert meta["page"] == 3
