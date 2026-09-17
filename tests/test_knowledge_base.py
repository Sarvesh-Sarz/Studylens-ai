"""End-to-end test for src.knowledge_base.KnowledgeBase.

Exercises the full ingest -> chunk -> embed -> index -> retrieve pipeline
against a real PDF (generated on the fly with PyMuPDF), using the same
FakeEmbeddingModel as test_retriever.py so this runs fast and without
downloading real model weights. This is the one test that proves the
wiring between every module in `src/` actually works together, not just
each module in isolation.
"""

from __future__ import annotations

import fitz
import pytest

from src.ingestion.pdf_loader import PDFLoadError
from src.knowledge_base import KnowledgeBase
from tests.test_retriever import FakeEmbeddingModel


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _kb(chunk_size=1000, chunk_overlap=100, top_k=3):
    return KnowledgeBase(
        embedding_model=FakeEmbeddingModel(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )


def test_add_document_indexes_pages_and_reports_chunk_count(tmp_path):
    pdf_path = tmp_path / "os.pdf"
    _make_pdf(
        pdf_path,
        [
            "Deadlock happens when processes wait forever for resources.",
            "Networking protocols like TCP and UDP move data between hosts.",
        ],
    )

    kb = _kb()
    progress_messages = []
    num_chunks = kb.add_document(pdf_path, source_name="OS.pdf", progress=progress_messages.append)

    assert num_chunks == 2
    assert not kb.is_empty
    assert "Knowledge base ready." in progress_messages[-1]


def test_stats_reflect_actual_indexed_data_not_fabricated(tmp_path):
    os_pdf = tmp_path / "os.pdf"
    _make_pdf(os_pdf, ["Deadlock content.", "Networking content."])
    db_pdf = tmp_path / "db.pdf"
    _make_pdf(db_pdf, ["Database transactions content."])

    kb = _kb()
    kb.add_document(os_pdf, source_name="OS.pdf")
    kb.add_document(db_pdf, source_name="DBMS.pdf")

    stats = kb.stats()
    assert stats.num_documents == 2
    assert stats.num_pages == 3  # 2 pages from OS.pdf + 1 from DBMS.pdf
    assert stats.num_chunks == 3
    assert set(stats.document_names) == {"OS.pdf", "DBMS.pdf"}


def test_retrieval_after_indexing_multiple_documents_returns_correct_source(tmp_path):
    os_pdf = tmp_path / "os.pdf"
    _make_pdf(os_pdf, ["Deadlock happens when processes wait forever."])
    db_pdf = tmp_path / "db.pdf"
    _make_pdf(db_pdf, ["Database transactions ensure ACID properties."])

    kb = _kb()
    kb.add_document(os_pdf, source_name="OS.pdf")
    kb.add_document(db_pdf, source_name="DBMS.pdf")

    result = kb.retriever.retrieve("What is a deadlock?")

    assert not result.is_empty
    assert result.chunks[0].chunk.source == "OS.pdf"
    assert result.chunks[0].chunk.page == 1


def test_add_document_raises_for_pdf_with_no_extractable_text(tmp_path):
    blank_pdf = tmp_path / "blank.pdf"
    _make_pdf(blank_pdf, ["", ""])

    kb = _kb()
    with pytest.raises(PDFLoadError):
        kb.add_document(blank_pdf, source_name="blank.pdf")

    assert kb.is_empty  # failed upload must not corrupt the index
