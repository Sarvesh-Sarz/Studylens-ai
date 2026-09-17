"""Tests for src.ingestion.pdf_loader.

Test PDFs are generated on the fly with PyMuPDF so the suite has no
external file dependencies.
"""

from __future__ import annotations

import fitz
import pytest

from src.ingestion.pdf_loader import PDFLoadError, load_pdf


def _make_pdf(path, pages_text):
    """Create a simple PDF at `path` with one page per string in pages_text.

    An empty string produces a genuinely blank page (no extractable text).
    """
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_load_pdf_extracts_text_and_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, ["Page one content about deadlocks.", "Page two content about paging."])

    pages = load_pdf(pdf_path, source_name="sample.pdf")

    assert len(pages) == 2
    assert pages[0].page == 1
    assert pages[1].page == 2
    assert "deadlocks" in pages[0].text
    assert "paging" in pages[1].text
    assert all(p.source == "sample.pdf" for p in pages)


def test_load_pdf_uses_custom_source_name(tmp_path):
    pdf_path = tmp_path / "temp123.pdf"
    _make_pdf(pdf_path, ["Some content."])

    pages = load_pdf(pdf_path, source_name="Operating Systems.pdf")

    assert pages[0].source == "Operating Systems.pdf"


def test_load_pdf_skips_blank_pages_but_keeps_text_pages(tmp_path):
    pdf_path = tmp_path / "mixed.pdf"
    _make_pdf(pdf_path, ["Has text.", "", "Also has text."])

    pages = load_pdf(pdf_path, source_name="mixed.pdf")

    # The blank page (page 2) should be skipped; pages 1 and 3 remain,
    # with their original page numbers preserved.
    assert [p.page for p in pages] == [1, 3]


def test_load_pdf_raises_on_all_blank_pdf(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    _make_pdf(pdf_path, ["", ""])

    with pytest.raises(PDFLoadError):
        load_pdf(pdf_path, source_name="blank.pdf")


def test_load_pdf_raises_on_missing_file(tmp_path):
    with pytest.raises(PDFLoadError):
        load_pdf(tmp_path / "does_not_exist.pdf")


def test_load_pdf_raises_on_corrupted_file(tmp_path):
    bad_path = tmp_path / "corrupted.pdf"
    bad_path.write_bytes(b"not a real pdf file")

    with pytest.raises(PDFLoadError):
        load_pdf(bad_path, source_name="corrupted.pdf")
