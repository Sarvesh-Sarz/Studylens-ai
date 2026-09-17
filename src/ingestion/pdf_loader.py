"""
PDF text extraction using PyMuPDF (fitz).

Responsible only for turning a PDF file into page-level text records.
Chunking is handled separately in `chunker.py` so this module stays
focused and easy to unit test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be opened or read."""


@dataclass
class PageText:
    """Text extracted from a single PDF page."""

    source: str  # original filename
    page: int  # 1-indexed page number
    text: str


def load_pdf(file_path: Union[str, Path], source_name: str | None = None) -> List[PageText]:
    """Extract text from every page of a PDF file.

    Args:
        file_path: Path to the PDF file on disk.
        source_name: Display name to record as the source (defaults to the
            file's own name). Useful when the file is stored under a
            temporary path but should be attributed to its original filename.

    Returns:
        A list of PageText records, one per page that contains extractable
        text. Pages with no extractable text are skipped (but logged), which
        is common for scanned/image-only PDFs.

    Raises:
        PDFLoadError: if the file cannot be opened, is corrupted, or is not
            a valid PDF.
    """
    path = Path(file_path)
    display_name = source_name or path.name

    if not path.exists():
        raise PDFLoadError(f"File not found: {path}")

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # PyMuPDF raises various fitz/RuntimeError types
        raise PDFLoadError(f"Could not open '{display_name}': the file may be corrupted "
                            f"or is not a valid PDF.") from exc

    if doc.is_encrypted:
        # Try an empty password first (some PDFs are "encrypted" but openable).
        if not doc.authenticate(""):
            doc.close()
            raise PDFLoadError(f"'{display_name}' is password-protected and cannot be read.")

    pages: List[PageText] = []
    try:
        if doc.page_count == 0:
            raise PDFLoadError(f"'{display_name}' has no pages.")

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            text = text.strip()
            if text:
                pages.append(PageText(source=display_name, page=page_index + 1, text=text))
            else:
                logger.info(
                    "No extractable text on page %d of '%s' (likely a scanned image).",
                    page_index + 1,
                    display_name,
                )
    finally:
        doc.close()

    if not pages:
        raise PDFLoadError(
            f"'{display_name}' contains no extractable text. It may be a scanned/"
            f"image-only PDF, which would require OCR (not supported in this version)."
        )

    return pages
