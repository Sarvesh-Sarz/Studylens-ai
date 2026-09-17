"""
Text chunking with preserved source/page metadata.

Splitting strategy: a character-based sliding window with configurable
overlap, applied per-page. We chunk per-page (rather than concatenating an
entire document first) so that every chunk can be attributed to a single,
correct page number -- this is what makes accurate citations possible later
in the pipeline. The trade-off is that a concept spanning a page boundary
may be split across two chunks; the configurable overlap mitigates this by
carrying trailing context into the next chunk.

Splitting attempts to break on paragraph/sentence boundaries where possible
so chunks read as coherent text rather than being cut mid-word.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List

from src.ingestion.pdf_loader import PageText

# Prefer breaking chunks at these boundaries, in order of preference.
_BREAK_PATTERNS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    """A single retrievable unit of text with full source metadata."""

    text: str
    source: str
    page: int
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_metadata(self) -> dict:
        """Return the metadata dict (without the text body) for storage."""
        return {"source": self.source, "page": self.page, "chunk_id": self.chunk_id}


def _find_break_point(text: str, target: int) -> int:
    """Find a natural break point at or before `target` characters.

    Falls back to a hard cut at `target` if no natural boundary is found
    within a reasonable lookback window.
    """
    if target >= len(text):
        return len(text)

    lookback_window = max(50, target // 4)
    window_start = max(0, target - lookback_window)

    for pattern in _BREAK_PATTERNS:
        idx = text.rfind(pattern, window_start, target)
        if idx != -1:
            return idx + len(pattern)

    return target


def chunk_page_text(
    page_text: PageText,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """Split a single page's text into overlapping chunks.

    Args:
        page_text: The extracted text for one PDF page.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of characters of overlap carried between
            consecutive chunks (must be smaller than chunk_size).

    Returns:
        A list of Chunk objects, each tagged with the page's source and
        page number.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = re.sub(r"[ \t]+", " ", page_text.text).strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [Chunk(text=text, source=page_text.source, page=page_text.page)]

    chunks: List[Chunk] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        target_end = min(start + chunk_size, text_len)
        end = _find_break_point(text, target_end) if target_end < text_len else text_len
        if end <= start:
            end = target_end  # safety net against zero-progress loops

        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=page_text.source, page=page_text.page))

        if end >= text_len:
            break

        start = max(end - chunk_overlap, start + 1)  # guarantee forward progress

    return chunks


def chunk_document(
    pages: List[PageText],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """Chunk every page of a document, preserving per-page metadata.

    Args:
        pages: Ordered list of PageText records for one document.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.

    Returns:
        A flat list of Chunk objects across all pages of the document.
    """
    all_chunks: List[Chunk] = []
    for page in pages:
        all_chunks.extend(chunk_page_text(page, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
