"""
KnowledgeBase: orchestrates ingestion -> chunking -> embedding -> indexing.

This module exists so `app.py` (the Streamlit UI) never has to import or
know about PyMuPDF, sentence-transformers, or FAISS directly -- it just
calls `KnowledgeBase.add_document(...)` and `KnowledgeBase.stats()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.pdf_loader import PDFLoadError, load_pdf
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import StoredChunk, VectorStore

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str], None]]


@dataclass
class KnowledgeBaseStats:
    """Summary statistics for the sidebar, computed from actual indexed data."""

    num_documents: int
    num_pages: int
    num_chunks: int
    document_names: List[str] = field(default_factory=list)


class KnowledgeBase:
    """Holds the indexed state for all documents uploaded in a session."""

    def __init__(self, embedding_model: EmbeddingModel, chunk_size: int, chunk_overlap: int, top_k: int):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = VectorStore(dimension=embedding_model.dimension)
        self.retriever = Retriever(embedding_model, self.vector_store, top_k=top_k)
        self._pages_by_source: dict[str, int] = {}

    def add_document(self, file_path: str | Path, source_name: str, progress: ProgressCallback = None) -> int:
        """Ingest, chunk, embed, and index a single PDF.

        Args:
            file_path: Path to the PDF on disk (e.g. a temp upload path).
            source_name: Display name for the document (e.g. original filename).
            progress: Optional callback invoked with human-readable status
                strings, used by the UI to show a live processing state.

        Returns:
            The number of chunks added to the index.

        Raises:
            PDFLoadError: if the PDF is empty, corrupted, or has no
                extractable text.
        """

        def report(msg: str) -> None:
            if progress:
                progress(msg)
            logger.info(msg)

        report(f"Reading '{source_name}'...")
        pages = load_pdf(file_path, source_name=source_name)

        report("Extracting text...")
        # (Extraction already happened in load_pdf; this stage is reported
        # separately to match the UX spec's step-by-step processing view.)

        report("Creating chunks...")
        chunks: List[Chunk] = chunk_document(pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise PDFLoadError(f"'{source_name}' produced no usable text chunks.")

        report("Generating embeddings...")
        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.encode(texts)

        report("Building knowledge index...")
        stored = [StoredChunk(text=c.text, source=c.source, page=c.page, chunk_id=c.chunk_id) for c in chunks]
        self.vector_store.add(embeddings, stored)

        self._pages_by_source[source_name] = len({p.page for p in pages})
        report("Knowledge base ready.")
        return len(chunks)

    def stats(self) -> KnowledgeBaseStats:
        """Compute knowledge-base statistics from the actual indexed data."""
        sources = self.vector_store.sources()
        return KnowledgeBaseStats(
            num_documents=len(sources),
            num_pages=sum(self._pages_by_source.values()),
            num_chunks=self.vector_store.num_chunks,
            document_names=sources,
        )

    @property
    def is_empty(self) -> bool:
        return self.vector_store.is_empty
