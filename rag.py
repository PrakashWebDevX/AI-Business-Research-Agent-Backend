"""
rag.py

Retrieval-Augmented Generation (RAG) layer for the AI Business Research Agent.

Responsibilities:
    - Store uploaded documents (PDF/TXT/MD) as chunks in the existing
      SQLite database (reusing database.py's engine — no new database file).
    - Build a TF-IDF index over all stored chunks for fast, dependency-light
      retrieval (no external model downloads, no GPU/ONNX runtime — chosen
      deliberately for deployment stability on Railway).
    - Expose add_document() / search_documents() for tools.py's
      document_search_tool and backend/main.py's upload endpoint to use.

Design notes:
    - TF-IDF (via scikit-learn) is used instead of dense embeddings
      (e.g. sentence-transformers, fastembed) specifically to avoid any
      external model download at runtime — those add cold-start latency
      and a new failure mode (network access to HuggingFace, disk space
      for model caching) that isn't worth the retrieval-quality gain for
      a project at this scale. This can be swapped for dense embeddings
      later without changing the public API of this module.
    - The TF-IDF index is rebuilt in memory whenever a document is added
      or deleted, and lazily built on first search if not yet built. This
      is fine for a personal/portfolio-scale document set (tens to low
      hundreds of documents); a system with thousands of documents would
      want an incremental/persistent vector index instead.
    - Like the rest of this project's SQLite usage, uploaded documents do
      NOT survive a Railway redeploy (ephemeral filesystem) unless a
      persistent volume is attached. Fine for demo use; flagged here so
      it's not a surprise later.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List, Optional

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship

from database import Base, get_engine, get_session

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100    # characters of overlap between consecutive chunks
DEFAULT_TOP_K = 4      # number of chunks returned per search


# --------------------------------------------------------------------------- #
# Models (added to the same SQLite database as the rest of the app)
# --------------------------------------------------------------------------- #

class Document(Base):
    __tablename__ = "documents"

    document_id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    uploaded_at = Column(DateTime, server_default=func.now())

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document id={self.document_id} filename={self.filename!r}>"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.document_id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)

    document = relationship("Document", back_populates="chunks")


def init_rag_tables() -> None:
    """Create the documents/document_chunks tables if they don't exist yet."""
    Base.metadata.create_all(bind=get_engine(), tables=[Document.__table__, DocumentChunk.__table__])


# --------------------------------------------------------------------------- #
# Text extraction & chunking
# --------------------------------------------------------------------------- #

def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Extract plain text from an uploaded file's raw bytes.

    Supports .pdf (via pypdf) and .txt/.md (decoded directly).

    Raises:
        ValueError: If the file type is unsupported.
    """
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if lower_name.endswith((".txt", ".md")):
        return file_bytes.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type for '{filename}'. Use .pdf, .txt, or .md.")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks by character count.

    Overlap helps avoid losing context at chunk boundaries (e.g. a
    sentence split exactly at a chunk edge is still fully present in
    the adjacent chunk).
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# --------------------------------------------------------------------------- #
# In-memory TF-IDF index
# --------------------------------------------------------------------------- #

@dataclass
class SearchResult:
    document_id: int
    filename: str
    chunk_index: int
    content: str
    score: float


class _DocumentIndex:
    """Lazily-built, in-memory TF-IDF index over all stored document chunks."""

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._chunk_rows: List[dict] = []  # parallel to _matrix rows
        self._built = False

    def rebuild(self) -> None:
        """Rebuild the index from every chunk currently in the database."""
        with get_session() as session:
            rows = (
                session.query(DocumentChunk, Document)
                .join(Document, DocumentChunk.document_id == Document.document_id)
                .all()
            )

            self._chunk_rows = [
                {
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                }
                for chunk, doc in rows
            ]

        self._built = True

        if not self._chunk_rows:
            self._vectorizer = None
            self._matrix = None
            logger.info("Document index rebuilt: 0 chunks (index empty).")
            return

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([row["content"] for row in self._chunk_rows])
        logger.info(
            "Document index rebuilt: %d chunks from %d document(s).",
            len(self._chunk_rows),
            len({r["document_id"] for r in self._chunk_rows}),
        )

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[SearchResult]:
        if not self._built:
            self.rebuild()
        if self._vectorizer is None or not self._chunk_rows:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        top_indices = scores.argsort()[::-1][:top_k]

        return [
            SearchResult(
                document_id=self._chunk_rows[i]["document_id"],
                filename=self._chunk_rows[i]["filename"],
                chunk_index=self._chunk_rows[i]["chunk_index"],
                content=self._chunk_rows[i]["content"],
                score=float(scores[i]),
            )
            for i in top_indices
        ]


_index = _DocumentIndex()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def add_document(filename: str, file_bytes: bytes) -> dict:
    """
    Extract, chunk, and store a document, then refresh the search index.

    Args:
        filename: Original filename (used to detect .pdf vs .txt/.md and
            for display/citation purposes).
        file_bytes: Raw uploaded file content.

    Returns:
        A dict: {"document_id": int, "filename": str, "chunk_count": int}.

    Raises:
        ValueError: If the file type is unsupported or contains no
            extractable text.
    """
    text = extract_text(filename, file_bytes)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"No extractable text found in '{filename}'.")

    with get_session() as session:
        document = Document(filename=filename)
        session.add(document)
        session.flush()  # populate document.document_id

        for i, chunk_content in enumerate(chunks):
            session.add(DocumentChunk(document_id=document.document_id, chunk_index=i, content=chunk_content))

        session.commit()
        document_id = document.document_id

    _index.rebuild()
    logger.info("Added document %r (%d chunks).", filename, len(chunks))

    return {"document_id": document_id, "filename": filename, "chunk_count": len(chunks)}


def search_documents(query: str, top_k: int = DEFAULT_TOP_K) -> List[SearchResult]:
    """Search stored document chunks for the most relevant matches to a query."""
    return _index.search(query, top_k=top_k)


def list_documents() -> List[dict]:
    """List all uploaded documents (id, filename, upload time, chunk count)."""
    with get_session() as session:
        documents = session.query(Document).all()
        return [
            {
                "document_id": d.document_id,
                "filename": d.filename,
                "uploaded_at": d.uploaded_at,
                "chunk_count": len(d.chunks),
            }
            for d in documents
        ]


def delete_document(document_id: int) -> bool:
    """Delete a document and all its chunks. Returns True if it existed."""
    with get_session() as session:
        document = session.query(Document).filter_by(document_id=document_id).first()
        if document is None:
            return False
        session.delete(document)
        session.commit()

    _index.rebuild()
    return True