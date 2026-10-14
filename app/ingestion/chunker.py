"""Paragraph-aware chunking with character overlap."""
import re

from app.ingestion.loader import Document


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if len(p) > size:
            pieces.extend(p[i:i + size] for i in range(0, len(p), size))
        else:
            pieces.append(p)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{piece}".strip() if tail else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: Document, size: int, overlap: int) -> list[tuple[str, str, dict]]:
    """Returns (chunk_id, text, metadata) tuples."""
    out = []
    for i, text in enumerate(chunk_text(doc.text, size, overlap)):
        out.append((f"{doc.doc_id}::{i}", text, {**doc.metadata, "chunk_index": i}))
    return out