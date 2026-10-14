"""Read a full document or the neighboring chunks around a promising hit."""
from app.retrieval.evidence_pool import Chunk
from app.retrieval.vector_store import get_store

SPEC = {
    "name": "fetch_document",
    "description": (
        "Read more context from a known source file: either neighboring chunks around "
        "a chunk_index, or the first chunks of the document."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Source file name/path"},
            "chunk_index": {"type": "integer", "description": "Center chunk (optional)"},
            "window": {"type": "integer", "description": "Neighbors on each side (default 1)"},
        },
        "required": ["source"],
    },
}


def run(source: str, chunk_index: int | None = None, window: int = 1) -> list[Chunk]:
    chunks = get_store().get(where={"source": source})
    chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
    if chunk_index is not None:
        lo, hi = max(0, chunk_index - window), chunk_index + window
        chunks = [c for c in chunks if lo <= c.metadata.get("chunk_index", 0) <= hi]
    else:
        chunks = chunks[:10]
    for c in chunks:
        c.score = 0.5
    return chunks