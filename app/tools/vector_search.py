from app.retrieval.evidence_pool import Chunk
from app.retrieval.vector_store import get_store

SPEC = {
    "name": "vector_search",
    "description": (
        "Semantic search over document chunks. Best for conceptual questions, "
        "paraphrases, and how/why queries."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Standalone search query"}},
        "required": ["query"],
    },
}


def run(query: str, k: int | None = None, where: dict | None = None) -> list[Chunk]:
    return get_store().query(query, k=k, where=where)