"""Vector + BM25 combined with reciprocal rank fusion."""
from app.config import settings
from app.retrieval.evidence_pool import Chunk
from app.tools import keyword_search, vector_search

SPEC = {
    "name": "hybrid_search",
    "description": (
        "Combined semantic + keyword search with rank fusion. A good default when "
        "unsure which single strategy fits."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def run(query: str, k: int | None = None, rrf_k: int = 60) -> list[Chunk]:
    k = k or settings.top_k
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for results in (vector_search.run(query, k=k * 2), keyword_search.run(query, k=k * 2)):
        for rank, chunk in enumerate(results):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1 / (rrf_k + rank + 1)
            by_id.setdefault(chunk.id, chunk)

    max_possible = 2 / (rrf_k + 1)  # rank 1 in both lists
    ranked = sorted(fused, key=fused.get, reverse=True)[:k]
    out = []
    for cid in ranked:
        chunk = by_id[cid]
        chunk.score = fused[cid] / max_possible
        out.append(chunk)
    return out