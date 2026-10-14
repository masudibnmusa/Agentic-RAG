"""Optional reranking. Set RERANKER=cross_encoder or cohere. Scores are normalized to 0-1."""
import math

from app.config import settings
from app.retrieval.evidence_pool import Chunk

_cross_encoder = None


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def rerank(query: str, chunks: list[Chunk], top_n: int | None = None) -> list[Chunk]:
    top_n = top_n or settings.rerank_top_n
    if not chunks or settings.reranker == "none":
        return chunks

    if settings.reranker == "cross_encoder":
        global _cross_encoder
        if _cross_encoder is None:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = _cross_encoder.predict([(query, c.text) for c in chunks])
        for chunk, score in zip(chunks, scores):
            chunk.score = _sigmoid(float(score))
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_n]

    if settings.reranker == "cohere":
        import cohere
        client = cohere.Client(settings.cohere_api_key)
        res = client.rerank(
            model="rerank-v3.5", query=query, documents=[c.text for c in chunks], top_n=top_n
        )
        out = []
        for r in res.results:
            chunks[r.index].score = float(r.relevance_score)
            out.append(chunks[r.index])
        return out

    return chunks