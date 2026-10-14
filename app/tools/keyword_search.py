import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.config import settings
from app.retrieval.evidence_pool import Chunk

SPEC = {
    "name": "keyword_search",
    "description": (
        "BM25 keyword search. Best for exact terms, IDs, error codes, product names, "
        "and rare words that semantic search may miss."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Keywords to match"}},
        "required": ["query"],
    },
}

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _load():
    path = settings.vector_db_dir / "bm25_corpus.json"
    if not path.exists():
        return None
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if not corpus:
        return None
    return BM25Okapi([tokenize(c["text"]) for c in corpus]), corpus


def clear_cache() -> None:
    _load.cache_clear()


def run(query: str, k: int | None = None) -> list[Chunk]:
    loaded = _load()
    if not loaded:
        return []
    k = k or settings.top_k
    bm25, corpus = loaded
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    best = max((scores[i] for i in top), default=0) or 1.0
    return [
        Chunk(
            id=corpus[i]["id"],
            text=corpus[i]["text"],
            source=corpus[i]["metadata"].get("source", ""),
            metadata=corpus[i]["metadata"],
            score=float(scores[i] / best),  # normalized to 0-1
        )
        for i in top
        if scores[i] > 0
    ]