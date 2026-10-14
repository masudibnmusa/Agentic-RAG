"""Web search fallback via Tavily (set TAVILY_API_KEY)."""
import requests

from app.config import settings
from app.retrieval.evidence_pool import Chunk

SPEC = {
    "name": "web_search",
    "description": "Search the public web. Use only when the document corpus lacks the answer.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def run(query: str, k: int = 5) -> list[Chunk]:
    if not settings.tavily_api_key:
        return []
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        json={"query": query, "max_results": k},
        timeout=20,
    )
    resp.raise_for_status()
    return [
        Chunk(
            id=r["url"],
            text=f"{r.get('title', '')}\n{r.get('content', '')}",
            source=r["url"],
            metadata={"web": True, "title": r.get("title", "")},
            score=float(r.get("score", 0.5)),
        )
        for r in resp.json().get("results", [])
    ]