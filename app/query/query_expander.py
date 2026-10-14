"""Synonyms, acronym expansion, alternate phrasings."""
from app.agent.llm_client import get_llm
from app.config import settings

SYSTEM = """Produce alternative search queries for the given query: expand acronyms, use synonyms,
and rephrase. Return JSON: {"queries": ["...", "..."]}"""


def expand(query: str, n: int = 2) -> list[str]:
    data = get_llm().ask_json(SYSTEM + f"\nReturn at most {n}.", query, model=settings.grader_model, max_tokens=200)
    return [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:n]