"""Decide: is retrieval needed at all, and which source?"""
from dataclasses import dataclass

from app.agent.llm_client import get_llm
from app.agent.state import format_history
from app.config import settings

SYSTEM = """You route user questions for a retrieval-augmented assistant.

Decide:
- needs_retrieval: false ONLY for casual chat, greetings, simple math, or universal general
  knowledge that does not depend on any specific document, product, or dataset. If unsure, true.
- source: one of the available sources:
  docs = the private document corpus (default)
  sql  = structured/numeric data lookups (counts, totals, records)
  web  = current public information unlikely to be in the corpus

Return JSON: {"needs_retrieval": true|false, "source": "docs|sql|web", "reason": "short"}"""


@dataclass
class Route:
    needs_retrieval: bool
    source: str
    reason: str


def route_question(question: str, history: list[dict]) -> Route:
    if not settings.enable_routing:
        return Route(True, "docs", "routing disabled")
    user = (
        f"Available sources: {', '.join(settings.enabled_sources)}\n\n"
        f"Conversation:\n{format_history(history)}\n\nQuestion: {question}"
    )
    data = get_llm().ask_json(SYSTEM, user, model=settings.grader_model, max_tokens=200)
    source = data.get("source", "docs")
    if source not in settings.enabled_sources:
        source = "docs"
    return Route(bool(data.get("needs_retrieval", True)), source, data.get("reason", ""))