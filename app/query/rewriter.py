"""Standalone, keyword-rich queries."""
from app.agent.llm_client import get_llm
from app.agent.state import format_history
from app.config import settings

STANDALONE_SYSTEM = """Rewrite the user's latest question as a fully standalone question.
Resolve pronouns and references using the conversation. Keep the meaning identical.
Return only the rewritten question."""

RETRY_SYSTEM = """A search for a sub-question did not find enough evidence.
Write ONE new search query that is meaningfully different from the previous queries:
use different key terms, synonyms, or a narrower/broader angle, and address what is missing.
Return only the query."""


def rewrite_standalone(question: str, history: list[dict]) -> str:
    if not settings.enable_rewriting or not history:
        return question
    user = f"Conversation:\n{format_history(history)}\n\nLatest question: {question}"
    out = get_llm().ask(STANDALONE_SYSTEM, user, model=settings.grader_model, max_tokens=200)
    return out or question


def rewrite_for_retry(sub_question: str, previous_queries: list[str], missing: str) -> str:
    if not settings.enable_rewriting:
        return sub_question
    user = (
        f"Sub-question: {sub_question}\n"
        f"Previous queries: {previous_queries}\n"
        f"What is missing: {missing or 'unknown'}"
    )
    out = get_llm().ask(RETRY_SYSTEM, user, model=settings.grader_model, max_tokens=100)
    return out.strip('"') or sub_question