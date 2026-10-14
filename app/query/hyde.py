"""Optional HyDE: embed a hypothetical answer instead of the raw query."""
from app.agent.llm_client import get_llm
from app.config import settings

SYSTEM = """Write a short, plausible passage (2-4 sentences) that would answer the question,
as it might appear in documentation. It does not need to be factually correct.
Return only the passage."""


def hypothetical_passage(question: str) -> str:
    try:
        return get_llm().ask(SYSTEM, question, model=settings.grader_model, max_tokens=200) or question
    except Exception:
        return question