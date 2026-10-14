"""Multi-part question -> sub-queries."""
from app.agent.llm_client import get_llm
from app.config import settings

SYSTEM = """Decide whether the question must be split into independent sub-questions to be answered
by search (comparisons, multi-part questions, multi-hop questions). Each sub-question must be
standalone. If the question is simple, return it unchanged as the only item.
Return JSON: {"sub_questions": ["...", "..."]}"""


def decompose(question: str) -> list[str]:
    data = get_llm().ask_json(
        SYSTEM + f"\nUse at most {settings.max_subquestions} sub-questions.",
        question, model=settings.grader_model, max_tokens=300,
    )
    subs = [s for s in data.get("sub_questions", []) if isinstance(s, str) and s.strip()]
    return subs[: settings.max_subquestions] or [question]