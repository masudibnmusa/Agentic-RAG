"""Plan: standalone question + sub-questions."""
from app.config import settings
from app.query.decomposer import decompose
from app.query.rewriter import rewrite_standalone


def plan(question: str, history: list[dict]) -> tuple[str, list[str]]:
    standalone = rewrite_standalone(question, history)
    sub_questions = decompose(standalone) if settings.enable_decomposition else [standalone]
    return standalone, sub_questions