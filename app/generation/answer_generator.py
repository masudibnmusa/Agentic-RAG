"""Grounded answer with inline citations."""
from app.agent.llm_client import get_llm
from app.agent.state import format_history
from app.config import settings
from app.generation.prompt_templates import ANSWER_SYSTEM
from app.retrieval.evidence_pool import Chunk


def generate_answer(question: str, history: list[dict], chunks: list[Chunk]) -> str:
    evidence = "\n".join(c.render() for c in chunks)
    user = (
        f"Conversation:\n{format_history(history)}\n\n"
        f"<evidence>\n{evidence}\n</evidence>\n\nQuestion: {question}"
    )
    return get_llm().ask(ANSWER_SYSTEM, user, model=settings.planner_model, max_tokens=1200)