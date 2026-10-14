"""Are these chunks relevant to the (sub-)question?"""
from app.agent.llm_client import get_llm
from app.config import settings
from app.grading.prompt_templates import RELEVANCE_SYSTEM
from app.retrieval.evidence_pool import Chunk


def grade_relevance(question: str, chunks: list[Chunk]) -> set[str]:
    """Returns the ids of relevant chunks."""
    if not chunks:
        return set()
    if not settings.enable_grading:
        return {c.id for c in chunks}
    if settings.grader_mode == "reranker":
        # Use with RERANKER=cross_encoder|cohere so scores are meaningful
        return {c.id for c in chunks if c.score >= settings.relevance_threshold}

    user = f"Question: {question}\n\n" + "\n".join(c.render(700) for c in chunks)
    data = get_llm().ask_json(RELEVANCE_SYSTEM, user, model=settings.grader_model, max_tokens=300)
    labels = {l for l in data.get("relevant", []) if isinstance(l, str)}
    return {c.id for c in chunks if c.label in labels}