"""Is the evidence enough to answer fully?"""
from dataclasses import dataclass

from app.agent.llm_client import get_llm
from app.config import settings
from app.grading.prompt_templates import SUFFICIENCY_SYSTEM
from app.retrieval.evidence_pool import Chunk


@dataclass
class Sufficiency:
    sufficient: bool
    missing: str = ""


def grade_sufficiency(question: str, chunks: list[Chunk]) -> Sufficiency:
    if not chunks:
        return Sufficiency(False, "no relevant evidence found yet")
    if not settings.enable_grading:
        return Sufficiency(True)
    user = f"Question: {question}\n\n" + "\n".join(c.render(900) for c in chunks[:8])
    data = get_llm().ask_json(SUFFICIENCY_SYSTEM, user, model=settings.grader_model, max_tokens=250)
    return Sufficiency(bool(data.get("sufficient", False)), str(data.get("missing", "")))