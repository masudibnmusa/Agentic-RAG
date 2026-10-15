"""Fixed retrieve-then-generate pipeline used as the comparison baseline.

One vector search with the raw question, then one generation call. No routing, rewriting,
grading, retries, or verification. Uses the same answer prompt as the agent so the comparison
isolates the agentic loop.

Quick test: python -m evaluation.baseline_rag "your question"
"""
import sys
from dataclasses import dataclass, field

from app.config import settings
from app.generation.answer_generator import generate_answer
from app.observability.cost_tracker import tracker
from app.retrieval.evidence_pool import EvidencePool
from app.tools import vector_search


@dataclass
class BaselineResult:
    answer: str
    evidence: list = field(default_factory=list)
    steps: int = 1
    cost: dict = field(default_factory=dict)


def run_baseline(question: str, history: list[dict] | None = None, k: int | None = None) -> BaselineResult:
    tracker.start()
    pool = EvidencePool()
    chunks = pool.add(vector_search.run(question, k=k or settings.top_k))  # raw question, no rewrite
    answer = generate_answer(question, history or [], chunks)
    evidence = [{"label": c.label, "source": c.source, "text": c.text} for c in chunks]
    return BaselineResult(answer=answer, evidence=evidence, steps=1, cost=tracker.summary())


if __name__ == "__main__":
    result = run_baseline(" ".join(sys.argv[1:]) or "What is this project about?")
    print(result.answer)
    print(result.cost)