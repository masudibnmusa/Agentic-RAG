"""Run state: question, sub-questions, evidence pool, attempts so far."""
from dataclasses import dataclass, field

from app.retrieval.evidence_pool import EvidencePool


@dataclass
class Attempt:
    sub_question: str
    query: str
    tools: list = field(default_factory=list)
    n_retrieved: int = 0
    n_relevant: int = 0
    sufficient: bool = False
    error: str | None = None


@dataclass
class AgentState:
    question: str
    history: list = field(default_factory=list)
    standalone: str = ""
    sub_questions: list = field(default_factory=list)
    source: str = "docs"
    pool: EvidencePool = field(default_factory=EvidencePool)
    attempts: list = field(default_factory=list)
    total_steps: int = 0


@dataclass
class AgentResult:
    answer: str
    abstained: bool
    sources: list
    steps: int
    cost: dict
    trace_id: str
    trace_path: str


def format_history(history: list[dict], limit: int = 6) -> str:
    if not history:
        return "(no prior conversation)"
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history[-limit:]]
    return "\n".join(lines)