"""Metrics: answer correctness, faithfulness, retrieval recall, abstention accuracy.

Correctness, abstention detection, and faithfulness use an LLM judge (the stronger planner model).
Judge calls run AFTER each system's cost is captured, so they never inflate reported tokens/latency.

Definitions
- answer_correctness: answerable questions -> judged against the reference (1 / 0.5 / 0);
                      unanswerable questions -> 1 if the system abstained, else 0.
- faithfulness:       share of the answer's claims supported by the evidence the system used
                      (skipped when the system abstained or used no evidence).
- retrieval_recall:   share of gold source files present in the evidence the system used.
- abstention_accuracy: share of questions where "abstained" matches "not answerable".
"""
from dataclasses import dataclass
from statistics import mean

from app.agent.llm_client import get_llm
from app.config import settings

JUDGE_SYSTEM = """You are a strict evaluator of question-answering systems.
You receive a question, a reference answer (or NONE if the question cannot be answered from the
corpus), and a system answer.

Return JSON: {"abstained": true|false, "correctness": "correct|partial|incorrect", "reason": "short"}

- abstained: true if the system answer declines to answer, or says the information was not found
  or is not in the sources, instead of giving a substantive answer. An answer that gives a
  substantive answer and merely notes a minor gap is NOT an abstention.
- correctness (meaningful only when a reference answer is given):
  "correct"   = conveys the same key facts as the reference (wording may differ; extra correct detail is fine)
  "partial"   = covers some but not all key facts, or has minor errors
  "incorrect" = wrong, contradicts the reference, or abstains."""

FAITH_SYSTEM = """You check whether an answer is supported by evidence.
Count the distinct factual claims in the answer, and how many of them are clearly supported by the
evidence. Judge only against the evidence provided. Do not accept inferences the evidence does not state.
Return JSON: {"total_claims": <int>, "supported_claims": <int>}"""

CORRECTNESS_VALUE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


@dataclass
class ExampleScore:
    id: str
    type: str
    answerable: bool
    abstained: bool
    abstention_correct: bool
    correctness: float
    faithfulness: float | None
    retrieval_recall: float | None
    steps: int | None
    tokens: int | None
    seconds: float | None


def _norm(source: str) -> str:
    return source.replace("\\", "/").strip()


def judge_answer(question: str, reference: str | None, answer: str) -> dict:
    user = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference or 'NONE (unanswerable from the corpus)'}\n\n"
        f"System answer:\n{answer}"
    )
    return get_llm().ask_json(JUDGE_SYSTEM, user, model=settings.planner_model, max_tokens=300)


def judge_faithfulness(answer: str, evidence: list[dict]) -> float | None:
    docs = "\n\n".join(f"[{e.get('label', '?')}] {e['text']}" for e in evidence)
    data = get_llm().ask_json(
        FAITH_SYSTEM, f"Evidence:\n{docs}\n\nAnswer:\n{answer}",
        model=settings.planner_model, max_tokens=300,
    )
    total, supported = data.get("total_claims"), data.get("supported_claims")
    if not isinstance(total, int) or not isinstance(supported, int) or total <= 0:
        return None
    return min(supported, total) / total


def retrieval_recall(gold: list[str], evidence: list[dict] | None) -> float | None:
    if not gold or evidence is None:
        return None
    gold_set = {_norm(g) for g in gold}
    found = {_norm(e["source"]) for e in evidence}
    return len(gold_set & found) / len(gold_set)


def score_example(example: dict, output: dict) -> ExampleScore:
    """`output` = {"answer", "evidence": [{"label","source","text"}] | None, "steps", "cost"}"""
    answerable = example.get("answerable", True)
    judged = judge_answer(
        example["question"], example.get("reference_answer") if answerable else None, output["answer"]
    )
    abstained = bool(judged.get("abstained", False))

    if answerable:
        correctness = 0.0 if abstained else CORRECTNESS_VALUE.get(judged.get("correctness"), 0.0)
    else:
        correctness = 1.0 if abstained else 0.0

    evidence = output.get("evidence")
    faithfulness = None
    if evidence and not abstained:
        faithfulness = judge_faithfulness(output["answer"], evidence)

    recall = retrieval_recall(example.get("relevant_sources", []), evidence) if answerable else None
    cost = output.get("cost") or {}
    return ExampleScore(
        id=example["id"],
        type=example.get("type", "single_hop"),
        answerable=answerable,
        abstained=abstained,
        abstention_correct=(abstained == (not answerable)),
        correctness=correctness,
        faithfulness=faithfulness,
        retrieval_recall=recall,
        steps=output.get("steps"),
        tokens=cost.get("total_tokens"),
        seconds=cost.get("wall_seconds"),
    )


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 3) if vals else None


def aggregate(scores: list[ExampleScore]) -> dict:
    answerable = [s for s in scores if s.answerable]
    unanswerable = [s for s in scores if not s.answerable]
    return {
        "n": len(scores),
        "answer_correctness": _mean(s.correctness for s in scores),
        "faithfulness": _mean(s.faithfulness for s in scores),
        "retrieval_recall": _mean(s.retrieval_recall for s in scores),
        "abstention_accuracy": _mean(1.0 if s.abstention_correct else 0.0 for s in scores),
        "abstain_rate_on_unanswerable": _mean(1.0 if s.abstained else 0.0 for s in unanswerable),
        "false_abstain_rate_on_answerable": _mean(1.0 if s.abstained else 0.0 for s in answerable),
        "avg_steps": _mean(s.steps for s in scores),
        "avg_tokens": _mean(s.tokens for s in scores),
        "avg_seconds": _mean(s.seconds for s in scores),
    }


def aggregate_by_type(scores: list[ExampleScore]) -> dict[str, dict]:
    groups: dict[str, list[ExampleScore]] = {}
    for s in scores:
        groups.setdefault(s.type, []).append(s)
    return {t: aggregate(group) for t, group in sorted(groups.items())}