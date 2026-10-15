"""Run the agentic pipeline and the baseline on the same questions.

Usage:
    python -m evaluation.benchmark_runner --dataset evaluation/datasets/sample.json
    python -m evaluation.benchmark_runner --dataset evaluation/datasets --limit 10
    python -m evaluation.benchmark_runner --systems agentic --types multi_hop unanswerable
"""
import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.agent.agent_loop import AgentLoop
from app.config import ROOT
from evaluation.baseline_rag import run_baseline
from evaluation.metrics import aggregate, aggregate_by_type, score_example

RESULTS_DIR = ROOT / "data" / "eval_results"
DEFAULT_DATASET = ROOT / "evaluation" / "datasets" / "sample.json"

METRIC_ROWS = [
    ("answer_correctness", "Answer correctness"),
    ("faithfulness", "Faithfulness"),
    ("retrieval_recall", "Retrieval recall"),
    ("abstention_accuracy", "Abstention accuracy"),
    ("abstain_rate_on_unanswerable", "Abstains on unanswerable"),
    ("false_abstain_rate_on_answerable", "False abstentions"),
    ("avg_steps", "Avg. steps"),
    ("avg_tokens", "Avg. tokens / question"),
    ("avg_seconds", "Avg. latency (s)"),
]


# ------------------------------------------------------------------ dataset
def _read_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data.get("examples", []) if isinstance(data, dict) else data


def load_dataset(path, limit: int | None = None, types: list[str] | None = None) -> list[dict]:
    """Load one .json/.jsonl file, or every such file in a directory."""
    path = Path(path)
    files = (
        sorted(p for p in path.iterdir() if p.suffix in {".json", ".jsonl"})
        if path.is_dir() else [path]
    )
    examples: list[dict] = []
    for f in files:
        examples.extend(_read_file(f))
    for i, ex in enumerate(examples, 1):
        ex.setdefault("id", f"q{i}")
        ex.setdefault("type", "single_hop")
        ex.setdefault("answerable", True)
        ex.setdefault("history", [])
    if types:
        examples = [e for e in examples if e["type"] in types]
    return examples[:limit] if limit else examples


# ------------------------------------------------------------------ runners
def agentic_runner():
    agent = AgentLoop()

    def run(example: dict) -> dict:
        r = agent.run(example["question"], example.get("history", []))
        return {
            "answer": r.answer,
            "evidence": getattr(r, "evidence", None),
            "steps": r.steps,
            "cost": r.cost,
            "trace": r.trace_path,
        }
    return run


def baseline_runner():
    def run(example: dict) -> dict:
        r = run_baseline(example["question"], example.get("history", []))
        return {"answer": r.answer, "evidence": r.evidence, "steps": r.steps, "cost": r.cost, "trace": None}
    return run


RUNNERS = {"baseline": baseline_runner, "agentic": agentic_runner}


# --------------------------------------------------------------- evaluation
def evaluate_system(name: str, runner, examples: list[dict]) -> dict:
    scores, records, errors = [], [], []
    for i, ex in enumerate(examples, 1):
        try:
            output = runner(ex)
            score = score_example(ex, output)
        except Exception as e:  # one bad question should not kill the run
            errors.append({"id": ex["id"], "error": f"{type(e).__name__}: {e}"})
            print(f"[{name}] {i}/{len(examples)} {ex['id']:<12} ERROR {type(e).__name__}: {e}")
            continue
        scores.append(score)
        records.append({
            "id": ex["id"], "type": ex["type"], "question": ex["question"],
            "answer": output["answer"], "trace": output.get("trace"), "score": asdict(score),
        })
        print(
            f"[{name}] {i}/{len(examples)} {ex['id']:<12} "
            f"correct={score.correctness:.1f} abstained={score.abstained} tokens={score.tokens}"
        )
    return {
        "system": name,
        "summary": aggregate(scores),
        "by_type": aggregate_by_type(scores),
        "errors": errors,
        "records": records,
    }


# ----------------------------------------------------------------- reporting
def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3f}"
    return str(v)


def markdown_table(results: list[dict]) -> str:
    names = [r["system"] for r in results]
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label in METRIC_ROWS:
        lines.append(f"| {label} | " + " | ".join(_fmt(r["summary"].get(key)) for r in results) + " |")
    return "\n".join(lines)


def by_type_table(results: list[dict], metric: str = "answer_correctness") -> str:
    names = [r["system"] for r in results]
    types = sorted({t for r in results for t in r["by_type"]})
    lines = [f"| Question type ({metric}) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for t in types:
        cells = [_fmt(r["by_type"].get(t, {}).get(metric)) for r in results]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(prefix: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic vs. baseline benchmark")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="JSON/JSONL file or a directory")
    parser.add_argument("--systems", nargs="+", choices=list(RUNNERS), default=list(RUNNERS))
    parser.add_argument("--limit", type=int, help="Max number of questions")
    parser.add_argument("--types", nargs="+", help="e.g. single_hop multi_hop unanswerable followup")
    args = parser.parse_args()

    examples = load_dataset(args.dataset, args.limit, args.types)
    if not examples:
        sys.exit("No examples found.")
    print(f"Loaded {len(examples)} questions from {args.dataset}\n")

    results = [evaluate_system(name, RUNNERS[name](), examples) for name in args.systems]

    print("\n" + markdown_table(results))
    print("\n" + by_type_table(results))
    errors = {r["system"]: len(r["errors"]) for r in results if r["errors"]}
    if errors:
        print(f"\nQuestions that errored (excluded from metrics): {errors}")
    path = save_results("benchmark", {"dataset": str(args.dataset), "results": results})
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()