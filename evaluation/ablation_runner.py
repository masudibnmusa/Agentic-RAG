"""Toggle rewriting, grading, verification (and more) to measure each component's contribution.

Runs the full agent first, then the agent with ONE component disabled at a time.

Usage:
    python -m evaluation.ablation_runner                                  # full + every ablation
    python -m evaluation.ablation_runner --disable rewriting verification
    python -m evaluation.ablation_runner --dataset evaluation/datasets --limit 20
"""
import argparse
import sys
from contextlib import contextmanager

from app.config import settings
from evaluation.benchmark_runner import (
    DEFAULT_DATASET,
    agentic_runner,
    by_type_table,
    evaluate_system,
    load_dataset,
    markdown_table,
    save_results,
)

ABLATIONS = {
    "routing": {"enable_routing": False},
    "rewriting": {"enable_rewriting": False},
    "decomposition": {"enable_decomposition": False},
    "grading": {"enable_grading": False},
    "verification": {"enable_verification": False},
}

DELTA_KEYS = ["answer_correctness", "faithfulness", "retrieval_recall", "abstention_accuracy", "avg_tokens"]


@contextmanager
def override(**changes):
    """Temporarily set fields on the global settings object."""
    old = {k: getattr(settings, k) for k in changes}
    for k, v in changes.items():
        setattr(settings, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(settings, k, v)


def delta_table(results: list[dict]) -> str:
    """Each ablation minus the full agent (first result). Negative quality delta = the component helps."""
    full = results[0]["summary"]
    lines = ["| Ablation | " + " | ".join(DELTA_KEYS) + " |", "|---|" + "---|" * len(DELTA_KEYS)]
    for r in results[1:]:
        cells = []
        for k in DELTA_KEYS:
            a, b = r["summary"].get(k), full.get(k)
            cells.append("-" if a is None or b is None else f"{a - b:+.2f}")
        lines.append(f"| {r['system']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--disable", nargs="*", choices=list(ABLATIONS), help="Components to ablate (default: all)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--types", nargs="+")
    args = parser.parse_args()

    examples = load_dataset(args.dataset, args.limit, args.types)
    if not examples:
        sys.exit("No examples found.")
    names = args.disable or list(ABLATIONS)

    runner = agentic_runner()
    results = [evaluate_system("full", runner, examples)]
    for name in names:
        with override(**ABLATIONS[name]):
            results.append(evaluate_system(f"no_{name}", runner, examples))

    print("\n" + markdown_table(results))
    print("\n" + by_type_table(results))
    print("\nDelta vs. full agent:\n" + delta_table(results))
    print(
        "\nWhere to look: rewriting -> followup questions, decomposition -> multi_hop, "
        "verification -> faithfulness, grading -> recall and tokens, routing -> tokens."
    )
    path = save_results("ablation", {"dataset": str(args.dataset), "results": results})
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()