"""Step-by-step view of a run.  Usage: python -m app.observability.trace_viewer [trace.json]"""
import json
import sys
from pathlib import Path

from app.config import settings


def _short(value, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def view(path: Path) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"Run:      {data['run_id']}")
    print(f"Question: {data['question']}\n")
    for event in data["events"]:
        event = dict(event)
        t = event.pop("t")
        name = event.pop("event")
        print(f"[{t:6.2f}s] {name.upper()}")
        for key, value in event.items():
            print(f"           {key}: {_short(value)}")
    if "cost" in data:
        print("\nCost:", json.dumps(data["cost"]))
    if "answer" in data:
        print("\nFinal answer:\n" + data["answer"])


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        traces = sorted(settings.traces_dir.glob("*.json"))
        if not traces:
            sys.exit("No traces found.")
        path = traces[-1]
    view(path)


if __name__ == "__main__":
    main()