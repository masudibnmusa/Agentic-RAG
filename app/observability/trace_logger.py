"""Records every query, tool call, grade, and decision for a single run."""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from app.config import settings


class TraceLogger:
    def __init__(self, question: str):
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.question = question
        self.events: list[dict] = []
        self._t0 = time.time()

    def log(self, event: str, **data) -> None:
        self.events.append({"t": round(time.time() - self._t0, 2), "event": event, **data})

    def save(self, extra: dict | None = None) -> Path:
        path = settings.traces_dir / f"{self.run_id}.json"
        payload = {
            "run_id": self.run_id,
            "question": self.question,
            "events": self.events,
            **(extra or {}),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path