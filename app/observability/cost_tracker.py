"""Tracks tokens and latency per question. (Module-level singleton; not thread-safe.)"""
import time
from dataclasses import dataclass, field


@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    llm_seconds: float = 0.0
    started_at: float = field(default_factory=time.time)
    by_model: dict = field(default_factory=dict)

    def start(self) -> None:
        self.input_tokens = self.output_tokens = self.calls = 0
        self.llm_seconds = 0.0
        self.by_model = {}
        self.started_at = time.time()

    def record(self, model: str, input_tokens: int, output_tokens: int, seconds: float) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        self.llm_seconds += seconds
        m = self.by_model.setdefault(model, {"calls": 0, "input": 0, "output": 0})
        m["calls"] += 1
        m["input"] += input_tokens
        m["output"] += output_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> dict:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_seconds": round(self.llm_seconds, 2),
            "wall_seconds": round(time.time() - self.started_at, 2),
            "by_model": self.by_model,
        }


tracker = CostTracker()