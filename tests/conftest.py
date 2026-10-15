"""Shared fixtures: a scriptable fake LLM, deterministic settings, and chunk helpers.
No test here calls the Anthropic API or touches the vector DB."""
from types import SimpleNamespace

import pytest

from app.config import settings
from app.retrieval.evidence_pool import Chunk, EvidencePool


class _Handler:
    """Static value, a list (returned in order; the last item repeats), or a callable(user)."""

    def __init__(self, value):
        self.value, self.i = value, 0

    def next(self, user: str):
        v = self.value
        if callable(v):
            return v(user)
        if isinstance(v, list):
            item = v[min(self.i, len(v) - 1)]
            self.i += 1
            return item
        return v


class FakeLLM:
    """Drop-in for LLMClient. Responses are matched by a substring of the system prompt."""

    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []  # (method, system, user)
        self._json: dict[str, _Handler] = {}
        self._text: dict[str, _Handler] = {}
        self.tool_call = ("vector_search", {"query": "test query"})

    def on_json(self, key: str, value):
        self._json[key] = _Handler(value)
        return self

    def on_text(self, key: str, value):
        self._text[key] = _Handler(value)
        return self

    @staticmethod
    def _respond(handlers, system, user, default):
        for key, handler in handlers.items():
            if key in system:
                return handler.next(user)
        return default

    def ask(self, system, user, model=None, max_tokens=800):
        self.calls.append(("ask", system, user))
        return self._respond(self._text, system, user, "")

    def ask_json(self, system, user, model=None, max_tokens=800):
        self.calls.append(("ask_json", system, user))
        return self._respond(self._json, system, user, {})

    def complete(self, system, messages, model=None, max_tokens=1024, tools=None, tool_choice=None):
        self.calls.append(("complete", system, messages[-1]["content"]))
        name, args = self.tool_call
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=name, input=dict(args))])

    def count(self, key: str) -> int:
        """How many calls had `key` in their system prompt."""
        return sum(1 for _, system, _ in self.calls if key in system)


@pytest.fixture
def fake_llm(monkeypatch):
    from app.agent import llm_client

    fake = FakeLLM()
    monkeypatch.setattr(llm_client, "LLMClient", lambda: fake)
    llm_client.get_llm.cache_clear()  # get_llm is lru_cached; every module shares this wrapper
    yield fake
    llm_client.get_llm.cache_clear()


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    defaults = dict(
        enable_routing=True, enable_rewriting=True, enable_decomposition=True,
        enable_grading=True, enable_verification=True, use_hyde=False,
        use_query_expansion=False, reranker="none", grader_mode="llm",
        relevance_threshold=0.5, enabled_sources=["docs"],
        max_steps_per_subquestion=3, max_total_steps=10, max_tokens_budget=60000,
        max_revisions=2, no_progress_patience=2, traces_dir=tmp_path,
    )
    for key, value in defaults.items():
        monkeypatch.setattr(settings, key, value)


@pytest.fixture
def make_chunks():
    """make_chunks(("text", score?), ...) -> (pool, labeled_chunks). Labels are S1, S2, ..."""
    def _make(*specs, pool=None):
        pool = pool or EvidencePool()
        chunks = pool.add([
            Chunk(
                id=f"c{i}", text=spec[0], source=f"doc{i}.md",
                score=spec[1] if len(spec) > 1 else 0.5,
            )
            for i, spec in enumerate(specs, 1)
        ])
        return pool, chunks
    return _make