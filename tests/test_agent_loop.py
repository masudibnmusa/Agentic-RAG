"""Integration tests of the full controller with a fake LLM and a stubbed search tool."""
import json
from pathlib import Path

import pytest

from app.agent.agent_loop import AgentLoop
from app.config import settings
from app.retrieval.evidence_pool import Chunk


class ToolStub:
    """Replaces run_tool. Each call returns the next list of chunks (or [] when exhausted)."""

    def __init__(self, *results):
        self.queue = list(results)
        self.calls = []

    def __call__(self, name, args):
        self.calls.append((name, args))
        return (self.queue.pop(0) if self.queue else []), None


def refund_chunk(i: int = 0) -> Chunk:
    return Chunk(
        id=f"refunds.md::{i}",
        text="Refunds are available within 30 days of purchase.",
        source="refunds.md",
        score=0.9,
    )


def configure_defaults(fake, answer="Refunds are available within 30 days of purchase [S1]."):
    fake.on_json("route user questions", {"needs_retrieval": True, "source": "docs", "reason": "test"})
    fake.on_json("split into independent sub-questions", {"sub_questions": ["What is the refund window?"]})
    fake.on_json("grade search results", {"relevant": ["S1"]})
    fake.on_json("enough to FULLY answer", {"sufficient": True, "missing": ""})
    fake.on_text("using ONLY the evidence", answer)
    fake.on_json("atomic factual claims", {"claims": [
        {"text": "Refunds are available within 30 days.", "citations": ["S1"]}
    ]})
    fake.on_json("verify claims against evidence", {"verdicts": [{"id": 1, "verdict": "supported", "reason": ""}]})


def events(result, name):
    data = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
    return [e for e in data["events"] if e["event"] == name]


@pytest.fixture
def wire(fake_llm, monkeypatch):
    def _wire(*results):
        stub = ToolStub(*results)
        monkeypatch.setattr("app.agent.agent_loop.run_tool", stub)
        monkeypatch.setattr("app.agent.agent_loop.tool_specs", lambda sources: [])
        return stub
    return _wire


def test_answers_directly_when_no_retrieval_is_needed(fake_llm, wire):
    stub = wire()
    fake_llm.on_json("route user questions", {"needs_retrieval": False, "source": "docs", "reason": "chat"})
    fake_llm.on_text("helpful assistant", "Hello! How can I help?")
    result = AgentLoop().run("hi there")
    assert result.answer == "Hello! How can I help?"
    assert result.steps == 0
    assert result.abstained is False
    assert stub.calls == []


def test_happy_path_returns_grounded_answer_with_sources(fake_llm, wire):
    configure_defaults(fake_llm)
    stub = wire([refund_chunk()])
    result = AgentLoop().run("What is the refund window?")
    assert result.answer == "Refunds are available within 30 days of purchase [S1]."
    assert result.abstained is False
    assert result.steps == 1
    assert len(stub.calls) == 1
    assert result.sources[0]["source"] == "refunds.md"
    assert Path(result.trace_path).exists()
    assert len(events(result, "verify")) == 1


def test_follow_up_question_is_rewritten_using_history(fake_llm, wire):
    configure_defaults(fake_llm)
    fake_llm.on_text("standalone question", "What is the refund window for monthly plans?")
    wire([refund_chunk()])
    history = [
        {"role": "user", "content": "What is the refund window?"},
        {"role": "assistant", "content": "30 days."},
    ]
    result = AgentLoop().run("And for monthly plans?", history)
    plan = events(result, "plan")[0]
    assert plan["standalone"] == "What is the refund window for monthly plans?"


def test_retries_with_a_rewritten_query_when_first_search_misses(fake_llm, wire):
    configure_defaults(fake_llm, answer="Refunds are available within 30 days of purchase [S2].")
    fake_llm.on_json("atomic factual claims", {"claims": [
        {"text": "Refunds are available within 30 days.", "citations": ["S2"]}
    ]})
    fake_llm.on_json("grade search results", [{"relevant": []}, {"relevant": ["S2"]}])
    fake_llm.on_text("did not find enough evidence", "refund window days")
    miss = Chunk(id="shipping.md::0", text="Shipping is free.", source="shipping.md", score=0.3)
    wire([miss], [refund_chunk()])

    result = AgentLoop().run("What is the refund window?")

    attempts = events(result, "attempt")
    assert len(attempts) == 2
    assert attempts[0]["relevant"] == 0
    assert attempts[1]["relevant"] == 1
    assert result.steps == 2
    assert result.abstained is False


def test_abstains_when_nothing_relevant_is_found(fake_llm, wire):
    configure_defaults(fake_llm)
    stub = wire()  # every search returns nothing
    result = AgentLoop().run("What is the refund window?")
    assert result.abstained is True
    assert "couldn't find" in result.answer
    assert result.steps == 2  # stopped by no-progress detection, not by exhausting the budget
    assert len(stub.calls) == 2
    assert fake_llm.count("using ONLY the evidence") == 0  # no draft was written


def test_revises_answer_when_a_claim_is_unsupported(fake_llm, wire):
    configure_defaults(fake_llm, answer="Refunds are available within 60 days [S1].")
    fake_llm.on_json("verify claims against evidence", [
        {"verdicts": [{"id": 1, "verdict": "unsupported", "reason": "evidence says 30"}]},
        {"verdicts": [{"id": 1, "verdict": "supported", "reason": ""}]},
    ])
    fake_llm.on_json("fix an answer", {
        "action": "revised",
        "answer": "Refunds are available within 30 days of purchase [S1].",
        "query": "",
    })
    wire([refund_chunk()])

    result = AgentLoop().run("What is the refund window?")

    assert result.answer == "Refunds are available within 30 days of purchase [S1]."
    assert result.abstained is False
    assert events(result, "revise")[0]["action"] == "revised"
    assert len(events(result, "verify")) == 2


def test_skips_verification_when_disabled(fake_llm, wire, monkeypatch):
    monkeypatch.setattr(settings, "enable_verification", False)
    configure_defaults(fake_llm)
    wire([refund_chunk()])
    result = AgentLoop().run("What is the refund window?")
    assert result.answer.startswith("Refunds are available")
    assert fake_llm.count("verify claims against evidence") == 0
    assert events(result, "verify") == []


def test_respects_the_total_step_budget(fake_llm, wire, monkeypatch):
    monkeypatch.setattr(settings, "max_total_steps", 1)
    configure_defaults(fake_llm)
    fake_llm.on_json("split into independent sub-questions", {"sub_questions": ["Q one?", "Q two?"]})
    fake_llm.on_json("enough to FULLY answer", {"sufficient": False, "missing": "more detail"})
    stub = wire([refund_chunk()], [refund_chunk(1)])

    result = AgentLoop().run("Compare one and two")

    assert len(stub.calls) == 1  # second sub-question was never researched
    assert result.steps == 1
    assert "total_step_limit" in [e["reason"] for e in events(result, "stop")]