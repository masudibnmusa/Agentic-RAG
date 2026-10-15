from app.config import settings
from app.query.rewriter import rewrite_for_retry, rewrite_standalone

HISTORY = [
    {"role": "user", "content": "What is the Pro plan rate limit?"},
    {"role": "assistant", "content": "1,000 requests per minute."},
]


def test_no_history_returns_question_without_calling_llm(fake_llm):
    assert rewrite_standalone("What is the limit?", []) == "What is the limit?"
    assert fake_llm.calls == []


def test_disabled_returns_question_unchanged(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "enable_rewriting", False)
    assert rewrite_standalone("And for Team?", HISTORY) == "And for Team?"
    assert fake_llm.calls == []


def test_resolves_follow_up_using_history(fake_llm):
    fake_llm.on_text("standalone question", "What is the Team plan rate limit?")
    out = rewrite_standalone("And for Team?", HISTORY)
    assert out == "What is the Team plan rate limit?"
    prompt = fake_llm.calls[-1][2]
    assert "Pro plan rate limit" in prompt  # history was sent to the model
    assert "And for Team?" in prompt


def test_falls_back_to_original_when_llm_returns_nothing(fake_llm):
    fake_llm.on_text("standalone question", "")
    assert rewrite_standalone("And for Team?", HISTORY) == "And for Team?"


def test_retry_rewrite_returns_new_query_and_strips_quotes(fake_llm):
    fake_llm.on_text("did not find enough evidence", '"pro tier requests per minute"')
    out = rewrite_for_retry("Pro plan rate limit", ["Pro plan rate limit"], "numeric limit")
    assert out == "pro tier requests per minute"


def test_retry_prompt_includes_previous_queries_and_the_gap(fake_llm):
    fake_llm.on_text("did not find enough evidence", "new query")
    rewrite_for_retry("Pro plan rate limit", ["first attempt query"], "the numeric limit")
    prompt = fake_llm.calls[-1][2]
    assert "first attempt query" in prompt
    assert "the numeric limit" in prompt


def test_retry_disabled_returns_subquestion(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "enable_rewriting", False)
    assert rewrite_for_retry("Pro plan rate limit", ["x"], "y") == "Pro plan rate limit"
    assert fake_llm.calls == []


def test_retry_falls_back_to_subquestion_on_empty_output(fake_llm):
    fake_llm.on_text("did not find enough evidence", "")
    assert rewrite_for_retry("Pro plan rate limit", ["x"], "y") == "Pro plan rate limit"