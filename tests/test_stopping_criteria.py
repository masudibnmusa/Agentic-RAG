import pytest

from app.agent.stopping_criteria import check_stop
from app.config import settings


def decide(**overrides):
    args = dict(sufficient=False, sub_steps=1, total_steps=1, tokens_used=100, no_progress_streak=0)
    args.update(overrides)
    return check_stop(**args)


def test_continues_when_nothing_triggers():
    d = decide()
    assert d.stop is False
    assert d.reason == ""


def test_stops_when_evidence_is_sufficient():
    d = decide(sufficient=True)
    assert d.stop is True
    assert d.reason == "sufficient_evidence"


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"sub_steps": 3}, "subquestion_step_limit"),
        ({"total_steps": 10}, "total_step_limit"),
        ({"tokens_used": 60000}, "token_budget"),
        ({"no_progress_streak": 2}, "no_progress"),
    ],
)
def test_stops_at_each_limit(overrides, reason):
    d = decide(**overrides)
    assert d.stop is True
    assert d.reason == reason


def test_a_single_unproductive_attempt_does_not_stop_the_loop():
    assert decide(no_progress_streak=1).stop is False


def test_sufficient_evidence_wins_over_every_limit():
    d = decide(sufficient=True, sub_steps=99, total_steps=99, tokens_used=10**9, no_progress_streak=99)
    assert d.reason == "sufficient_evidence"


def test_limits_follow_settings(monkeypatch):
    monkeypatch.setattr(settings, "max_steps_per_subquestion", 5)
    assert decide(sub_steps=3).stop is False
    assert decide(sub_steps=5).stop is True