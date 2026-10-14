"""Enough evidence, step/cost limit, no-progress detection."""
from dataclasses import dataclass

from app.config import settings


@dataclass
class StopDecision:
    stop: bool
    reason: str = ""


def check_stop(
    *, sufficient: bool, sub_steps: int, total_steps: int, tokens_used: int, no_progress_streak: int
) -> StopDecision:
    if sufficient:
        return StopDecision(True, "sufficient_evidence")
    if sub_steps >= settings.max_steps_per_subquestion:
        return StopDecision(True, "subquestion_step_limit")
    if total_steps >= settings.max_total_steps:
        return StopDecision(True, "total_step_limit")
    if tokens_used >= settings.max_tokens_budget:
        return StopDecision(True, "token_budget")
    if no_progress_streak >= settings.no_progress_patience:
        return StopDecision(True, "no_progress")
    return StopDecision(False)