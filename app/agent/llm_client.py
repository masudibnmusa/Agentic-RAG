"""Thin Claude API wrapper with native tool use, JSON helpers, and cost tracking."""
import json
import re
import time
from functools import lru_cache

import anthropic

from app.config import settings
from app.observability.cost_tracker import tracker


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


class LLMClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def complete(self, system, messages, model=None, max_tokens=1024, tools=None, tool_choice=None):
        model = model or settings.planner_model
        kwargs = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        start = time.time()
        resp = self.client.messages.create(**kwargs)
        tracker.record(model, resp.usage.input_tokens, resp.usage.output_tokens, time.time() - start)
        return resp

    def ask(self, system: str, user: str, model: str | None = None, max_tokens: int = 800) -> str:
        resp = self.complete(system, [{"role": "user", "content": user}], model, max_tokens)
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def ask_json(self, system: str, user: str, model: str | None = None, max_tokens: int = 800) -> dict:
        text = self.ask(
            system + "\n\nRespond with a single JSON object and nothing else.",
            user, model, max_tokens,
        )
        return parse_json(text)


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()