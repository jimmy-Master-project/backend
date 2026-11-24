"""The generation LLM (PROMPT_AGENT_* settings, i.e. the LLM behind the personas).

Every group uses this same model; calls are counted per group for the
process table (generation cost).
"""
from __future__ import annotations

import json
import re
from typing import Any

from django.conf import settings
from openai import OpenAI


class GenerationLLM:
    def __init__(self, model: str | None = None, temperature: float = 0.8, client: Any = None):
        self.model = model or settings.PROMPT_AGENT_MODEL_NAME
        self.temperature = temperature
        self.calls = 0
        self.client = client or OpenAI(
            base_url=settings.PROMPT_AGENT_BASE_URL or None,
            api_key=settings.PROMPT_AGENT_API_KEY or "not-needed",
            timeout=settings.PROMPT_AGENT_TIMEOUT_SECONDS or None,
            max_retries=settings.PROMPT_AGENT_MAX_RETRIES,
        )

    def text(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        self.calls += 1
        kwargs: dict[str, Any] = {}
        if not self.model.lower().startswith("gpt-5"):
            kwargs["temperature"] = self.temperature if temperature is None else temperature
        completion = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return (completion.choices[0].message.content or "").strip()

    def json(self, messages: list[dict[str, str]], temperature: float | None = None) -> Any:
        return parse_json(self.text(messages, temperature))


def parse_json(text: str) -> Any:
    """The JSON object/array in a reply, tolerating code fences and prose around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("The generation LLM did not return valid JSON.")
