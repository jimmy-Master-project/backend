import json
import threading
import time
import uuid

from django.conf import settings
from openai import OpenAI

from .base import BasePromptAgentClient


class PromptAgentError(Exception):
    pass


class OpenAICompatiblePromptAgentClient(BasePromptAgentClient):
    REQUEST_INTERVAL_SECONDS = 0.5
    _request_lock = threading.Lock()
    _last_request_at = 0.0

    @classmethod
    def _wait_for_request_interval(cls):
        with cls._request_lock:
            elapsed = time.monotonic() - cls._last_request_at
            remaining = cls.REQUEST_INTERVAL_SECONDS - elapsed
            if remaining > 0:
                time.sleep(remaining)
            cls._last_request_at = time.monotonic()

    def __init__(self, model_name, base_url=None, temperature=0.8):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url or settings.PROMPT_AGENT_BASE_URL or None,
            api_key=settings.PROMPT_AGENT_API_KEY or "not-needed",
        )

    def _create_completion(self, **kwargs):
        self._wait_for_request_interval()
        return self.client.chat.completions.create(**kwargs)

    def _temperature_kwargs(self):
        return {} if self.model_name.lower().startswith("gpt-5") else {"temperature": self.temperature}

    def generate(self, messages):
        request_id = uuid.uuid4().hex
        no_cache_messages = [
            {
                "role": "system",
                "content": (
                    "No cache / fresh prompt generation request. "
                    f"Request ID: {request_id}. "
                    "Generate new content from the current instructions only; do not reuse any previous output. "
                    "Do not mention the request ID."
                ),
            },
            *messages,
        ]
        extra_headers = {
            "Cache-Control": "no-cache, no-store, max-age=0",
            "Pragma": "no-cache",
            "X-Request-ID": request_id,
        }
        try:
            completion = self._create_completion(
                model=self.model_name,
                messages=no_cache_messages,
                **self._temperature_kwargs(),
                response_format={"type": "json_object"},
                extra_headers=extra_headers,
            )
        except Exception:
            # Some OpenAI-compatible servers (vLLM/llama.cpp builds) reject
            # response_format. Retry without it and rely on the prompt
            # instruction to enforce JSON-only output.
            completion = self._create_completion(
                model=self.model_name,
                messages=no_cache_messages,
                **self._temperature_kwargs(),
                extra_headers=extra_headers,
            )

        content = completion.choices[0].message.content
        if not content:
            raise PromptAgentError("Prompt agent returned an empty response.")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise PromptAgentError(
                f"Prompt agent did not return valid JSON: {exc}"
            ) from exc
