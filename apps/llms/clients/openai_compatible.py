import time
import uuid
from typing import cast

from django.conf import settings
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import BaseLLMClient


def _supports_custom_temperature(model_name: str) -> bool:
    return not model_name.lower().startswith("gpt-5")


class OpenAICompatibleLLMClient(BaseLLMClient):
    def __init__(
        self,
        model_name,
        base_url=None,
        api_key=None,
        temperature=0.7,
        top_p=1.0,
        max_tokens=512,
        language="en",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.language = language
        timeout = settings.TARGET_LLM_TIMEOUT_SECONDS
        self.client = OpenAI(
            base_url=base_url or settings.TARGET_LLM_BASE_URL or None,
            api_key=api_key or settings.TARGET_LLM_API_KEY or "not-needed",
            timeout=timeout if timeout > 0 else None,
            max_retries=settings.TARGET_LLM_MAX_RETRIES,
        )

    def generate(self, prompt: str):
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        # Send the prompt alone: any extra system message would become part
        # of the measured input and contaminate the bias evaluation.
        messages = cast(
            list[ChatCompletionMessageParam],
            [{"role": "user", "content": prompt}],
        )
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            # Tracing only; not part of the model input.
            "extra_headers": {"X-Request-ID": request_id},
        }
        if _supports_custom_temperature(self.model_name):
            request_kwargs["temperature"] = self.temperature
            request_kwargs["top_p"] = self.top_p
        request_kwargs[
            "max_tokens" if _supports_custom_temperature(self.model_name) else "max_completion_tokens"
        ] = self.max_tokens if _supports_custom_temperature(self.model_name) else max(self.max_tokens, 8192)
        completion = self.client.chat.completions.create(**request_kwargs)
        try:
            latency_ms = int((time.monotonic() - started) * 1000)
        except (TypeError, ValueError, OverflowError):
            latency_ms = 0

        usage = completion.usage
        response_text = completion.choices[0].message.content or ""
        if not response_text.strip():
            raise RuntimeError("Target LLM returned an empty response.")
        return {
            "text": response_text,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "latency_ms": latency_ms,
        }
