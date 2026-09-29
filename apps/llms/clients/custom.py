import time

import httpx
from django.conf import settings

from .base import BaseLLMClient


class CustomLLMClient(BaseLLMClient):
    """POST a stable request shape to any external robot endpoint."""

    def __init__(
        self,
        model_name,
        base_url,
        api_key=None,
        temperature=0.7,
        top_p=1.0,
        max_tokens=512,
        language="en",
    ):
        if not base_url:
            raise ValueError("A base URL is required for the custom provider.")
        self.url = base_url
        self.payload = {
            "prompt": None,
            "model": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def generate(self, prompt: str):
        started = time.monotonic()
        payload = {**self.payload, "prompt": prompt}
        timeout = settings.TARGET_LLM_TIMEOUT_SECONDS
        response = httpx.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=timeout if timeout > 0 else None,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            text = data.get("text") or data.get("response") or data.get("output")
            if text is None and isinstance(data.get("message"), dict):
                text = data["message"].get("content")
        else:
            text = None
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(
                "Custom LLM response must contain text, response, output, or message.content."
            )
        return {
            "text": text,
            "input_tokens": data.get("input_tokens") if isinstance(data, dict) else None,
            "output_tokens": data.get("output_tokens") if isinstance(data, dict) else None,
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
