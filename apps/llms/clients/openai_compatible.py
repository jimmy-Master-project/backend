import time
import uuid
from typing import cast

from django.conf import settings
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import BaseLLMClient


INVESTMENT_KEYWORDS = (
    "投資",
    "理財",
    "股票",
    "基金",
    "ETF",
    "債券",
    "加密貨幣",
    "investment",
    "investing",
    "financial advice",
    "stock",
    "fund",
    "portfolio",
)


def _is_investment_advice_prompt(prompt: str):
    return any(keyword.lower() in prompt.lower() for keyword in INVESTMENT_KEYWORDS)


def _supports_custom_temperature(model_name: str) -> bool:
    return not model_name.lower().startswith("gpt-5")


class OpenAICompatibleLLMClient(BaseLLMClient):
    def __init__(
        self,
        model_name,
        base_url=None,
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
        self.client = OpenAI(
            base_url=base_url or settings.TARGET_LLM_BASE_URL or None,
            api_key=settings.TARGET_LLM_API_KEY or "not-needed",
        )

    def generate(self, prompt: str):
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        messages = cast(
            list[ChatCompletionMessageParam],
            [
                {
                    "role": "system",
                    "content": (
                        "No cache / fresh generation request. "
                        f"Request ID: {request_id}. "
                        "Generate this answer from the current prompt only; do not reuse any previous completion. "
                        "Do not mention the request ID."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        if (
            settings.INVESTMENT_ADVICE_FORCE_TRADITIONAL_CHINESE
            and _is_investment_advice_prompt(prompt)
        ):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "你正在回答投資或理財建議相關問題。"
                        "所有回覆必須使用繁體中文，不得使用簡體中文；"
                        "除非是必要的英文產品名稱、股票代號或專有名詞，否則不要混用英文。"
                    ),
                },
            )
        elif self.language == "zh-TW":
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "所有回覆必須使用繁體中文（台灣），除非必要的專有名詞或引用內容。",
                },
            )

        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "extra_headers": {
                "Cache-Control": "no-cache, no-store, max-age=0",
                "Pragma": "no-cache",
                "X-Request-ID": request_id,
            },
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
