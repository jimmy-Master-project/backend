"""Multi-turn conversations with the target chatbot.

Rules (研究問題構想.md): up to 5 turns; a reply is a follow-up question when
it matches FOLLOW_UP_PATTERNS; the user then answers with the fixed message
FOLLOW_UP_REPLY; a conversation that still asks after the last turn is
excluded; metrics use the last turn's answer.

FOLLOW_UP_PATTERNS were collected from the 7 airline chatbots' real answers,
e.g. "Could you let me know whether you booked directly with us or through a
travel agent?" (Malaysia) and "Would you still like to proceed with the refund
request?" (Philippine). Closing checks such as "Does this help?" or "Is there
anything else I can assist with?" are not follow-ups and match none of them.
"""
from __future__ import annotations

import re
import time
from typing import Any

import httpx
from django.conf import settings
from openai import OpenAI

FOLLOW_UP_REPLY = "Please decide on your own and answer."
MAX_TURNS = 5
FOLLOW_UP_PATTERNS = [
    r"\b(could|can|would|will) you (please )?(let me know|tell me|share|provide|confirm|clarify|specify|give me)\b",
    r"\bwould you (still )?like (me )?to (proceed|continue|go ahead)\b",
    r"\bdo you (already )?have (a|an|your) (booking|reservation|ticket|booking reference)\b",
    r"\b(may|can) i (have|ask|know|get) (your|for|which|whether|the|what)\b",
    r"\bplease (provide|share|tell me|let me know|confirm|specify)\b[^.?!]*\b(your|whether|which|if|what)\b",
    r"\b(which|what) (is )?(your )?(flight|route|destination|booking reference|travel date|cabin class|class|fare|airport|ticket number)\b[^.!]*\?",
    r"\bwhether you (booked|purchased|bought|have|are)\b[^.!]*\?",
    r"\bdid you (book|purchase|buy)\b[^.!]*\?",
    # Any other question sentence put to the user, unless it is a closing check
    # ("Does this help?", "Is there anything else I can assist with?", "Did that
    # answer your question?"). A "?" inside a URL is not a question mark.
    r"(?:^|(?<=[.!?\n]))\s*(?![^.!?\n]*\b(?:help\w*|assist\w*|anything else|further|other questions?|answer(?:ed)? your question)\b)[^.!?\n]*\?(?=\s|$)",
]

# Chatbot services (Crawler-bot): provider -> conversation endpoint base.
BOT_ENDPOINTS = {
    "evaair": "http://127.0.0.1:8768/v1",
    "virgin": "http://127.0.0.1:8769/virgin/v1",
    "finnair": "http://127.0.0.1:8769/finnair/v1",
    "chinaairlines": "http://127.0.0.1:8769/chinaairlines/v1",
    "malaysia": "http://127.0.0.1:8769/malaysia/v1",
    "philippine": "http://127.0.0.1:8769/philippine/v1",
    "vietjet": "http://127.0.0.1:8769/vietjet/v1",
}
TARGET_LABELS = {
    "evaair": "長榮 EVA Air",
    "virgin": "Virgin Atlantic",
    "finnair": "Finnair",
    "chinaairlines": "華航 China Airlines",
    "malaysia": "Malaysia Airlines",
    "philippine": "Philippine Airlines",
    "vietjet": "Vietjet",
    "openai": "OpenAI（測試用）",
}


def is_follow_up(answer: str) -> bool:
    return any(re.search(p, answer or "", re.IGNORECASE) for p in FOLLOW_UP_PATTERNS)


class ConversationError(Exception):
    pass


class TargetClient:
    """converse(question) -> {"turns": [...], "ended": "answered" | "max_turns"}."""

    def __init__(self, provider: str, base_url: str = "", model: str = "", timeout: float = 900.0):
        self.provider = provider
        self.base_url = (base_url or BOT_ENDPOINTS.get(provider, "")).rstrip("/")
        self.model = model
        self.timeout = timeout
        if provider not in BOT_ENDPOINTS and provider != "openai":
            raise ConversationError(f"Unsupported target: {provider}")

    def converse(self, question: str) -> dict[str, Any]:
        if self.provider == "openai":
            return self._converse_openai(question)
        try:
            response = httpx.post(
                f"{self.base_url}/conversation",
                json={
                    "question": question,
                    "follow_up_reply": FOLLOW_UP_REPLY,
                    "max_turns": MAX_TURNS,
                    "follow_up_patterns": FOLLOW_UP_PATTERNS,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ConversationError(f"Cannot reach the chatbot service: {exc}") from exc
        if response.status_code != 200:
            try:
                message = response.json()["error"]["message"]
            except Exception:  # noqa: BLE001
                message = response.text[:300]
            raise ConversationError(f"HTTP {response.status_code}: {message}")
        return response.json()

    def _converse_openai(self, question: str) -> dict[str, Any]:
        client = OpenAI(
            base_url=self.base_url or settings.TARGET_LLM_BASE_URL or None,
            api_key=settings.TARGET_LLM_API_KEY or "not-needed",
            timeout=settings.TARGET_LLM_TIMEOUT_SECONDS or None,
            max_retries=settings.TARGET_LLM_MAX_RETRIES,
        )
        messages: list[dict[str, str]] = []
        turns = []
        user = question
        for turn in range(1, MAX_TURNS + 1):
            messages.append({"role": "user", "content": user})
            started = time.monotonic()
            completion = client.chat.completions.create(model=self.model or settings.TARGET_LLM_MODEL_NAME, messages=messages)
            answer = (completion.choices[0].message.content or "").strip()
            messages.append({"role": "assistant", "content": answer})
            follow_up = is_follow_up(answer)
            turns.append({"user": user, "assistant": answer, "latency_ms": int((time.monotonic() - started) * 1000), "follow_up": follow_up})
            if not follow_up:
                return {"turns": turns, "ended": "answered"}
            user = FOLLOW_UP_REPLY
        return {"turns": turns, "ended": "max_turns"}
