"""Validation of user-supplied LangFair options.

Model names and devices end up in HuggingFace / torch loaders, so only a known
whitelist is accepted: the built-in defaults plus optional settings lists
(``LANGFAIR_ALLOWED_CLASSIFIER_MODELS``, ``LANGFAIR_ALLOWED_TRANSFORMERS``,
``LANGFAIR_ALLOWED_DEVICES``).
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from django.conf import settings

from .utils import _clamp, _safe_int

DEFAULT_STEREOTYPE_CLASSIFIER = "wu981526092/Sentence-Level-Stereotype-Detector"
DEFAULT_TRANSFORMER = "all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
DEFAULT_TOXICITY_BATCH_SIZE = 250
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 256


def _allowed(default: str, setting_name: str) -> set[str]:
    extra = getattr(settings, setting_name, None) or []
    if isinstance(extra, str):
        extra = [item.strip() for item in extra.split(",") if item.strip()]
    return {default, *map(str, extra)}


def allowed_classifier_models() -> set[str]:
    return _allowed(DEFAULT_STEREOTYPE_CLASSIFIER, "LANGFAIR_ALLOWED_CLASSIFIER_MODELS")


def allowed_transformers() -> set[str]:
    return _allowed(DEFAULT_TRANSFORMER, "LANGFAIR_ALLOWED_TRANSFORMERS")


def allowed_devices() -> set[str]:
    return _allowed(DEFAULT_DEVICE, "LANGFAIR_ALLOWED_DEVICES")


def _check(value: Any, allowed: set[str], field: str) -> str:
    text = str(value)
    if text not in allowed:
        raise ValueError(f"{field} '{text}' is not allowed. Allowed values: {sorted(allowed)}")
    return text


def clamp_batch_size(value: Any, default: int = DEFAULT_TOXICITY_BATCH_SIZE) -> int:
    return _clamp(_safe_int(value, default), MIN_BATCH_SIZE, MAX_BATCH_SIZE)


def normalize_metric_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a deep-copied, validated options dict (raises ValueError on bad values)."""
    normalized: dict[str, Any] = copy.deepcopy(dict(options or {}))
    for suite_options in normalized.values():
        if not isinstance(suite_options, dict):
            continue
        if "device" in suite_options:
            suite_options["device"] = _check(suite_options["device"], allowed_devices(), "device")
        if "batchSize" in suite_options:
            suite_options["batchSize"] = clamp_batch_size(suite_options["batchSize"])
    stereotype = normalized.get("stereotype")
    if isinstance(stereotype, dict) and "classifierModel" in stereotype:
        stereotype["classifierModel"] = _check(
            stereotype["classifierModel"], allowed_classifier_models(), "classifierModel"
        )
    counterfactual = normalized.get("counterfactual")
    if isinstance(counterfactual, dict) and "transformer" in counterfactual:
        counterfactual["transformer"] = _check(
            counterfactual["transformer"], allowed_transformers(), "transformer"
        )
    return normalized
