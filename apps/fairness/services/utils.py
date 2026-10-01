"""Shared constants and small helpers for LangFair metric computation."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SUPPORTED_SUITES = [
    "toxicity",
    "stereotype",
    "classification",
    "counterfactual",
    "recommendation",
]
DEFAULT_SUITES = ["toxicity", "stereotype", "counterfactual"]
SUPPORTED_LANGUAGES = {"en", "auto"}
TASK_TYPE_DEFAULT_SUITES = {
    "text_generation": ["toxicity", "stereotype", "counterfactual"],
    "classification": ["classification"],
    "recommendation": ["recommendation"],
    "other": ["toxicity", "stereotype"],
}
PAIR_ROLES = {"original", "counterfactual"}


def get_default_suites_for_run(run: Any) -> list[str]:
    task_type = str(getattr(run.evaluation, "task_type", "other"))
    return TASK_TYPE_DEFAULT_SUITES.get(task_type, TASK_TYPE_DEFAULT_SUITES["other"])


def normalize_langfair_language(value: Any = None, responses: Sequence[str] = ()) -> str:
    language = str(value or "en")
    if language == "auto":
        return "en"
    if language not in SUPPORTED_LANGUAGES - {"auto"}:
        raise ValueError("language must be 'en' or 'auto'.")
    return language


def prompt_metadata(response: Any) -> Mapping[str, Any]:
    prompt = getattr(response, "prompt", None)
    metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
    return metadata if isinstance(metadata, Mapping) else {}


def fairness_metadata(response: Any) -> Mapping[str, Any] | None:
    fairness = prompt_metadata(response).get("fairness")
    return fairness if isinstance(fairness, Mapping) else None


def pair_key(response: Any, fairness: Mapping[str, Any], pair_id: Any) -> tuple[str, str, int]:
    """Key a counterfactual pair member as (objective, pair_id, generation_index).

    Older prompt populations restarted pair ids (``cf_001``...) for every task
    objective, so pair_id alone collides across objectives. The objective is
    taken from ``fairness.objective_index`` (new populations) or the prompt's
    ``task_objective`` metadata (all populations generated per objective).
    """
    objective = fairness.get("objective_index")
    if objective is None:
        objective = prompt_metadata(response).get("task_objective") or ""
    generation_index = _safe_int(getattr(response, "generation_index", 1), 1)
    return (str(objective), str(pair_id), generation_index)


def extract_protected_group(response: Any) -> str | None:
    metadata = prompt_metadata(response)
    if not metadata:
        return None

    fairness = metadata.get("fairness")
    if isinstance(fairness, Mapping) and fairness.get("group"):
        return str(fairness["group"])

    persona_attributes = metadata.get("persona_attributes")
    if isinstance(persona_attributes, Mapping):
        for key in ("gender", "gender_identity"):
            group = normalize_gender_group(persona_attributes.get(key))
            if group:
                return group
    return None


def normalize_gender_group(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"m", "man", "men", "male", "男", "男性"}:
        return "male"
    if normalized in {"f", "woman", "women", "female", "女", "女性"}:
        return "female"
    return normalized or None


def _missing_keys(options: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key not in options]


def _missing_input(suite: str, missing: Sequence[str]) -> dict[str, Any]:
    return {
        "status": "missing_input",
        "suite": suite,
        "missingOptions": list(missing),
        "message": f"LangFair {suite} metrics require additional input data.",
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas values to plain JSON types; NaN/inf become None.

    DRF renders JSON strictly (allow_nan=False), so a single NaN metric (e.g. a
    group with no positives) would otherwise turn the whole response into a 500.
    """
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, str | bool | int) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item"):
        return _to_jsonable(value.item())
    return value
