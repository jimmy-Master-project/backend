"""The protected attributes a fairness diagnosis may target.

LangFair only supports gender and race/ethnicity out of the box (Bouchard et
al., 2025, sec. 2: CounterfactualGenerator; groups male/female and
white/black/hispanic/asian). Each is backed by one MatrAIx persona dimension,
whose values are mapped onto LangFair's groups; values with no LangFair group
(e.g. Non-binary, Middle Eastern) take part in prompt generation but not in
that attribute's group comparison.
"""
from __future__ import annotations

from typing import Any

PROTECTED_ATTRIBUTES: dict[str, dict[str, Any]] = {
    "gender_identity": {
        "attribute": "gender",
        "label": "Gender",
        # Gender pairs are built by swapping gendered words (see
        # apps.prompts.services.counterfactual); race has no counterfactual.
        "counterfactual": True,
        "groups": {
            "female": ["Woman", "Female", "F", "Women"],
            "male": ["Man", "Male", "M", "Men"],
        },
    },
    "demo_ethnicity_broad": {
        "attribute": "race",
        "label": "Race / ethnicity",
        "counterfactual": False,
        "groups": {
            "white": ["White / European"],
            "black": ["Black / African"],
            "hispanic": ["Hispanic / Latino"],
            "asian": ["East Asian", "South Asian", "Southeast Asian"],
        },
    },
}

# Older populations / clients may name the gender dimension "gender".
ATTRIBUTE_ALIASES = {"gender": "gender_identity"}

# Extra spellings accepted when mapping a persona value onto a group.
_EXTRA_VALUES = {"gender_identity": {"男": "male", "男性": "male", "女": "female", "女性": "female"}}


def canonical_key(key: Any) -> str:
    text = str(key or "").strip()
    return ATTRIBUTE_ALIASES.get(text, text)


def is_protected_attribute(key: Any) -> bool:
    return canonical_key(key) in PROTECTED_ATTRIBUTES


def langfair_attribute(key: Any) -> str | None:
    """'gender' / 'race' for a supported persona dimension, else None."""
    spec = PROTECTED_ATTRIBUTES.get(canonical_key(key))
    return spec["attribute"] if spec else None


def langfair_group(key: Any, value: Any) -> str | None:
    """The LangFair group of a persona value, or None when it has none."""
    canonical = canonical_key(key)
    spec = PROTECTED_ATTRIBUTES.get(canonical)
    if spec is None or value is None:
        return None
    normalized = str(value).strip().lower()
    for group, values in spec["groups"].items():
        if normalized in {item.lower() for item in values} or normalized == group:
            return group
    return _EXTRA_VALUES.get(canonical, {}).get(str(value).strip())


def serialize_protected_attributes() -> list[dict[str, Any]]:
    """camelCase description for clients (selector, value filters)."""
    return [
        {
            "key": key,
            "attribute": spec["attribute"],
            "label": spec["label"],
            "counterfactual": spec["counterfactual"],
            "groups": [{"name": group, "values": list(values)} for group, values in spec["groups"].items()],
        }
        for key, spec in PROTECTED_ATTRIBUTES.items()
    ]
