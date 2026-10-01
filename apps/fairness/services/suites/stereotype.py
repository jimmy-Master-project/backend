from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..evaluators import as_key, get_stereotype_evaluator
from ..options import DEFAULT_STEREOTYPE_CLASSIFIER

DEFAULT_STEREOTYPE_METRICS = ["Stereotype Association", "Cooccurrence Bias", "Stereotype Classifier"]


def evaluate_stereotype(
    responses: list[str],
    prompts: list[str] | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    opts = dict(options or {})
    opts.pop("language", None)
    by_prompt = bool(opts.pop("byPrompt", True))
    return_data = bool(opts.pop("returnData", False))
    show_progress_bars = bool(opts.pop("showProgressBars", False))
    evaluator = get_stereotype_evaluator(
        as_key(opts.pop("metrics", DEFAULT_STEREOTYPE_METRICS)),
        str(opts.pop("classifierModel", DEFAULT_STEREOTYPE_CLASSIFIER)),
    )

    evaluate_kwargs: dict[str, Any] = {
        "responses": responses,
        "return_data": return_data,
        "categories": opts.pop("categories", ["gender"]),
        "show_progress_bars": show_progress_bars,
    }
    if by_prompt:
        evaluate_kwargs["prompts"] = prompts or []

    return evaluator.evaluate(**evaluate_kwargs)


STEREOTYPE_CATEGORY_BY_ATTRIBUTE = {
    "gender": "gender",
    "gender_identity": "gender",
    "sex": "gender",
    "race": "race",
    "ethnicity": "race",
    "race_ethnicity": "race",
    "demo_ethnicity_broad": "race",
    "profession": "profession",
    "occupation": "profession",
    "job": "profession",
    "religion": "religion",
    "faith": "religion",
}


def infer_stereotype_categories(response_rows: Sequence[Any]) -> list[str]:
    """Infer which Stereotype Classifier categories (gender/race/profession/religion) to score.

    Only derives categories from `fairness.attribute` / `fairness.source_attribute`
    -- the attribute(s) a counterfactual/diagnostic pipeline was actually built
    around -- never from the raw keys of `persona_attributes`. Since personas
    now carry every dimension MatrAIx returns (not just the one being
    diagnosed; see PersonaService.get_projected_persona), scanning
    persona_attributes keys would flag categories like "race" just because an
    incidental ethnicity-like field happened to ride along on the persona,
    even though no race counterfactual pair was ever generated for it. That
    would silently attach an unrelated, undesigned category's stereotype score
    to results the user reads as being about their actual experiment.
    """
    categories: list[str] = []
    seen: set[str] = set()

    def add_from_attribute(attribute: Any) -> None:
        key = str(attribute or "").strip().lower()
        category = STEREOTYPE_CATEGORY_BY_ATTRIBUTE.get(key)
        if category and category not in seen:
            seen.add(category)
            categories.append(category)

    for response in response_rows:
        prompt = getattr(response, "prompt", None)
        metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
        if not isinstance(metadata, Mapping):
            continue

        fairness = metadata.get("fairness")
        if isinstance(fairness, Mapping):
            add_from_attribute(fairness.get("attribute"))
            add_from_attribute(fairness.get("source_attribute"))

    return categories or ["gender"]

