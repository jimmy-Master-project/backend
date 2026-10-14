from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..evaluators import as_key, get_stereotype_evaluator
from ..grouping import as_group_sets, group_comparison
from ..options import DEFAULT_STEREOTYPE_CLASSIFIER

DEFAULT_STEREOTYPE_METRICS = ["Stereotype Association", "Cooccurrence Bias", "Stereotype Classifier"]
DEFAULT_STEREOTYPE_THRESHOLD = 0.5


def _classifier_threshold(evaluator: Any) -> float:
    for metric in getattr(evaluator, "metrics", []):
        if getattr(metric, "name", "") == "Stereotype Classifier":
            return float(getattr(metric, "threshold", DEFAULT_STEREOTYPE_THRESHOLD))
    return DEFAULT_STEREOTYPE_THRESHOLD


def _group_comparisons(
    data: Mapping[str, Any],
    prompts: list[str] | None,
    groups: Sequence[str | None],
    categories: Sequence[str],
    threshold: float,
    by_prompt: bool,
) -> dict[str, Any]:
    """Per-category group breakdown of the Stereotype Classifier metrics."""
    from langfair.metrics.utils.classifier_metrics import ExpectedMaximum, Fraction, Probability

    comparisons: dict[str, Any] = {}
    for category in categories:
        scores = data.get("stereotype_score_" + str(category).lower())
        if scores is None:
            continue
        functions = {f"Stereotype Fraction - {category}": Fraction.metric_function}
        if by_prompt:
            functions[f"Expected Maximum Stereotype - {category}"] = ExpectedMaximum.metric_function
            functions[f"Stereotype Probability - {category}"] = Probability.metric_function
        comparison = group_comparison(
            scores=scores,
            prompts=prompts if by_prompt else None,
            groups=groups,
            threshold=threshold,
            metric_functions=functions,
            by_prompt=by_prompt,
        )
        if comparison is not None:
            comparisons[str(category)] = comparison
    return comparisons


def evaluate_stereotype(
    responses: list[str],
    prompts: list[str] | None = None,
    options: Mapping[str, Any] | None = None,
    groups: Mapping[str, Sequence[str | None]] | Sequence[str | None] | None = None,
) -> dict[str, Any]:
    opts = dict(options or {})
    opts.pop("language", None)
    by_prompt = bool(opts.pop("byPrompt", True))
    return_data = bool(opts.pop("returnData", False))
    show_progress_bars = bool(opts.pop("showProgressBars", False))
    metric_names = as_key(opts.pop("metrics", DEFAULT_STEREOTYPE_METRICS)) or ()
    evaluator = get_stereotype_evaluator(
        metric_names,
        str(opts.pop("classifierModel", DEFAULT_STEREOTYPE_CLASSIFIER)),
    )
    categories = list(opts.pop("categories", ["gender"]))
    # LangFair only builds response-level data inside the Stereotype
    # Classifier (and raises when return_data is set without it).
    has_classifier = "Stereotype Classifier" in metric_names

    evaluate_kwargs: dict[str, Any] = {
        "responses": responses,
        "return_data": has_classifier,
        "categories": categories,
        "show_progress_bars": show_progress_bars,
    }
    if by_prompt:
        evaluate_kwargs["prompts"] = prompts or []

    result = evaluator.evaluate(**evaluate_kwargs)
    if has_classifier:
        from ..utils import STEREOTYPE_SCORING_UNIT

        result["scoringUnit"] = STEREOTYPE_SCORING_UNIT
    data = result.pop("data", None) or {}
    if return_data and has_classifier:
        result["data"] = data
    if groups is not None and data:
        # {attribute: {stereotype category: comparison}}
        by_attribute = {}
        for attribute, attribute_groups in as_group_sets(groups).items():
            comparisons = _group_comparisons(
                data, prompts, attribute_groups, categories, _classifier_threshold(evaluator), by_prompt
            )
            if comparisons:
                by_attribute[attribute] = comparisons
        if by_attribute:
            result["groupComparison"] = by_attribute
    return result


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


def infer_stereotype_categories(
    response_rows: Sequence[Any], selected_attributes: Sequence[str] = ()
) -> list[str]:
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

    # The fairness labels the population was built for (e.g. race) come first.
    for attribute in selected_attributes:
        add_from_attribute(attribute)

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

