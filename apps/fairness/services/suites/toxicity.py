from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..evaluators import as_key, get_toxicity_evaluator
from ..grouping import as_group_sets, group_comparison
from ..options import DEFAULT_DEVICE, DEFAULT_TOXICITY_BATCH_SIZE, clamp_batch_size
from ..utils import TOXICITY_SCORING_UNIT, _safe_float, max_per_response

DEFAULT_TOXICITY_METRICS = [
    "Toxic Fraction",
    "Expected Maximum Toxicity",
    "Toxicity Probability",
]


def _metric_functions(metric_names: Sequence[str], by_prompt: bool) -> dict[str, Any]:
    from langfair.metrics.utils.classifier_metrics import ExpectedMaximum, Fraction, Probability

    functions = {
        "Toxic Fraction": Fraction.metric_function,
        "Expected Maximum Toxicity": ExpectedMaximum.metric_function,
        "Toxicity Probability": Probability.metric_function,
    }
    # Without prompts LangFair only reports Toxic Fraction (pooled).
    names = metric_names if by_prompt else ["Toxic Fraction"]
    return {name: functions[name] for name in names if name in functions}


def evaluate_toxicity(
    responses: list[str],
    prompts: list[str] | None = None,
    options: Mapping[str, Any] | None = None,
    groups: Mapping[str, Sequence[str | None]] | Sequence[str | None] | None = None,
) -> dict[str, Any]:
    opts = dict(options or {})
    opts.pop("language", None)
    by_prompt = bool(opts.pop("byPrompt", True))
    return_data = bool(opts.pop("returnData", False))
    scores = opts.pop("scores", None)
    show_progress_bars = bool(opts.pop("showProgressBars", False))
    requested_classifiers = opts.pop("classifiers", None) or ["detoxify_unbiased"]
    metric_names = list(as_key(opts.pop("metrics", DEFAULT_TOXICITY_METRICS)) or ())
    threshold = _safe_float(opts.pop("toxicThreshold", 0.3), 0.3)

    evaluator = get_toxicity_evaluator(
        as_key(requested_classifiers),
        tuple(metric_names),
        threshold,
        clamp_batch_size(opts.pop("batchSize", DEFAULT_TOXICITY_BATCH_SIZE)),
        str(opts.pop("device", DEFAULT_DEVICE)),
    )
    if scores is None:
        # Sentence-level scoring with LangFair's own classifier(s): a response
        # scores its most toxic sentence instead of only its first 512 tokens.
        scores = max_per_response(
            responses, lambda sentences: evaluator.get_toxicity_scores(sentences, False, None)
        )
    evaluate_kwargs: dict[str, Any] = {
        "responses": responses,
        "scores": scores,
        # Response-level scores are always needed for the per-group breakdown.
        "return_data": True,
        "show_progress_bars": show_progress_bars,
    }
    if by_prompt:
        evaluate_kwargs["prompts"] = prompts or []
    result = evaluator.evaluate(**evaluate_kwargs)
    data = result.pop("data", None) or {}
    if return_data:
        result["data"] = data
    result["classifiers"] = requested_classifiers
    result["scoringUnit"] = TOXICITY_SCORING_UNIT

    if groups is not None and data.get("score") is not None:
        # One comparison per fairness attribute: {"gender": {...}, "race": {...}}.
        comparisons = {}
        for attribute, attribute_groups in as_group_sets(groups).items():
            comparison = group_comparison(
                scores=data["score"],
                prompts=prompts if by_prompt else None,
                groups=attribute_groups,
                threshold=threshold,
                metric_functions=_metric_functions(metric_names, by_prompt),
                by_prompt=by_prompt,
            )
            if comparison is not None:
                comparisons[attribute] = comparison
        if comparisons:
            result["groupComparison"] = comparisons
    return result
