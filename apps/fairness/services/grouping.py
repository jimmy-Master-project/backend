"""Per-protected-group breakdown of classifier-score metrics (toxicity / stereotype).

LangFair scores toxicity and stereotype over the whole response pool, which
measures overall safety, not fairness. This recomputes the same metric
functions separately for each protected group and reports the gap between
groups with a bootstrap CI, so a difference can be judged against noise.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .stats import bootstrap_interval

MetricFunction = Callable[[list[float], float], float]


def _group_value(units: Sequence[list[float]], function: MetricFunction, threshold: float) -> float:
    """Mean of the metric over units -- same aggregation as LangFair's Metric.evaluate."""
    return sum(function(scores, threshold) for scores in units) / len(units)


def _gap(values: Sequence[float]) -> float:
    return values[0] - values[1] if len(values) == 2 else max(values) - min(values)


def as_group_sets(groups: Any) -> dict[str, Sequence[str | None]]:
    """``{attribute: groups}``; a plain list (older callers) means gender."""
    if groups is None:
        return {}
    if isinstance(groups, Mapping):
        return {str(attribute): values for attribute, values in groups.items()}
    return {"gender": groups}


def group_comparison(
    scores: Sequence[float],
    prompts: Sequence[str] | None,
    groups: Sequence[str | None],
    threshold: float,
    metric_functions: Mapping[str, MetricFunction],
    by_prompt: bool,
) -> dict[str, Any] | None:
    """Compute ``metric_functions`` per group plus the between-group gap.

    With ``by_prompt`` a unit is all responses of one prompt (as LangFair's
    by-prompt metrics); otherwise each response is its own unit. Units are
    resampled within each group for the CI. Returns None with fewer than two
    groups.
    """
    if not metric_functions:
        return None
    units: dict[str, dict[Any, list[float]]] = {}
    for index, (score, group) in enumerate(zip(scores, groups, strict=False)):
        if group is None or score is None:
            continue
        key = prompts[index] if by_prompt and prompts else index
        units.setdefault(str(group), {}).setdefault(key, []).append(float(score))
    if len(units) < 2:
        return None

    group_names = sorted(units)
    strata = [list(units[group].values()) for group in group_names]
    by_group = {
        group: {name: _group_value(stratum, function, threshold) for name, function in metric_functions.items()}
        for group, stratum in zip(group_names, strata, strict=True)
    }
    gaps = {name: _gap([by_group[group][name] for group in group_names]) for name in metric_functions}
    intervals = {
        name: bootstrap_interval(
            strata,
            lambda sample, function=function: _gap([_group_value(units, function, threshold) for units in sample]),
        )
        for name, function in metric_functions.items()
    }
    return {
        "groups": group_names,
        "comparison": f"{group_names[0]} - {group_names[1]}" if len(group_names) == 2 else "max - min",
        "unit": "prompt" if by_prompt else "response",
        "counts": {
            group: {"responses": sum(len(unit) for unit in stratum), "units": len(stratum)}
            for group, stratum in zip(group_names, strata, strict=True)
        },
        "byGroup": by_group,
        "gap": {
            "metrics": gaps,
            "confidenceIntervals": {name: interval for name, interval in intervals.items() if interval},
        },
    }
