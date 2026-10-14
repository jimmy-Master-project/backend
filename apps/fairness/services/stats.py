"""Bootstrap confidence intervals for metric values.

Resampling uses a fixed seed so a cached result is reproducible for the same
inputs. Units (responses, prompts or pairs) are resampled with replacement,
separately within each stratum (protected group) when strata are given.

Observations are often not independent: every generation of one prompt pair
(and every prompt of one reused persona) shares its source. Passing
``clusters`` resamples whole clusters instead of single values (cluster
bootstrap); treating dependent values as independent makes intervals too
narrow.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")

BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 0


def bootstrap_interval(
    strata: Sequence[Sequence[T]],
    statistic: Callable[[list[list[T]]], float | None],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any] | None:
    """Percentile bootstrap CI of ``statistic`` over units resampled per stratum."""
    if not strata or any(len(stratum) == 0 for stratum in strata):
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample = [[stratum[i] for i in rng.integers(0, len(stratum), len(stratum))] for stratum in strata]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    if not values:
        return None
    tail = (1 - CONFIDENCE_LEVEL) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return {"ciLow": float(low), "ciHigh": float(high), "confidence": CONFIDENCE_LEVEL, "resamples": resamples}


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _is_finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def group_by_cluster(items: Sequence[T], clusters: Sequence[Any] | None) -> list[list[T]]:
    """Items grouped by cluster label (one cluster per item without labels)."""
    if clusters is None:
        return [[item] for item in items]
    grouped: dict[Any, list[T]] = {}
    for item, cluster in zip(items, clusters, strict=True):
        grouped.setdefault(cluster, []).append(item)
    return list(grouped.values())


def mean_with_ci(values: Sequence[float], clusters: Sequence[Any] | None = None) -> dict[str, Any]:
    """Mean of per-unit values plus a (cluster) bootstrap CI of that mean."""
    kept = [(float(value), None if clusters is None else clusters[index])
            for index, value in enumerate(values) if _is_finite(value)]
    finite = [value for value, _ in kept]
    grouped = group_by_cluster(finite, None if clusters is None else [label for _, label in kept])
    interval = (
        bootstrap_interval([grouped], lambda sample: _mean([value for cluster in sample[0] for value in cluster]))
        if finite
        else None
    )
    return {"value": _mean(finite), "n": len(finite), "clusters": len(grouped), **(interval or {})}


def difference_with_ci(
    values_a: Sequence[float],
    clusters_a: Sequence[Any],
    values_b: Sequence[float],
    clusters_b: Sequence[Any],
) -> dict[str, Any]:
    """mean(a) - mean(b), CI by resampling the clusters both sides share.

    Used for cross-group similarity minus its same-group noise baseline: both
    are measured on the same prompt pairs, so a pair is resampled together
    with its baseline values.
    """
    by_cluster: dict[Any, tuple[list[float], list[float]]] = {}
    for value, cluster in zip(values_a, clusters_a, strict=True):
        if _is_finite(value):
            by_cluster.setdefault(cluster, ([], []))[0].append(float(value))
    for value, cluster in zip(values_b, clusters_b, strict=True):
        if _is_finite(value):
            by_cluster.setdefault(cluster, ([], []))[1].append(float(value))
    shared = [pair for pair in by_cluster.values() if pair[0] and pair[1]]
    if not shared:
        return {"value": None, "clusters": 0}

    def statistic(sample: list[list[tuple[list[float], list[float]]]]) -> float | None:
        first = _mean([value for a, _ in sample[0] for value in a])
        second = _mean([value for _, b in sample[0] for value in b])
        return None if first is None or second is None else first - second

    value = statistic([shared])
    interval = bootstrap_interval([shared], statistic)
    return {"value": value, "clusters": len(shared), **(interval or {})}


def wasserstein_1(scores1: Sequence[float], scores2: Sequence[float]) -> float:
    """Same Wasserstein-1 distance LangFair uses for strong-parity Sentiment Bias."""
    return float(np.mean(np.abs(np.sort(np.asarray(scores1)) - np.sort(np.asarray(scores2)))))


def paired_wasserstein_with_ci(
    scores1: Sequence[float], scores2: Sequence[float], clusters: Sequence[Any] | None = None
) -> dict[str, Any]:
    """W1 between two score lists, bootstrapped over pairs (or clusters of pairs)."""
    pairs = list(zip(scores1, scores2, strict=True))
    if not pairs:
        return {"value": None, "n": 0}
    grouped = group_by_cluster(pairs, clusters)

    def statistic(sample: list[list[list[tuple[float, float]]]]) -> float:
        first, second = zip(*(pair for cluster in sample[0] for pair in cluster), strict=True)
        return wasserstein_1(first, second)

    interval = bootstrap_interval([grouped], statistic)
    return {"value": wasserstein_1(scores1, scores2), "n": len(pairs), "clusters": len(grouped), **(interval or {})}
