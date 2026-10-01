"""Process-wide caches of LangFair evaluator objects.

Constructing these loads HuggingFace / detoxify / sentence-transformer models,
which takes seconds and lots of memory, so build each configuration once per
process. LangFair evaluators keep per-call state on ``self``; callers must hold
``COMPUTE_LOCK`` while using them.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

COMPUTE_LOCK = threading.RLock()


def as_key(values: Sequence[Any] | None) -> tuple[Any, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return (values,)
    return tuple(values)


@lru_cache(maxsize=4)
def get_toxicity_evaluator(
    classifiers: tuple[str, ...],
    metrics: tuple[str, ...],
    toxic_threshold: float,
    batch_size: int,
    device: str,
) -> Any:
    from langfair.metrics.toxicity.toxicity import ToxicityMetrics

    return ToxicityMetrics(
        classifiers=list(classifiers),
        metrics=list(metrics),
        toxic_threshold=toxic_threshold,
        batch_size=batch_size,
        device=device,
    )


@lru_cache(maxsize=4)
def get_stereotype_evaluator(metrics: tuple[str, ...], classifier_model: str) -> Any:
    from langfair.metrics.stereotype.stereotype import StereotypeMetrics

    return StereotypeMetrics(metrics=list(metrics), _classifier_model=classifier_model)


@lru_cache(maxsize=4)
def get_counterfactual_evaluator(
    metrics: tuple[str, ...],
    neutralize_tokens: bool,
    sentiment_classifier: str,
    transformer: str,
    device: str,
    how: str,
) -> Any:
    from langfair.metrics.counterfactual.counterfactual import CounterfactualMetrics

    return CounterfactualMetrics(
        metrics=list(metrics),
        neutralize_tokens=neutralize_tokens,
        sentiment_classifier=sentiment_classifier,
        transformer=transformer,
        device=device,
        how=how,
    )


@lru_cache(maxsize=4)
def get_recommendation_evaluator(metrics: tuple[str, ...]) -> Any:
    from langfair.metrics.recommendation.recommendation import RecommendationMetrics

    return RecommendationMetrics(metrics=list(metrics))


def clear_evaluator_caches() -> None:
    for factory in (
        get_toxicity_evaluator,
        get_stereotype_evaluator,
        get_counterfactual_evaluator,
        get_recommendation_evaluator,
    ):
        factory.cache_clear()
