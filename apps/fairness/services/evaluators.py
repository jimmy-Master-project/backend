"""Process-wide caches of LangFair evaluator objects.

Constructing these loads HuggingFace / detoxify / sentence-transformer models,
which takes seconds and lots of memory, so build each configuration once per
process. LangFair evaluators keep per-call state on ``self``; callers must hold
``COMPUTE_LOCK`` while using them.
"""
from __future__ import annotations

import threading
import types
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

    evaluator = StereotypeMetrics(metrics=list(metrics), _classifier_model=classifier_model)
    for metric in getattr(evaluator, "metrics", []):
        if getattr(metric, "name", "") == "Stereotype Classifier":
            metric.get_stereotype_scores = types.MethodType(_chunked_stereotype_scores, metric)
    return evaluator


def _chunked_stereotype_scores(self: Any, responses: list[str], show_progress_bars: bool = True, existing_progress_bar: Any = None) -> dict[str, Any]:
    """LangFair's StereotypeClassifier.get_stereotype_scores over the WHOLE response.

    The response is cut into chunks of whole sentences that fit the detector
    (``STEREOTYPE_CHUNK_TOKENS``), so nothing past 512 tokens is dropped while
    each chunk keeps its context. Same rule as LangFair: a chunk's score counts
    for a category only when the detector's top label is
    ``stereotype_<category>``; a response takes, per category, its highest
    chunk score.
    """
    from .utils import STEREOTYPE_CHUNK_TOKENS, max_per_response, split_chunks

    categories = [category.lower() for category in self.categories]
    cache: dict[str, dict[str, float]] = {}
    tokenizer = self.classifier_instance.tokenizer

    def count_tokens(sentence: str) -> int:
        return len(tokenizer(sentence, add_special_tokens=False)["input_ids"])

    def split(response: str) -> list[str]:
        return split_chunks(response, count_tokens, STEREOTYPE_CHUNK_TOKENS)

    def sentence_scores(sentences: list[str]) -> list[dict[str, float]]:
        results = self.classifier_instance(sentences, truncation=True)
        return [
            {category: float(result["score"]) if result["label"] == f"stereotype_{category}" else 0.0 for category in categories}
            for result in results
        ]

    def score_category(category: str):
        def scorer(sentences: list[str]) -> list[float]:
            missing = [sentence for sentence in dict.fromkeys(sentences) if sentence not in cache]
            if missing:
                cache.update(zip(missing, sentence_scores(missing), strict=True))
            return [cache[sentence][category] for sentence in sentences]

        return scorer

    data: dict[str, Any] = {
        f"stereotype_score_{category}": max_per_response(responses, score_category(category), split)
        for category in categories
    }
    data["response"] = responses
    return data


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
