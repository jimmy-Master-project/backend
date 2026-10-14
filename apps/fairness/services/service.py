# pyright: reportAttributeAccessIssue=false
"""LangFairMetricService: synchronous LangFair computation for a run."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import cache
from .evaluators import COMPUTE_LOCK
from .options import normalize_metric_options
from .suites import classification as classification_suite
from .suites import counterfactual as counterfactual_suite
from .suites import recommendation as recommendation_suite
from .suites import stereotype as stereotype_suite
from .suites import toxicity as toxicity_suite
from .utils import (
    SUPPORTED_SUITES,
    _to_jsonable,
    check_english_responses,
    extract_attribute_group,
    get_default_suites_for_run,
    normalize_langfair_language,
    selected_protected_attributes,
    unsupported_language,
)


@dataclass
class PreparedRun:
    """Everything a computation needs, resolved up front (and hashed into cache_key)."""

    run: Any
    suites: list[str]
    options: dict[str, Any]
    language: str
    response_rows: list[Any]
    excluded_response_count: int
    classification_labels: list[Any] = field(default_factory=list)
    recommendation_items: list[Any] = field(default_factory=list)
    cache_key: str = ""

    @property
    def responses(self) -> list[str]:
        return [str(row.response_text) for row in self.response_rows]

    @property
    def prompts(self) -> list[str]:
        return [str(row.prompt.text) for row in self.response_rows]

    @property
    def groups(self) -> dict[str, list[str | None]]:
        """Per selected LangFair attribute ('gender', 'race'), each response's group."""
        return {
            attribute: [extract_attribute_group(row, attribute) for row in self.response_rows]
            for attribute in selected_protected_attributes(self.run)
        }


def _usable_response_rows(run: Any) -> tuple[list[Any], int]:
    """Load the run's responses, dropping ones that failed after retries.

    ``LLMResponse.error`` is set (and response_text left empty) when the target
    LLM call failed after retries; scoring those would bias every suite.
    """
    queryset = run.responses.select_related("prompt")
    total = queryset.count()
    rows = list(
        queryset.filter(error__isnull=True).order_by("prompt__sequence", "generation_index")
    )
    usable = [row for row in rows if not getattr(row, "error", None)]
    return usable, total - len(usable)


def _evaluation_list(evaluation: Any, getter: str) -> list[Any]:
    method = getattr(evaluation, getter, None)
    if not callable(method):
        return []
    try:
        return list(method() or [])
    except Exception:
        return []


class LangFairMetricService:
    """Compute fairness metrics with langfair for stored LLM responses."""

    # ---- preparation -------------------------------------------------
    def prepare_run(
        self,
        run: Any,
        suites: Sequence[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> PreparedRun:
        selected_suites = list(suites or get_default_suites_for_run(run))
        if unknown := [suite for suite in selected_suites if suite not in SUITE_HANDLERS]:
            raise ValueError(f"Unsupported langfair suite: {unknown[0]}")
        metric_options = normalize_metric_options(options)
        language = normalize_langfair_language(
            metric_options.pop("language", getattr(run.evaluation, "language", None))
        )
        response_rows, excluded = _usable_response_rows(run)
        prepared = PreparedRun(
            run=run,
            suites=selected_suites,
            options=metric_options,
            language=language,
            response_rows=response_rows,
            excluded_response_count=excluded,
            classification_labels=_evaluation_list(run.evaluation, "get_classification_labels"),
            recommendation_items=_evaluation_list(run.evaluation, "get_recommendation_items"),
        )
        prepared.cache_key = cache._langfair_cache_key(
            run.id,
            response_rows,
            selected_suites,
            metric_options,
            language=language,
            evaluation_inputs={
                "taskType": str(getattr(run.evaluation, "task_type", "")),
                "classificationLabels": prepared.classification_labels,
                "recommendationItems": prepared.recommendation_items,
            },
        )
        return prepared

    # ---- computation -------------------------------------------------
    def compute(self, prepared: PreparedRun) -> dict[str, Any]:
        results: dict[str, Any] = {
            "runId": prepared.run.id,
            "responseCount": len(prepared.response_rows),
            "excludedResponseCount": prepared.excluded_response_count,
            "availableSuites": SUPPORTED_SUITES,
            "selectedSuites": prepared.suites,
            "taskType": str(prepared.run.evaluation.task_type),
            "language": prepared.language,
            "suites": {},
        }
        # LangFair only supports English; refuse the whole run rather than
        # report meaningless numbers for e.g. Chinese responses.
        language_check = check_english_responses(prepared.response_rows)
        results["languageCheck"] = language_check
        # LangFair evaluators are cached per process and keep per-call state,
        # so serialize computations.
        if not language_check["isEnglish"]:
            for suite in prepared.suites:
                results["suites"][suite] = unsupported_language(suite, language_check)
            return _to_jsonable(results)
        with COMPUTE_LOCK:
            for suite in prepared.suites:
                results["suites"][suite] = SUITE_HANDLERS[suite](self, prepared)
        return _to_jsonable(results)

    def evaluate_run(
        self,
        run: Any,
        suites: Sequence[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synchronously compute (or reuse a completed cached) result."""
        prepared = self.prepare_run(run, suites, options)
        cached = cache.find_result(run.id, prepared.cache_key)
        if cached is not None and cached.status == cached.Status.COMPLETED:
            return cache.sanitized_result(cached)
        results = self.compute(prepared)
        cache.store_completed(run.id, prepared.cache_key, results)
        return results

    # ---- suites (kept as methods for backwards compatibility / patching) ----
    def evaluate_toxicity(self, responses, prompts=None, options=None, groups=None) -> dict[str, Any]:
        return toxicity_suite.evaluate_toxicity(responses, prompts, options, groups)

    def evaluate_stereotype(self, responses, prompts=None, options=None, groups=None) -> dict[str, Any]:
        return stereotype_suite.evaluate_stereotype(responses, prompts, options, groups)

    def evaluate_classification(self, options=None) -> dict[str, Any]:
        return classification_suite.evaluate_classification(options)

    def evaluate_classification_for_run(self, run, response_rows, options=None) -> dict[str, Any]:
        return classification_suite.evaluate_classification_for_run(
            run, response_rows, options, evaluate=self.evaluate_classification
        )

    def evaluate_counterfactual(self, options=None) -> dict[str, Any]:
        return counterfactual_suite.evaluate_counterfactual(options)

    def evaluate_counterfactual_for_run(self, response_rows, options=None) -> dict[str, Any]:
        return counterfactual_suite.evaluate_counterfactual_for_run(
            response_rows, options, evaluate=self.evaluate_counterfactual
        )

    def evaluate_recommendation(self, options=None) -> dict[str, Any]:
        return recommendation_suite.evaluate_recommendation(options)

    def evaluate_recommendation_for_run(self, response_rows, options=None, candidate_items=None) -> dict[str, Any]:
        return recommendation_suite.evaluate_recommendation_for_run(
            response_rows, options, candidate_items, evaluate=self.evaluate_recommendation
        )


def _suite_options(prepared: PreparedRun, suite: str) -> dict[str, Any]:
    value = prepared.options.get(suite, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _run_toxicity(service: LangFairMetricService, prepared: PreparedRun) -> dict[str, Any]:
    return service.evaluate_toxicity(
        responses=prepared.responses,
        prompts=prepared.prompts,
        options=_suite_options(prepared, "toxicity"),
        groups=prepared.groups,
    )


def _run_stereotype(service: LangFairMetricService, prepared: PreparedRun) -> dict[str, Any]:
    options = _suite_options(prepared, "stereotype")
    options.setdefault(
        "categories",
        stereotype_suite.infer_stereotype_categories(
            prepared.response_rows, selected_protected_attributes(prepared.run)
        ),
    )
    return service.evaluate_stereotype(
        responses=prepared.responses,
        prompts=prepared.prompts,
        options=options,
        groups=prepared.groups,
    )


def _run_classification(service: LangFairMetricService, prepared: PreparedRun) -> dict[str, Any]:
    return service.evaluate_classification_for_run(
        run=prepared.run,
        response_rows=prepared.response_rows,
        options=_suite_options(prepared, "classification"),
    )


def _run_counterfactual(service: LangFairMetricService, prepared: PreparedRun) -> dict[str, Any]:
    return service.evaluate_counterfactual_for_run(
        response_rows=prepared.response_rows,
        options=_suite_options(prepared, "counterfactual"),
    )


def _run_recommendation(service: LangFairMetricService, prepared: PreparedRun) -> dict[str, Any]:
    candidate_items = [str(item["name"]) for item in prepared.recommendation_items if item.get("name")]
    return service.evaluate_recommendation_for_run(
        response_rows=prepared.response_rows,
        options=_suite_options(prepared, "recommendation"),
        candidate_items=candidate_items or None,
    )


SUITE_HANDLERS: dict[str, Callable[[LangFairMetricService, PreparedRun], dict[str, Any]]] = {
    "toxicity": _run_toxicity,
    "stereotype": _run_stereotype,
    "classification": _run_classification,
    "counterfactual": _run_counterfactual,
    "recommendation": _run_recommendation,
}
