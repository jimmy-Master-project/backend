# ruff: noqa: F401
"""LangFair fairness metric services.

Package layout:
  service.py     LangFairMetricService (synchronous compute, suite dispatch dict)
  runner.py      LangFairMetricJobRunner (background thread + in-process registry)
  cache.py       cache key + LangFairMetricResult persistence / pruning
  evaluators.py  process-wide lru_cache of LangFair evaluator objects
  options.py     whitelist / clamping of user-supplied options
  suites/        one module per metric suite
  utils.py       constants and shared helpers

Everything previously importable from ``apps.fairness.services`` is re-exported.
"""
from apps.fairness.models import LangFairMetricResult  # noqa: F401

from .cache import METRICS_SCHEMA_VERSION, _langfair_cache_key, prune_results
from .evaluators import clear_evaluator_caches
from .options import normalize_metric_options
from .runner import LangFairMetricJobRunner, reap_stale_jobs
from .service import SUITE_HANDLERS, LangFairMetricService, PreparedRun
from .suites.classification import (
    _normalize_classification_mapping,
    _resolve_label_refs,
    build_classification_confusion_summary,
    build_classification_metric_inputs,
    evaluate_classification_metrics_safely,
    extract_expected_classification_label,
    parse_classification_response,
    resolve_classification_label,
    validate_classification_metric_arrays,
)
from .suites.counterfactual import build_counterfactual_text_pairs
from .suites.recommendation import (
    RECOMMENDATION_PAIR_TASKS,
    build_recommendation_metric_inputs,
    extract_recommendation_items,
)
from .suites.stereotype import STEREOTYPE_CATEGORY_BY_ATTRIBUTE, infer_stereotype_categories
from .utils import (
    DEFAULT_SUITES,
    SUPPORTED_LANGUAGES,
    SUPPORTED_SUITES,
    TASK_TYPE_DEFAULT_SUITES,
    _missing_input,
    _missing_keys,
    _safe_float,
    _safe_int,
    _to_jsonable,
    extract_protected_group,
    get_default_suites_for_run,
    normalize_gender_group,
    normalize_langfair_language,
)

__all__ = [
    "DEFAULT_SUITES",
    "METRICS_SCHEMA_VERSION",
    "RECOMMENDATION_PAIR_TASKS",
    "STEREOTYPE_CATEGORY_BY_ATTRIBUTE",
    "SUITE_HANDLERS",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_SUITES",
    "TASK_TYPE_DEFAULT_SUITES",
    "LangFairMetricJobRunner",
    "LangFairMetricResult",
    "LangFairMetricService",
    "PreparedRun",
    "build_classification_confusion_summary",
    "build_classification_metric_inputs",
    "build_counterfactual_text_pairs",
    "build_recommendation_metric_inputs",
    "clear_evaluator_caches",
    "evaluate_classification_metrics_safely",
    "extract_expected_classification_label",
    "extract_protected_group",
    "extract_recommendation_items",
    "get_default_suites_for_run",
    "infer_stereotype_categories",
    "normalize_gender_group",
    "normalize_langfair_language",
    "normalize_metric_options",
    "parse_classification_response",
    "prune_results",
    "reap_stale_jobs",
    "resolve_classification_label",
    "validate_classification_metric_arrays",
]
