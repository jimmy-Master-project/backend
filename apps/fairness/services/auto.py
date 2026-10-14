"""Start the default LangFair computation as soon as a run completes.

The request uses exactly the suites/options the results page asks for by
default (see frontend `buildLangFairBody` / `defaultClassificationMapping`),
so opening the page finds the cached result instead of starting a new one.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)

# Same preference as the frontend's defaultLabelSelection.
_POSITIVE_LABEL = re.compile(r"approve|positive|正面|通過", re.IGNORECASE)


def default_metric_request(evaluation: Any) -> tuple[list[str] | None, dict[str, Any]]:
    """(suites, options) the results page requests by default for this evaluation."""
    language = str(getattr(evaluation, "language", "") or "en")
    if str(getattr(evaluation, "task_type", "")) == "classification":
        labels = list(evaluation.get_classification_labels() or [])
        if len(labels) == 2:
            preferred = next(
                (label for label in labels if _POSITIVE_LABEL.search(f"{label.get('name', '')} {label.get('description', '')}")),
                None,
            )
            positive = (preferred or labels[0]).get("value")
            negative = next((label.get("value") for label in labels if label.get("value") != positive), None)
            mapping = {"positiveLabels": [positive], "negativeLabels": [] if negative is None else [negative]}
            return ["classification"], {
                "language": language,
                "classification": {"classificationMetricMapping": mapping},
            }
    return None, {"language": language}


def compute_metrics_for_completed_run(run_id: int) -> None:
    """Queue the default computation; never raises (a run must not fail because of it)."""
    if not getattr(settings, "FAIRNESS_AUTO_COMPUTE", True):
        return
    try:
        from apps.runs.models import EvaluationRun

        from .runner import LangFairMetricJobRunner

        run = EvaluationRun.objects.select_related("evaluation").get(pk=run_id)
        if run.status != EvaluationRun.Status.COMPLETED:
            return
        suites, options = default_metric_request(run.evaluation)
        LangFairMetricJobRunner().request(run, suites, options)
    except Exception:  # noqa: BLE001 - logged, the run itself already succeeded
        logger.exception("Automatic LangFair computation could not be started for run %s", run_id)
