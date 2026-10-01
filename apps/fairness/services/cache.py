"""Persistence of LangFair results (``LangFairMetricResult``)."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from django.conf import settings

from apps.fairness.models import LangFairMetricResult

from .utils import _to_jsonable

# Bump whenever computation logic changes in a way that invalidates old rows
# (2: pairs keyed by objective + generation_index; error responses excluded).
METRICS_SCHEMA_VERSION = 2
DEFAULT_KEEP_PER_RUN = 5


@lru_cache(maxsize=1)
def langfair_version() -> str:
    try:
        from importlib.metadata import version

        return version("langfair")
    except Exception:  # pragma: no cover - metadata missing
        return "unknown"


def _langfair_cache_key(
    run_id: int,
    response_rows: Sequence[Any],
    suites: Sequence[str],
    options: Mapping[str, Any],
    language: str = "en",
    evaluation_inputs: Mapping[str, Any] | None = None,
) -> str:
    snapshot = {
        "schema": METRICS_SCHEMA_VERSION,
        "langfair": langfair_version(),
        "run": run_id,
        "responses": [
            {
                "id": row.id,
                "text": row.response_text,
                "generation": getattr(row, "generation_index", None),
                "prompt": row.prompt.text,
                "metadata": row.prompt.metadata,
            }
            for row in response_rows
        ],
        "suites": list(suites),
        "options": options,
        "language": language,
        "evaluation": dict(evaluation_inputs or {}),
    }
    encoded = json.dumps(snapshot, sort_keys=True, ensure_ascii=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def find_result(run_id: int, cache_key: str) -> LangFairMetricResult | None:
    return LangFairMetricResult.objects.filter(run_id=run_id, cache_key=cache_key).first()


def sanitized_result(row: Any) -> dict[str, Any]:
    """Rows written before NaN sanitization may contain NaN; clean on read too."""
    result = _to_jsonable(getattr(row, "result", None) or {})
    return result if isinstance(result, dict) else {"result": result}


def store_completed(run_id: int, cache_key: str, result: Mapping[str, Any]) -> None:
    LangFairMetricResult.objects.update_or_create(
        run_id=run_id,
        cache_key=cache_key,
        defaults={
            "result": dict(result),
            "status": LangFairMetricResult.Status.COMPLETED,
            "error": "",
        },
    )
    prune_results(run_id)


def prune_results(run_id: int, keep: int | None = None) -> None:
    """Keep only the newest few rows per run (never deletes in-flight rows)."""
    if keep is None:
        keep = int(getattr(settings, "LANGFAIR_RESULT_CACHE_KEEP_PER_RUN", DEFAULT_KEEP_PER_RUN))
    keep = max(1, keep)
    finished = LangFairMetricResult.objects.filter(run_id=run_id).exclude(
        status__in=[LangFairMetricResult.Status.PENDING, LangFairMetricResult.Status.RUNNING]
    )
    stale_ids = list(finished.order_by("-updated_at", "-id").values_list("id", flat=True)[keep:])
    if stale_ids:
        LangFairMetricResult.objects.filter(id__in=stale_ids).delete()
