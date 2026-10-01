# pyright: reportAttributeAccessIssue=false
"""Background (threaded) LangFair computation with an in-process job registry.

Same pattern as EvaluationRunService / PromptPopulationGenerationRunner: the
request thread records a ``running`` row and returns immediately; a daemon
thread computes and stores the result. Rows left ``pending``/``running`` with
no live thread in this process (server restart, crash) are marked ``failed``
the next time they are looked at. Note: the registry is per process, so this
assumes a single backend process (as in ``runserver`` / a single worker).
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from django.db import IntegrityError, connections, transaction
from django.utils import timezone

from apps.fairness.models import LangFairMetricResult

from . import cache
from .service import LangFairMetricService, PreparedRun

logger = logging.getLogger(__name__)

_ACTIVE_JOBS: set[int] = set()
_REGISTRY_LOCK = threading.Lock()
IN_FLIGHT = (LangFairMetricResult.Status.PENDING, LangFairMetricResult.Status.RUNNING)
INTERRUPTED_MESSAGE = "Computation was interrupted (server restarted or worker died). Request it again."


def _run_meta(run: Any) -> dict[str, Any]:
    run_status = str(getattr(run, "status", ""))
    completed = getattr(getattr(run, "Status", None), "COMPLETED", "completed")
    return {"runId": run.id, "runStatus": run_status, "partial": run_status != str(completed)}


def reap_stale_jobs(run_id: int | None = None) -> int:
    """Mark in-flight rows without a live thread in this process as failed."""
    with _REGISTRY_LOCK:
        queryset = LangFairMetricResult.objects.filter(status__in=IN_FLIGHT)
        if run_id is not None:
            queryset = queryset.filter(run_id=run_id)
        if _ACTIVE_JOBS:
            queryset = queryset.exclude(id__in=list(_ACTIVE_JOBS))
        return queryset.update(
            status=LangFairMetricResult.Status.FAILED, error=INTERRUPTED_MESSAGE, updated_at=timezone.now()
        )


class LangFairMetricJobRunner:
    def __init__(self, service: LangFairMetricService | None = None) -> None:
        self.service = service or LangFairMetricService()

    # ---- read side (GET) --------------------------------------------
    def get_status(
        self,
        run: Any,
        suites: Sequence[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Never computes. Returns completed payload / running / failed / none."""
        reap_stale_jobs(run.id)
        prepared = self.service.prepare_run(run, suites, options)
        row = cache.find_result(run.id, prepared.cache_key)
        return self._payload_for_row(row, run)

    # ---- write side (POST) ------------------------------------------
    def request(
        self,
        run: Any,
        suites: Sequence[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Return (http_status, payload): 200 when cached, 202 when (already) running."""
        reap_stale_jobs(run.id)
        prepared = self.service.prepare_run(run, suites, options)
        with _REGISTRY_LOCK:
            row = cache.find_result(run.id, prepared.cache_key)
            if row is not None and row.status == LangFairMetricResult.Status.COMPLETED:
                return 200, self._payload_for_row(row, run)
            if row is not None and row.status in IN_FLIGHT and row.id in _ACTIVE_JOBS:
                return 202, self._payload_for_row(row, run)
            row = self._claim_row(run.id, prepared.cache_key, row)
            _ACTIVE_JOBS.add(row.id)
        self._spawn(row.id, prepared)
        return 202, self._payload_for_row(row, run)

    def _claim_row(self, run_id: int, cache_key: str, row: LangFairMetricResult | None) -> LangFairMetricResult:
        if row is None:
            try:
                with transaction.atomic():
                    return LangFairMetricResult.objects.create(
                        run_id=run_id,
                        cache_key=cache_key,
                        status=LangFairMetricResult.Status.RUNNING,
                        result={},
                    )
            except IntegrityError:
                row = LangFairMetricResult.objects.get(run_id=run_id, cache_key=cache_key)
        row.status = LangFairMetricResult.Status.RUNNING
        row.error = ""
        row.result = {}
        row.save(update_fields=["status", "error", "result", "updated_at"])
        return row

    def _spawn(self, row_id: int, prepared: PreparedRun) -> None:
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(row_id, prepared),
            daemon=True,
            name=f"langfair-metrics-{row_id}",
        )
        # Start only after the request's transaction (if any) commits, so the
        # worker thread can see the row it is about to update.
        transaction.on_commit(thread.start)

    def _run_in_thread(self, row_id: int, prepared: PreparedRun) -> None:
        try:
            result = self.service.compute(prepared)
            LangFairMetricResult.objects.filter(pk=row_id).update(
                status=LangFairMetricResult.Status.COMPLETED,
                result=result,
                error="",
                updated_at=timezone.now(),
            )
            cache.prune_results(prepared.run.id)
        except Exception as exc:  # any failure must surface as status=failed
            logger.exception("LangFair computation failed for result #%s", row_id)
            LangFairMetricResult.objects.filter(pk=row_id).update(
                status=LangFairMetricResult.Status.FAILED,
                error=(str(exc).strip() or exc.__class__.__name__)[:4000],
                updated_at=timezone.now(),
            )
        finally:
            with _REGISTRY_LOCK:
                _ACTIVE_JOBS.discard(row_id)
            connections.close_all()

    # ---- payloads -----------------------------------------------------
    def _payload_for_row(self, row: LangFairMetricResult | None, run: Any) -> dict[str, Any]:
        meta = _run_meta(run)
        if row is None:
            return {"status": "none", **meta}
        if row.status == LangFairMetricResult.Status.COMPLETED:
            return {**cache.sanitized_result(row), "status": "completed", **meta, "computedAt": row.updated_at}
        if row.status == LangFairMetricResult.Status.FAILED:
            return {"status": "failed", "error": row.error or "Unknown error", **meta}
        return {"status": "running", **meta}
