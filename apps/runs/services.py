# pyright: reportAttributeAccessIssue=false, reportMissingImports=false
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import timedelta

from django.conf import settings
from django.db import connections
from django.utils import timezone as django_timezone

from apps.evaluations.models import Evaluation
from apps.llms.clients.factory import build_llm_client
from apps.llms.models import LLMConfig
from apps.prompts.models import Prompt
from apps.responses.models import LLMResponse
from apps.prompts.services import (
    append_instruction,
    build_classification_output_instruction,
    build_recommendation_output_instruction,
)

from .models import EvaluationRun

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = (EvaluationRun.Status.PENDING, EvaluationRun.Status.RUNNING)
INTERRUPTED_MESSAGE = "interrupted (server restarted)"
EXTERNAL_STALE_MESSAGE = "external bot inactive (no responses received)"
# A run that was just created/resumed may not have registered its thread yet.
ORPHAN_GRACE_SECONDS = 30
# Persist progress counters at most this often (plus once at the end).
PROGRESS_SAVE_INTERVAL_SECONDS = 1.0

# Ids of runs whose worker thread is alive in *this* process. After a server
# restart this is empty, which is how orphaned runs are detected.
_live_runs: set[int] = set()
_live_runs_lock = threading.Lock()


def is_run_thread_alive(run_id) -> bool:
    with _live_runs_lock:
        return run_id in _live_runs


def reap_orphaned_runs(evaluation_id=None) -> int:
    """Mark active runs that can no longer make progress as FAILED.

    * Worker-thread runs (``external=False``) with no live thread in this
      process: the server restarted (or the thread died) mid-run.
    * External-bot runs are never reaped for lacking a thread; they are only
      failed after ``EXTERNAL_RUN_STALE_HOURS`` without any POSTed responses.

    Assumes a single Django server process (``runserver --noreload``); the
    thread registry is per-process.
    """
    now = django_timezone.now()
    queryset = EvaluationRun.objects.filter(status__in=ACTIVE_STATUSES)
    if evaluation_id is not None:
        queryset = queryset.filter(evaluation_id=evaluation_id)

    grace_cutoff = now - timedelta(seconds=ORPHAN_GRACE_SECONDS)
    with _live_runs_lock:
        live = set(_live_runs)
    orphan_ids = [
        run_id
        for run_id, heartbeat_at in queryset.filter(external=False).values_list(
            "id", "heartbeat_at"
        )
        if run_id not in live and (heartbeat_at is None or heartbeat_at < grace_cutoff)
    ]

    stale_cutoff = now - timedelta(hours=settings.EXTERNAL_RUN_STALE_HOURS)
    stale_external_ids = list(
        queryset.filter(external=True)
        .exclude(heartbeat_at__gte=stale_cutoff)
        .values_list("id", flat=True)
    )

    reaped = 0
    for run_ids, message in (
        (orphan_ids, INTERRUPTED_MESSAGE),
        (stale_external_ids, EXTERNAL_STALE_MESSAGE),
    ):
        if not run_ids:
            continue
        # Re-check status in the UPDATE so a concurrent transition wins.
        targets = EvaluationRun.objects.filter(pk__in=run_ids, status__in=ACTIVE_STATUSES)
        evaluation_ids = list(targets.values_list("evaluation_id", flat=True))
        count = targets.update(
            status=EvaluationRun.Status.FAILED,
            error_message=message,
            finished_at=now,
        )
        if count:
            logger.warning("Marked run(s) %s as failed: %s", run_ids, message)
            Evaluation.objects.filter(
                pk__in=evaluation_ids, status=Evaluation.Status.RUNNING
            ).update(status=Evaluation.Status.FAILED, updated_at=now)
        reaped += count
    return reaped


def build_prompt_text(prompt, evaluation):
    prompt_text = str(prompt.text)
    if evaluation.task_type == evaluation.TaskType.CLASSIFICATION:
        prompt_text = append_instruction(
            prompt_text,
            build_classification_output_instruction(evaluation),
        )
    elif evaluation.task_type == evaluation.TaskType.RECOMMENDATION:
        instruction = build_recommendation_output_instruction(evaluation)
        if instruction:
            prompt_text = append_instruction(prompt_text, instruction)
    return prompt_text


class SystemicRunError(RuntimeError):
    pass


class EvaluationRunService:
    def start(self, run, llm_config):
        # Register before the thread starts so the reaper never sees a gap.
        with _live_runs_lock:
            _live_runs.add(run.id)
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(run.id, llm_config.id),
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            with _live_runs_lock:
                _live_runs.discard(run.id)
            raise

    def _run_in_thread(self, run_id, llm_config_id):
        try:
            self._execute(run_id, llm_config_id)
        finally:
            with _live_runs_lock:
                _live_runs.discard(run_id)
            connections.close_all()

    def _execute(self, run_id, llm_config_id):
        try:
            self._execute_run(run_id, llm_config_id)
        except Exception as exc:
            logger.exception("Evaluation run %s failed", run_id)
            self._mark_failed(run_id, exc)

    def _mark_failed(self, run_id, exc):
        try:
            now = django_timezone.now()
            EvaluationRun.objects.filter(pk=run_id).update(
                status=EvaluationRun.Status.FAILED,
                error_message=str(exc) or exc.__class__.__name__,
                finished_at=now,
            )
            Evaluation.objects.filter(runs__id=run_id).update(
                status=Evaluation.Status.FAILED, updated_at=now
            )
        except Exception:
            logger.exception("Could not mark evaluation run %s as failed", run_id)

    def _execute_run(self, run_id, llm_config_id):
        run = EvaluationRun.objects.select_related("evaluation").get(pk=run_id)
        llm_config = LLMConfig.objects.get(pk=llm_config_id)

        now = django_timezone.now()
        run.status = EvaluationRun.Status.RUNNING
        run.started_at = now
        run.heartbeat_at = now
        run.save(update_fields=["status", "started_at", "heartbeat_at"])

        client = build_llm_client(llm_config)
        prompts = list(
            Prompt.objects.filter(population=run.prompt_population).order_by("sequence")
        )

        # Generations that already succeeded (resume); errored ones are retried.
        done = set(
            LLMResponse.objects.filter(run=run, error__isnull=True).values_list(
                "prompt_id", "generation_index"
            )
        )
        tasks = []
        for prompt in prompts:
            # Instruction text is identical for every generation of a prompt.
            prompt_text = build_prompt_text(prompt, run.evaluation)
            for generation_index in range(1, llm_config.generations_per_prompt + 1):
                if (prompt.id, generation_index) not in done:
                    tasks.append((prompt, generation_index, prompt_text))

        self._save_progress(run, done)
        failfast = settings.TARGET_LLM_FAILFAST_GENERATIONS
        attempted = 0
        failed = 0
        last_progress_save = time.monotonic()

        for prompt, generation_index, result, error in self._generate_all(client, tasks):
            attempted += 1
            if error is None:
                LLMResponse.objects.update_or_create(
                    run=run,
                    prompt=prompt,
                    generation_index=generation_index,
                    defaults={
                        "response_text": result["text"],
                        "input_tokens": result.get("input_tokens"),
                        "output_tokens": result.get("output_tokens"),
                        "latency_ms": result.get("latency_ms"),
                        "error": None,
                    },
                )
                done.add((prompt.id, generation_index))
            else:
                failed += 1
                logger.warning(
                    "Run %s prompt %s generation %s failed: %s",
                    run.id, prompt.id, generation_index, error,
                )
                LLMResponse.objects.update_or_create(
                    run=run,
                    prompt=prompt,
                    generation_index=generation_index,
                    defaults={
                        "response_text": "",
                        "input_tokens": None,
                        "output_tokens": None,
                        "latency_ms": None,
                        "error": error,
                    },
                )
                # Systemic failure: every one of the first N attempts failed
                # (bad base URL, model name, credentials, ...).
                if failed == attempted and attempted >= min(failfast, len(tasks)):
                    raise SystemicRunError(
                        f"The first {attempted} generation(s) all failed; last error: {error}"
                    )

            if time.monotonic() - last_progress_save >= PROGRESS_SAVE_INTERVAL_SECONDS:
                self._save_progress(run, done)
                last_progress_save = time.monotonic()

        self._save_progress(run, done)
        now = django_timezone.now()
        # Only complete a run that is still ours (it may have been reaped).
        completed = EvaluationRun.objects.filter(
            pk=run.id, status=EvaluationRun.Status.RUNNING
        ).update(status=EvaluationRun.Status.COMPLETED, finished_at=now, heartbeat_at=now)
        if completed:
            run.evaluation.status = run.evaluation.Status.COMPLETED
            run.evaluation.save(update_fields=["status", "updated_at"])
        if failed:
            logger.warning(
                "Run %s finished with %s failed generation(s) out of %s attempted",
                run.id, failed, attempted,
            )

    def _save_progress(self, run, done):
        # completed_* count successful generations only, derived from the set
        # of stored (prompt_id, generation_index) pairs so resume never double counts.
        run.completed_generations = len(done)
        run.completed_prompts = len({prompt_id for prompt_id, _ in done})
        run.heartbeat_at = django_timezone.now()
        run.save(update_fields=["completed_generations", "completed_prompts", "heartbeat_at"])

    def _generate_all(self, client, tasks):
        """Yield (prompt, generation_index, result, error) for every task.

        LLM calls may run in a thread pool (TARGET_LLM_CONCURRENCY); results
        are yielded to the caller's thread, which does all DB writes.
        """

        def call(task):
            prompt, generation_index, prompt_text = task
            try:
                return prompt, generation_index, client.generate(prompt_text), None
            except Exception as exc:
                return prompt, generation_index, None, str(exc) or exc.__class__.__name__

        concurrency = max(int(settings.TARGET_LLM_CONCURRENCY), 1)
        if concurrency == 1:
            for task in tasks:
                yield call(task)
            return

        executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="run-llm")
        try:
            pending = set()
            iterator = iter(tasks)
            # Keep a bounded number of calls in flight so an abort stops quickly.
            for task in iterator:
                pending.add(executor.submit(call, task))
                if len(pending) >= concurrency:
                    break
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    yield future.result()
                    next_task = next(iterator, None)
                    if next_task is not None:
                        pending.add(executor.submit(call, next_task))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
