"""Runs one experiment end to end in a background thread, resumably.

Phases (each skips work already stored, so "start" after a restart resumes):
1. sample personas (shared by every group);
2. per group: directions (A) / baseline generation (B) -> base questions;
3. identity versions of every valid base question;
4. multi-turn conversations with the target for every pending version;
5. metrics per group x comparison, then CSV / JSON files.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

from django.conf import settings
from django.db import connections
from django.utils import timezone

from apps.personas.services import PersonaService

from . import questions as Q
from . import strategies
from .checklist import classify_scenario
from .conversation import TargetClient
from .identity import IDENTITY_KEYS, build_versions, persona_identity
from .models import BaseQuestion, Experiment, GroupGeneration, QuestionVersion

logger = logging.getLogger(__name__)

GROUPS: dict[str, str] = {
    "A1": "直接生成策略＋虛擬人",
    "A2": "範例迭代擴展策略＋虛擬人",
    "A3": "規則與條件組合策略＋虛擬人",
    "A4": "測試技術引導策略＋虛擬人",
    "B1": "BiasAsker 參考改編",
    "B2": "BiasTestGPT 參考改編",
    "B3": "GenFair 參考改編",
    "B4": "LangBiTe 參考改編",
    "B5": "CAFFE 參考改編",
}

_live: set[int] = set()
_stop: set[int] = set()
_lock = threading.Lock()


class StopRequested(Exception):
    pass


def is_running(experiment_id: int) -> bool:
    with _lock:
        return experiment_id in _live


def request_stop(experiment_id: int) -> None:
    with _lock:
        _stop.add(experiment_id)


def start(experiment: Experiment) -> bool:
    with _lock:
        if experiment.id in _live:
            return False
        _live.add(experiment.id)
        _stop.discard(experiment.id)
    threading.Thread(target=_run_thread, args=(experiment.id,), daemon=True).start()
    return True


def reap_orphans() -> None:
    """Experiments left 'running' by a restarted server become 'stopped' (resumable)."""
    active = [s for s in Experiment.Status.values if s not in ("completed", "failed", "stopped", "pending")]
    with _lock:
        live = set(_live)
    Experiment.objects.filter(status__in=active).exclude(id__in=live).update(
        status=Experiment.Status.STOPPED, error_message="interrupted (server restarted); start again to resume"
    )


def _run_thread(experiment_id: int) -> None:
    try:
        ExperimentRunner(experiment_id).run()
    finally:
        with _lock:
            _live.discard(experiment_id)
            _stop.discard(experiment_id)
        connections.close_all()


class ExperimentRunner:
    def __init__(self, experiment_id: int):
        self.experiment = Experiment.objects.select_related("evaluation").get(pk=experiment_id)

    # -- plumbing ------------------------------------------------------------

    def _set(self, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self.experiment, key, value)
        self.experiment.heartbeat_at = timezone.now()
        self.experiment.save(update_fields=[*fields, "heartbeat_at"])

    def _progress(self, **values: Any) -> None:
        self._set(progress={**(self.experiment.progress or {}), **values})

    def _check_stop(self) -> None:
        with _lock:
            stop = self.experiment.id in _stop
        if stop:
            raise StopRequested()

    def run(self) -> None:
        exp = self.experiment
        if not exp.started_at:
            self._set(started_at=timezone.now())
        self._set(error_message="")
        try:
            self._set(status=Experiment.Status.SAMPLING)
            self.sample_personas()
            self._set(status=Experiment.Status.GENERATING)
            self.generate_groups()
            self.build_versions()
            self._set(status=Experiment.Status.QUERYING)
            self.query_target()
            self._set(status=Experiment.Status.COMPUTING)
            from .export import export_results

            output_dir = export_results(exp)
            self._set(status=Experiment.Status.COMPLETED, output_dir=str(output_dir), finished_at=timezone.now())
        except StopRequested:
            self._set(status=Experiment.Status.STOPPED, error_message="stopped by user; start again to resume")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Experiment %s failed", exp.id)
            self._set(status=Experiment.Status.FAILED, error_message=str(exc) or exc.__class__.__name__)

    # -- 1. personas -----------------------------------------------------------

    def sample_personas(self) -> None:
        exp = self.experiment
        needed = exp.questions_per_group
        if len(exp.personas) >= needed:
            return
        seed = int(exp.config.get("seed") or random.SystemRandom().randint(0, 2_000_000_000))
        exp.config = {**exp.config, "seed": seed}
        service = PersonaService()
        chosen: list[dict[str, Any]] = []
        seen: set[str] = set()
        rejected = 0
        for attempt in range(20):
            self._check_stop()
            sample = service.sample_projected_personas(
                sample_size=needed * 2,
                selected_attributes=list(IDENTITY_KEYS),
                pool=exp.config.get("pool") or None,
                seed=seed + attempt * 1000,
                allow_partial=True,
            )
            for persona in sample.get("personas") or []:
                if persona["persona_id"] in seen:
                    continue
                seen.add(persona["persona_id"])
                identity = persona_identity(persona.get("attributes") or {})
                if identity is None:  # under 35, non-binary or no race label
                    rejected += 1
                    continue
                chosen.append({"persona_id": persona["persona_id"], "attributes": persona["attributes"], "identity": identity})
                if len(chosen) >= needed:
                    break
            self._progress(personas=len(chosen), personas_rejected=rejected)
            if len(chosen) >= needed:
                break
        if len(chosen) < needed:
            raise RuntimeError(f"Only {len(chosen)} eligible personas found (need {needed}).")
        exp.personas = chosen[:needed]
        exp.save(update_fields=["personas", "config"])

    def races(self) -> list[str]:
        return sorted({p["identity"]["race"] for p in self.experiment.personas})

    # -- 2. questions ----------------------------------------------------------

    def generate_groups(self) -> None:
        exp = self.experiment
        for group in exp.groups:
            self._check_stop()
            generation, _ = GroupGeneration.objects.get_or_create(experiment=exp, group=group)
            if generation.status == "completed":
                continue
            self._progress(current_group=group)
            BaseQuestion.objects.filter(experiment=exp, group=group).delete()
            llm = Q.GenerationLLM()
            started = time.monotonic()
            directions: list[str] = []
            extra: dict[str, Any] = {}
            seed = int(exp.config.get("seed", 0))
            personas = exp.personas
            if group.startswith("A"):
                plan = strategies.STRATEGIES[group](exp.evaluation, llm)
                directions = plan["directions"]
                extra = {k: v for k, v in plan.items() if k != "directions"}
                results = Q.persona_questions(exp.evaluation, llm, personas, directions)
            elif group == "B1":
                results = Q.biasasker_questions(personas, seed)
            elif group == "B2":
                results = Q.biastestgpt_questions(personas, llm, seed)
            elif group == "B3":
                results = Q.genfair_questions(personas)
            elif group == "B4":
                results, templates = Q.langbite_questions(personas, llm, seed)
                extra = {"templates": templates}
            elif group == "B5":
                results = Q.caffe_questions(personas, llm, exp.evaluation.use_case_description[:300], seed)
            else:
                raise ValueError(f"Unknown group {group}")
            BaseQuestion.objects.bulk_create(
                [
                    BaseQuestion(
                        experiment=exp,
                        group=group,
                        index=i,
                        persona_id=personas[i]["persona_id"],
                        identity=personas[i]["identity"],
                        text=r["text"],
                        scenario=classify_scenario(r["text"]),
                        valid=r["valid"],
                        invalid_reason=r["invalid_reason"],
                        metadata=r["metadata"],
                    )
                    for i, r in enumerate(results)
                ]
            )
            generation.directions = directions
            generation.stats = {
                "llm_calls": llm.calls,
                "seconds": round(time.monotonic() - started, 1),
                "generated": sum(r["valid"] for r in results),
                "invalid": sum(not r["valid"] for r in results),
                **extra,
            }
            generation.status = "completed"
            generation.save()

    # -- 3. versions -----------------------------------------------------------

    def build_versions(self) -> None:
        races = self.races()
        pending = BaseQuestion.objects.filter(experiment=self.experiment, valid=True, versions__isnull=True)
        for question in pending:
            QuestionVersion.objects.bulk_create(
                [
                    QuestionVersion(
                        question=question,
                        code=v["code"],
                        gender=v["gender"] or "",
                        age=v["age"] or "",
                        race=v["race"] or "",
                        text=v["text"],
                    )
                    for v in build_versions(question.text, question.identity, races)
                ]
            )

    # -- 4. target ---------------------------------------------------------------

    def query_target(self) -> None:
        exp = self.experiment
        client = TargetClient(exp.target_provider, exp.target_base_url, exp.target_model)
        versions = QuestionVersion.objects.filter(question__experiment=exp)
        total = versions.count()
        consecutive_errors = 0
        max_consecutive = int(getattr(settings, "EXPERIMENT_MAX_CONSECUTIVE_ERRORS", 10))
        for version in versions.filter(status=QuestionVersion.Status.PENDING).order_by("question__group", "question__index", "id"):
            self._check_stop()
            try:
                result = client.converse(version.text)
            except Exception as exc:  # noqa: BLE001
                consecutive_errors += 1
                version.status = QuestionVersion.Status.ERROR
                version.error = str(exc)[:2000]
                version.answered_at = timezone.now()
                version.save(update_fields=["status", "error", "answered_at"])
                if consecutive_errors >= max_consecutive:
                    raise RuntimeError(f"{consecutive_errors} conversations in a row failed; last error: {exc}") from exc
            else:
                consecutive_errors = 0
                turns = result.get("turns") or []
                version.turns = turns
                version.final_answer = (turns[-1]["assistant"] if turns else "") or ""
                version.status = (
                    QuestionVersion.Status.FOLLOW_UP_EXCLUDED
                    if result.get("ended") == "max_turns"
                    else QuestionVersion.Status.ANSWERED
                )
                version.answered_at = timezone.now()
                version.save(update_fields=["turns", "final_answer", "status", "answered_at"])
            done = versions.exclude(status=QuestionVersion.Status.PENDING).count()
            self._progress(versions_total=total, versions_done=done)
        self._progress(versions_total=total, versions_done=versions.exclude(status=QuestionVersion.Status.PENDING).count())

    def retry_errors(self) -> int:
        """Put failed conversations back to pending (used by the 'resume' action)."""
        return QuestionVersion.objects.filter(
            question__experiment=self.experiment, status=QuestionVersion.Status.ERROR
        ).update(status=QuestionVersion.Status.PENDING, error="")
