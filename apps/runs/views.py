from typing import Any, cast

from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.evaluations.models import Evaluation
from apps.llms.models import LLMConfig
from apps.llms.views import build_default_llm_config
from apps.prompts.models import PromptPopulation

from .models import EvaluationRun
from .serializers import RunStatusSerializer, RunUpdateSerializer, StartRunSerializer, llm_snapshot
from .services import ACTIVE_STATUSES, EvaluationRunService, reap_orphaned_runs


class StartRunView(APIView):
    def get(self, request, evaluation_id):
        """Every run of the evaluation, newest first (the run history)."""
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        reap_orphaned_runs(evaluation_id=evaluation.id)
        runs = evaluation.runs.annotate(
            failed_generations=Count("responses", filter=Q(responses__error__isnull=False))
        ).order_by("-id")
        return Response(RunStatusSerializer(runs, many=True).data)

    def post(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)

        params = StartRunSerializer(data=request.data)
        params.is_valid(raise_exception=True)

        validated_data = cast(dict[str, Any], params.validated_data)
        resume_run_id = validated_data.get("resume_run_id")
        prompt_population_id = validated_data.get("prompt_population_id")
        external = validated_data["external"]
        label = validated_data.get("label", "").strip()

        # Runs left RUNNING by a server restart become FAILED (resumable) here.
        reap_orphaned_runs(evaluation_id=evaluation.id)

        if resume_run_id is not None:
            run = get_object_or_404(
                EvaluationRun,
                pk=resume_run_id,
                evaluation=evaluation,
            )
            if external or run.external:
                return Response(
                    {"error": "Failed external runs cannot be resumed here."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if run.status != EvaluationRun.Status.FAILED:
                return Response(
                    {"error": "Only failed runs can be resumed."},
                    status=status.HTTP_409_CONFLICT,
                )
            try:
                llm_config = evaluation.llm_config
            except ObjectDoesNotExist:
                llm_config = build_default_llm_config(evaluation, save=True)
            if not llm_config.model_name:
                return Response(
                    {"error": "TARGET_LLM_MODEL_NAME is not set."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if (
                not run.total_prompts
                or run.total_generations % run.total_prompts
                or llm_config.generations_per_prompt
                != run.total_generations // run.total_prompts
            ):
                return Response(
                    {"error": "The saved run uses a different generations-per-prompt setting."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            with transaction.atomic():
                if self._has_active_run(evaluation):
                    return self._active_run_conflict()
                # Atomic FAILED -> RUNNING transition: concurrent resumes start one thread.
                claimed = EvaluationRun.objects.filter(
                    pk=run.pk, status=EvaluationRun.Status.FAILED
                ).update(
                    status=EvaluationRun.Status.RUNNING,
                    error_message="",
                    finished_at=None,
                    heartbeat_at=timezone.now(),
                )
                if not claimed:
                    return Response(
                        {"error": "This run is no longer failed; it may already have been resumed."},
                        status=status.HTTP_409_CONFLICT,
                    )
                evaluation.status = Evaluation.Status.RUNNING
                evaluation.save(update_fields=["status", "updated_at"])
            run.refresh_from_db()
            # Runs created before snapshots existed get the resuming config.
            if not run.llm_snapshot.get("modelName"):
                run.llm_snapshot = llm_snapshot(llm_config)
                run.save(update_fields=["llm_snapshot"])
            EvaluationRunService().start(run, llm_config)
            return Response(RunStatusSerializer(run).data, status=status.HTTP_201_CREATED)

        if prompt_population_id is not None:
            population = get_object_or_404(
                PromptPopulation,
                pk=prompt_population_id,
                evaluation=evaluation,
            )
            if population.status != PromptPopulation.Status.COMPLETED:
                return Response(
                    {"error": "The prompt population has not completed generation."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if not population.actual_prompt_count:
                return Response(
                    {"error": "The prompt population has no prompts."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            population = evaluation.prompt_populations.filter(
                status=PromptPopulation.Status.COMPLETED,
                actual_prompt_count__gt=0,
            ).order_by("-created_at", "-id").first()
            if population is None:
                return Response(
                    {"error": "No completed prompt population exists for this evaluation."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        if external:
            # External bots are not called through llm_config, but its
            # generationsPerPrompt still decides how many answers each prompt needs.
            llm_config = LLMConfig.objects.filter(evaluation=evaluation).first()
        else:
            try:
                llm_config = evaluation.llm_config
            except ObjectDoesNotExist:
                llm_config = build_default_llm_config(evaluation, save=True)

            if not llm_config.model_name:
                return Response(
                    {"error": "TARGET_LLM_MODEL_NAME is not set."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # Check + create in one write transaction so two concurrent starts
        # cannot both create a run (SQLite transaction_mode is IMMEDIATE).
        with transaction.atomic():
            if self._has_active_run(evaluation):
                return self._active_run_conflict()

            run = evaluation.runs.create(
                prompt_population=population,
                status=EvaluationRun.Status.RUNNING,
                external=external,
                total_prompts=population.actual_prompt_count,
                total_generations=(
                    population.actual_prompt_count
                    * (llm_config.generations_per_prompt if llm_config else 1)
                ),
                llm_snapshot=llm_snapshot(llm_config, external=external),
                label=label,
            )

            evaluation.status = Evaluation.Status.RUNNING
            evaluation.save(update_fields=["status", "updated_at"])

        if not external:
            EvaluationRunService().start(run, llm_config)

        return Response(RunStatusSerializer(run).data, status=status.HTTP_201_CREATED)

    @staticmethod
    def _has_active_run(evaluation):
        return evaluation.runs.filter(status__in=ACTIVE_STATUSES).exists()

    @staticmethod
    def _active_run_conflict():
        return Response(
            {"error": "A run is already in progress for this evaluation."},
            status=status.HTTP_409_CONFLICT,
        )


class RunStatusView(APIView):
    def get(self, request, run_id):
        run = get_object_or_404(EvaluationRun, pk=run_id)
        if reap_orphaned_runs(evaluation_id=run.evaluation_id):
            run.refresh_from_db()
        return Response(RunStatusSerializer(run).data)


class RunDetailView(APIView):
    """Rename (PATCH {label}) or delete (with its responses) a past run."""

    def patch(self, request, run_id):
        run = get_object_or_404(EvaluationRun, pk=run_id)
        params = RunUpdateSerializer(data=request.data)
        params.is_valid(raise_exception=True)
        run.label = cast(dict[str, Any], params.validated_data)["label"].strip()
        run.save(update_fields=["label"])
        return Response(RunStatusSerializer(run).data)

    def delete(self, request, run_id):
        with transaction.atomic():
            run = get_object_or_404(EvaluationRun.objects.select_for_update(), pk=run_id)
            if run.status in ACTIVE_STATUSES:
                return Response(
                    {"error": "An active run cannot be deleted."},
                    status=status.HTTP_409_CONFLICT,
                )
            run.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class LatestRunStatusView(APIView):
    def get(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        reap_orphaned_runs(evaluation_id=evaluation.id)
        run = evaluation.runs.first()
        if run is None:
            return Response({"detail": "No runs found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(RunStatusSerializer(run).data)
