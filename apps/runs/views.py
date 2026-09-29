from typing import Any, cast

from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.evaluations.models import Evaluation
from apps.llms.views import build_default_llm_config
from apps.prompts.models import PromptPopulation

from .models import EvaluationRun
from .serializers import RunStatusSerializer, StartRunSerializer
from .services import EvaluationRunService


class StartRunView(APIView):
    def post(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)

        params = StartRunSerializer(data=request.data)
        params.is_valid(raise_exception=True)

        validated_data = cast(dict[str, Any], params.validated_data)
        resume_run_id = validated_data.get("resume_run_id")
        prompt_population_id = validated_data.get("prompt_population_id")
        external = validated_data["external"]

        if resume_run_id is not None:
            run = get_object_or_404(
                EvaluationRun,
                pk=resume_run_id,
                evaluation=evaluation,
                status=EvaluationRun.Status.FAILED,
            )
            if external:
                return Response(
                    {"error": "Failed external runs cannot be resumed here."},
                    status=status.HTTP_400_BAD_REQUEST,
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
            run.status = EvaluationRun.Status.RUNNING
            run.error_message = ""
            run.finished_at = None
            run.save(update_fields=["status", "error_message", "finished_at"])
            evaluation.status = Evaluation.Status.RUNNING
            evaluation.save(update_fields=["status", "updated_at"])
            EvaluationRunService().start(run, llm_config)
            return Response({"run_id": run.id, "status": run.status}, status=status.HTTP_201_CREATED)

        if prompt_population_id is not None:
            population = get_object_or_404(
                PromptPopulation,
                pk=prompt_population_id,
                evaluation=evaluation,
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
            llm_config = None
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

        run = evaluation.runs.create(
            prompt_population=population,
            status=EvaluationRun.Status.RUNNING,
            total_prompts=population.actual_prompt_count,
            total_generations=(
                population.actual_prompt_count
                * (llm_config.generations_per_prompt if llm_config else 1)
            ),
        )

        evaluation.status = Evaluation.Status.RUNNING
        evaluation.save(update_fields=["status", "updated_at"])

        if not external:
            EvaluationRunService().start(run, llm_config)

        return Response(
            {"run_id": run.id, "status": run.status},
            status=status.HTTP_201_CREATED,
        )


class RunStatusView(APIView):
    def get(self, request, run_id):
        run = get_object_or_404(EvaluationRun, pk=run_id)
        return Response(RunStatusSerializer(run).data)


class LatestRunStatusView(APIView):
    def get(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        run = evaluation.runs.first()
        if run is None:
            return Response({"detail": "No runs found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(RunStatusSerializer(run).data)
