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
        prompt_population_id = validated_data.get("prompt_population_id")
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
                population.actual_prompt_count * llm_config.generations_per_prompt
            ),
        )

        evaluation.status = Evaluation.Status.RUNNING
        evaluation.save(update_fields=["status", "updated_at"])

        EvaluationRunService().start(run, llm_config)

        return Response(
            {"run_id": run.id, "status": run.status},
            status=status.HTTP_201_CREATED,
        )


class RunStatusView(APIView):
    def get(self, request, run_id):
        run = get_object_or_404(EvaluationRun, pk=run_id)
        return Response(RunStatusSerializer(run).data)
