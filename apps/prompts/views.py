# pyright: reportAttributeAccessIssue=false, reportOptionalSubscript=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportMissingImports=false
from typing import Any, cast

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.generics import RetrieveUpdateDestroyAPIView
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.evaluations.models import Evaluation

from .models import Prompt, PromptPopulation
from .services import PromptPopulationGenerationRunner
from .services import get_task_objectives
from .serializers import (
    GeneratePromptPopulationSerializer,
    PromptCreateSerializer,
    PromptPopulationSerializer,
    PromptSerializer,
)


def _latest_population(evaluation):
    # Prefer the latest completed population for read endpoints. A browser refresh,
    # request abort, or server restart can leave a newer population stuck in
    # GENERATING with no prompts; using it would make the UI look empty/failed
    # even though a completed population exists.
    return (
        evaluation.prompt_populations.filter(status=PromptPopulation.Status.COMPLETED)
        .order_by("-created_at")
        .first()
        or evaluation.prompt_populations.order_by("-created_at").first()
    )


def _serialize_prompt(prompt):
    return {
        "id": str(prompt.id),
        "index": prompt.sequence,
        "prompt": prompt.text,
        "text": prompt.text,
        "category": prompt.category,
        "persona_id": prompt.metadata.get("persona_id"),
        "persona_attributes": prompt.metadata.get("persona_attributes", {}),
        "missing_attributes": prompt.metadata.get("missing_attributes", []),
        "fairness": prompt.metadata.get("fairness", {}),
    }


def _serialize_population_summary(population):
    if population is None:
        return {
            "generationModel": "",
            "provider": "",
            "temperature": 0,
            "requested": 0,
            "generated": 0,
        }
    metadata = population.metadata if isinstance(population.metadata, dict) else {}
    persona = metadata.get("persona") if isinstance(metadata.get("persona"), dict) else {}
    fairness = metadata.get("fairness") if isinstance(metadata.get("fairness"), dict) else {}
    return {
        "generationModel": population.model_name,
        "provider": population.provider,
        "temperature": population.temperature,
        "requested": population.requested_prompt_count,
        "generated": population.actual_prompt_count,
        "personaBased": bool(persona.get("enabled")),
        "persona": persona or None,
        "personaAttributes": persona.get("selected_attributes", []),
        "stratifiedBy": persona.get("stratify"),
        "seed": persona.get("sampling_seed"),
        "fairness": fairness,
    }


class GeneratePromptPopulationView(APIView):
    def post(self, request, evaluation_id=None):
        if evaluation_id is None:
            task = (request.data.get("task") or "").strip()
            if not task:
                return Response({"task": "This field is required."}, status=status.HTTP_400_BAD_REQUEST)
            evaluation = Evaluation.objects.create(
                name=task[:255],
                use_case_description=task,
                task_type=Evaluation.TaskType.TEXT_GENERATION,
            )
        else:
            evaluation = get_object_or_404(Evaluation, pk=evaluation_id)

        params = GeneratePromptPopulationSerializer(data=request.data)
        params.is_valid(raise_exception=True)
        data = cast(dict[str, Any], params.validated_data)

        objective_count = len(get_task_objectives(evaluation))
        persona = data.get("persona") or {}
        counterfactual = bool(persona.get("enabled")) and (
            bool(persona.get("counterfactual_enabled"))
            or any(attribute in {"gender", "gender_identity"} for attribute in persona.get("selected_attributes", []))
        )
        requested_prompt_count = data["prompt_count"] * objective_count * (2 if counterfactual else 1)
        population = PromptPopulation.objects.create(
            evaluation=evaluation,
            provider=data["provider"],
            model_name=data["model_name"],
            temperature=data["temperature"],
            requested_prompt_count=requested_prompt_count,
            additional_instruction=data.get("additional_instruction", ""),
            status=PromptPopulation.Status.GENERATING,
        )
        evaluation.status = Evaluation.Status.GENERATING_PROMPTS
        evaluation.save(update_fields=["status", "updated_at"])

        if data.get("task"):
            evaluation.use_case_description = data["task"]
            evaluation.save(update_fields=["use_case_description", "updated_at"])

        # Generation calls one LLM request per prompt (or per persona for
        # counterfactual pairs), so it can run well past a typical HTTP client
        # timeout for larger populations. Run it in the background -- same
        # pattern as EvaluationRunService -- and let the frontend poll
        # PromptPopulation.status via GET /prompt-populations/{id}/.
        PromptPopulationGenerationRunner().start(evaluation, data, population)

        return Response(
            {
                "population_id": population.id,
                "status": population.status,
                "requested_prompt_count": population.requested_prompt_count,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class PromptPopulationDetailView(APIView):
    def get(self, request, pk):
        population = get_object_or_404(PromptPopulation, pk=pk)
        return Response(PromptPopulationSerializer(population).data)


class EvaluationPromptPopulationView(APIView):
    def get(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        return Response(_serialize_population_summary(_latest_population(evaluation)))


class EvaluationPromptListCreateView(APIView):
    def get(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        population = _latest_population(evaluation)
        if population is None:
            return Response([])
        prompts = population.prompts.order_by("sequence")
        return Response([_serialize_prompt(prompt) for prompt in prompts])

    def post(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        population = _latest_population(evaluation)
        if population is None:
            population = PromptPopulation.objects.create(
                evaluation=evaluation,
                provider="manual",
                model_name="manual",
                temperature=0,
                requested_prompt_count=0,
                status=PromptPopulation.Status.COMPLETED,
            )

        text = (request.data.get("prompt") or request.data.get("text") or "").strip()
        category = (request.data.get("category") or "").strip()
        if not text:
            return Response({"prompt": "This field is required."}, status=status.HTTP_400_BAD_REQUEST)

        prompt = Prompt.objects.create(
            population=population,
            text=text,
            category=category,
            sequence=population.prompts.count() + 1,
            metadata={"source": "manual"},
        )
        population.actual_prompt_count = population.prompts.count()
        population.save(update_fields=["actual_prompt_count"])
        return Response(_serialize_prompt(prompt), status=status.HTTP_201_CREATED)


class EvaluationPromptDetailView(APIView):
    def patch(self, request, evaluation_id, pk):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        prompt = get_object_or_404(Prompt, pk=pk, population__evaluation=evaluation)
        if "prompt" in request.data:
            prompt.text = request.data["prompt"]
        if "text" in request.data:
            prompt.text = request.data["text"]
        if "category" in request.data:
            prompt.category = request.data["category"]
        prompt.save(update_fields=["text", "category"])
        return Response(_serialize_prompt(prompt))

    def delete(self, request, evaluation_id, pk):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        prompt = get_object_or_404(Prompt, pk=pk, population__evaluation=evaluation)
        population = prompt.population
        prompt.delete()
        population.actual_prompt_count = population.prompts.count()
        population.save(update_fields=["actual_prompt_count"])
        return Response(status=status.HTTP_204_NO_CONTENT)


class PromptListCreateView(APIView):
    def get(self, request, population_id):
        population = get_object_or_404(PromptPopulation, pk=population_id)
        prompts = population.prompts.all()
        return Response(
            {
                "count": prompts.count(),
                "results": PromptSerializer(prompts, many=True).data,
            }
        )

    def post(self, request, population_id):
        population = get_object_or_404(PromptPopulation, pk=population_id)
        serializer = PromptCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        next_sequence = (population.prompts.count()) + 1
        prompt = Prompt.objects.create(
            population=population,
            text=serializer.validated_data["text"],
            category=serializer.validated_data.get("category", ""),
            sequence=next_sequence,
            metadata={"source": "manual"},
        )
        population.actual_prompt_count = population.prompts.count()
        population.save(update_fields=["actual_prompt_count"])

        return Response(
            PromptSerializer(prompt).data, status=status.HTTP_201_CREATED
        )


class PromptDetailView(RetrieveUpdateDestroyAPIView):
    queryset = Prompt.objects.all()
    serializer_class = PromptSerializer

    def perform_destroy(self, instance):
        population = instance.population
        instance.delete()
        population.actual_prompt_count = population.prompts.count()
        population.save(update_fields=["actual_prompt_count"])
