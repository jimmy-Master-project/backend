# pyright: reportAttributeAccessIssue=false
from django.db import transaction
from django.conf import settings
from django.db.models import Q
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.evaluations.models import Evaluation
from apps.runs.models import EvaluationRun
from apps.fairness.services import LangFairMetricService

from .serializers import ExternalResponsesSerializer, LLMResponseSerializer


def _serialize_response_item(response, model_name=""):
    prompt = response.prompt
    return {
        "id": str(response.id),
        "promptId": str(prompt.id),
        "promptIndex": prompt.sequence,
        "promptText": prompt.text,
        "promptCategory": prompt.category,
        "response": response.response_text,
        "generationIndex": response.generation_index,
        "model": model_name,
        "inputTokens": response.input_tokens or 0,
        "outputTokens": response.output_tokens or 0,
        "latencyMs": response.latency_ms or 0,
    }


class EvaluationResponseListView(APIView):
    def get(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        run = evaluation.runs.order_by("-id").first()
        if run is None:
            return Response({
                "items": [],
                "total": 0,
                "model": "",
                "promptCount": 0,
                "runId": None,
                "taskType": evaluation.task_type,
                "language": evaluation.language,
                "classificationLabels": evaluation.get_classification_labels(),
            })

        queryset = run.responses.select_related("prompt")

        keyword = request.query_params.get("keyword")
        if keyword:
            queryset = queryset.filter(
                Q(prompt__text__icontains=keyword) | Q(response_text__icontains=keyword)
            )

        category = request.query_params.get("category")
        if category:
            queryset = queryset.filter(prompt__category=category)

        generation_index = request.query_params.get("generationIndex")
        if generation_index:
            try:
                generation_index_value = int(generation_index)
            except (TypeError, ValueError):
                return Response(
                    {"error": "generationIndex must be an integer."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            queryset = queryset.filter(generation_index=generation_index_value)

        total = queryset.count()
        try:
            page = max(int(request.query_params.get("page", 1)), 1)
            page_size = max(int(request.query_params.get("pageSize", 20)), 1)
        except ValueError:
            return Response(
                {"error": "page and pageSize must be integers."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        start = (page - 1) * page_size
        end = start + page_size

        model_name = getattr(getattr(evaluation, "llm_config", None), "model_name", "")
        items = [_serialize_response_item(response, model_name) for response in queryset[start:end]]

        return Response(
            {
                "items": items,
                "total": total,
                "model": model_name,
                "promptCount": run.prompt_population.actual_prompt_count,
                "runId": run.id,
                "taskType": evaluation.task_type,
                "language": evaluation.language,
                "classificationLabels": evaluation.get_classification_labels(),
            }
        )


class RunResponseListView(APIView):
    def post(self, request, run_id):
        expected_key = getattr(settings, "EXTERNAL_RESPONSE_API_KEY", "")
        if expected_key and request.headers.get("X-API-Key") != expected_key:
            return Response({"error": "Invalid API key."}, status=status.HTTP_401_UNAUTHORIZED)
        run = get_object_or_404(EvaluationRun, pk=run_id)
        serializer = ExternalResponsesSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        prompts = {
            prompt.id: prompt
            for prompt in run.prompt_population.prompts.all()
        }
        resolved_items = []
        for item in data["responses"]:
            prompt = prompts.get(item["promptId"])
            if prompt is None:
                return Response(
                    {"error": f"promptId {item['promptId']} does not belong to this run."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            resolved_items.append((item, prompt))

        with transaction.atomic():
            created = []
            for item, prompt in resolved_items:
                created.append(
                    LLMResponse.objects.create(
                        run=run,
                        prompt=prompt,
                        generation_index=item["generationIndex"],
                        response_text=item["response"],
                        input_tokens=item.get("inputTokens"),
                        output_tokens=item.get("outputTokens"),
                        latency_ms=item.get("latencyMs"),
                    )
                )

            run.completed_generations = run.responses.count()
            run.completed_prompts = run.responses.values("prompt_id").distinct().count()
            update_fields = ["completed_generations", "completed_prompts"]
            if run.completed_generations >= run.total_generations:
                from django.utils import timezone

                run.status = EvaluationRun.Status.COMPLETED
                run.finished_at = timezone.now()
                update_fields += ["status", "finished_at"]
                run.evaluation.status = Evaluation.Status.COMPLETED
                run.evaluation.save(update_fields=["status", "updated_at"])
            run.save(update_fields=update_fields)

        result = {"stored": len(created), "responseIds": [row.id for row in created]}
        if data["calculate"]:
            try:
                result["metrics"] = LangFairMetricService().evaluate_run(
                    run=run,
                    suites=data.get("suites"),
                    options=data.get("options"),
                )
            except (AssertionError, TypeError, ValueError) as exc:
                return Response({"error": str(exc), **result}, status=status.HTTP_400_BAD_REQUEST)
        return Response(result, status=status.HTTP_201_CREATED)

    def get(self, request, run_id):
        run = get_object_or_404(EvaluationRun, pk=run_id)
        queryset = run.responses.select_related("prompt")

        prompt_id = request.query_params.get("prompt_id")
        if prompt_id:
            if not prompt_id.isdigit():
                return Response(
                    {"error": "prompt_id must be an integer."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            queryset = queryset.filter(prompt_id=prompt_id)

        category = request.query_params.get("category")
        if category:
            queryset = queryset.filter(prompt__category=category)

        return Response(
            {
                "count": queryset.count(),
                "results": LLMResponseSerializer(queryset, many=True).data,
            }
        )


class RunResponseDeleteView(APIView):
    def delete(self, request, run_id, response_id):
        with transaction.atomic():
            run = get_object_or_404(
                EvaluationRun.objects.select_for_update(), pk=run_id
            )
            if run.status == EvaluationRun.Status.RUNNING:
                return Response(
                    {"error": "Responses cannot be deleted while a run is active."},
                    status=status.HTTP_409_CONFLICT,
                )
            response = get_object_or_404(run.responses, pk=response_id)
            response.delete()
            run.completed_generations = run.responses.count()
            run.completed_prompts = run.responses.values("prompt_id").distinct().count()
            run.save(update_fields=["completed_generations", "completed_prompts"])
        return Response(status=status.HTTP_204_NO_CONTENT)
