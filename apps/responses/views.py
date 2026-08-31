# pyright: reportAttributeAccessIssue=false
from django.db.models import Q
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.evaluations.models import Evaluation
from apps.runs.models import EvaluationRun

from .serializers import LLMResponseSerializer


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
