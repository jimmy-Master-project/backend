# pyright: reportMissingImports=false
"""Run-level LangFair metrics API (asynchronous).

POST /api/runs/{id}/fairness-metrics/  body: {"suites"?: [...], "options"?: {...}}
    200 {...payload, "status": "completed", "runStatus", "partial", "computedAt"}
        when a completed result for exactly these inputs is cached;
    202 {"status": "running", "runId", "runStatus", "partial"} otherwise
        (starts a background computation unless one is already running).
GET  /api/runs/{id}/fairness-metrics/?suite=a&suite=b[&options=<json>]
    never computes; always 200 with "status" in completed|running|failed|none.
"""
import json
from typing import Any, cast

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.runs.models import EvaluationRun

from apps.fairness.serializers import LangFairRunMetricsSerializer
from apps.fairness.services import LangFairMetricJobRunner

BAD_INPUT_ERRORS = (AssertionError, TypeError, ValueError)


class RunLangFairMetricsView(APIView):
    def get(self, request, run_id):
        run = get_object_or_404(EvaluationRun.objects.select_related("evaluation"), pk=run_id)
        raw_options = request.query_params.get("options")
        payload: dict[str, Any] = {"suites": request.query_params.getlist("suite") or None}
        if raw_options:
            try:
                payload["options"] = json.loads(raw_options)
            except json.JSONDecodeError:
                return Response({"error": "options must be valid JSON."}, status=status.HTTP_400_BAD_REQUEST)
        serializer = LangFairRunMetricsSerializer(data={k: v for k, v in payload.items() if v is not None})
        serializer.is_valid(raise_exception=True)
        validated = cast(dict[str, Any], serializer.validated_data)

        try:
            result = LangFairMetricJobRunner().get_status(
                run=run,
                suites=validated.get("suites") or None,
                options=validated.get("options") or {},
            )
        except BAD_INPUT_ERRORS as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(result)

    def post(self, request, run_id):
        run = get_object_or_404(EvaluationRun.objects.select_related("evaluation"), pk=run_id)
        serializer = LangFairRunMetricsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated = cast(dict[str, Any], serializer.validated_data)

        try:
            http_status, result = LangFairMetricJobRunner().request(
                run=run,
                suites=validated.get("suites") or None,
                options=validated.get("options") or {},
            )
        except BAD_INPUT_ERRORS as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(result, status=http_status)
