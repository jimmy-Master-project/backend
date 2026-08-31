# pyright: reportMissingImports=false
from typing import Any, cast

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.runs.models import EvaluationRun

from apps.fairness.serializers import LangFairRunMetricsSerializer
from apps.fairness.services import LangFairMetricService


class RunLangFairMetricsView(APIView):
    def get(self, request, run_id):
        run = get_object_or_404(EvaluationRun, pk=run_id)
        suites = request.query_params.getlist("suite") or None

        try:
            result = LangFairMetricService().evaluate_run(run=run, suites=suites)
        except (AssertionError, TypeError, ValueError) as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(result)

    def post(self, request, run_id):
        run = get_object_or_404(EvaluationRun, pk=run_id)
        serializer = LangFairRunMetricsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        validated_data = cast(dict[str, Any], serializer.validated_data)
        suites = validated_data.get("suites") or None
        options = validated_data.get("options") or {}

        try:
            result = LangFairMetricService().evaluate_run(
                run=run,
                suites=suites,
                options=options,
            )
        except (AssertionError, TypeError, ValueError) as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(result)
