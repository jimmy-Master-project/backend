# pyright: reportAttributeAccessIssue=false
import json

import openai
from django.db import connection
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView
from rest_framework.response import Response
from rest_framework import status

from .models import Evaluation
from apps.prompt_agents.clients.openai_compatible import PromptAgentError
from apps.prompt_agents.planner import TestPlanningError, TestPlanningService
from apps.prompt_agents.services import PromptGenerationService, UnsupportedProviderError

from .serializers import EvaluationSerializer, GenerateTestPlanSerializer
from .task_input import TaskInputError, parse_task_file

MULTIPART_JSON_FIELDS = ("classificationLabels", "recommendationItems", "testDirections", "taskObjectives")


class EvaluationListCreateView(ListCreateAPIView):
    queryset = Evaluation.objects.all()
    serializer_class = EvaluationSerializer

    def create(self, request, *args, **kwargs):
        # Convert multipart QueryDict to a plain dict before inserting the
        # parsed list; assigning a list directly to QueryDict adds one nesting level.
        data = request.data.dict() if hasattr(request.data, "dict") else request.data.copy()
        if hasattr(request.data, "dict"):
            # Multipart forms can only carry strings; the frontend JSON-encodes
            # list fields (labels, candidate items) alongside the task file.
            for key in MULTIPART_JSON_FIELDS:
                if isinstance(data.get(key), str):
                    try:
                        data[key] = json.loads(data[key])
                    except json.JSONDecodeError:
                        return Response(
                            {key: "Must be a JSON array."}, status=status.HTTP_400_BAD_REQUEST
                        )
        upload = request.FILES.get("taskFile")
        if upload:
            # The uploaded file holds the user's test directions (seed tasks);
            # they are appended to any typed in the form.
            try:
                typed = data.get("testDirections") or []
                if not isinstance(typed, list):
                    typed = [typed]
                data["testDirections"] = [*typed, *(item["objective"] for item in parse_task_file(upload))]
            except TaskInputError as exc:
                return Response({"taskFile": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class EvaluationDetailView(RetrieveUpdateDestroyAPIView):
    queryset = Evaluation.objects.all()
    serializer_class = EvaluationSerializer

    def perform_destroy(self, instance):
        delete_legacy_classifier_results(instance.id)
        instance.delete()


def delete_legacy_classifier_results(evaluation_id):
    """Remove rows from the removed classifiers app before cascading responses.

    Older local SQLite databases may still have apps_classifiers_classificationresult
    with a foreign key to responses. Since the classifiers app/model was removed,
    Django's cascade collector no longer knows to delete those rows first.
    """
    table_names = connection.introspection.table_names()
    with connection.cursor() as cursor:
        if "classifiers_classificationresult" in table_names:
            cursor.execute(
                """
                DELETE FROM classifiers_classificationresult
                WHERE response_id IN (
                    SELECT response.id
                    FROM responses_llmresponse AS response
                    INNER JOIN runs_evaluationrun AS run
                        ON response.run_id = run.id
                    WHERE run.evaluation_id = %s
                )
                """,
                [evaluation_id],
            )
        if "apps_classifiers_classificationresult" in table_names:
            cursor.execute(
                """
                DELETE FROM apps_classifiers_classificationresult
                WHERE response_id IN (
                    SELECT response.id
                    FROM responses_llmresponse AS response
                    INNER JOIN runs_evaluationrun AS run
                        ON response.run_id = run.id
                    WHERE run.evaluation_id = %s
                )
                """,
                [evaluation_id],
            )


class EvaluationTestPlanView(APIView):
    """Initial prompt agent: plan question directions (task objectives).

    Inputs are user-defined when the evaluation is created: the target LLM
    description (GHL's requirement document) and the test directions
    (Self-Instruct seed tasks). Runs GHL's two steps (strategy/technique
    selection, then directions per technique) with HELM's fairness/bias
    desiderata as the test strategy, expanding every test direction.
    With ``apply`` (default true) the directions replace the evaluation's
    task objectives, so the next prompt population generation uses them.
    """

    def post(self, request, pk):
        evaluation = get_object_or_404(Evaluation, pk=pk)
        if not evaluation.test_directions:
            return Response(
                {"testDirections": "This evaluation has no test directions. Add them to the evaluation first."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        serializer = GenerateTestPlanSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            client = PromptGenerationService()._build_client(
                data["provider"], data["model_name"], data.get("base_url") or None, data["temperature"]
            )
            plan = TestPlanningService(client).plan(
                evaluation,
                target_info=data.get("target_info"),
                protected_attributes=data.get("protected_attributes") or None,
                total_directions=data["total_directions"],
            )
        except UnsupportedProviderError as exc:
            return Response({"provider": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except (TestPlanningError, PromptAgentError, openai.OpenAIError) as exc:
            return Response({"detail": f"Test planning failed: {exc}"}, status=status.HTTP_502_BAD_GATEWAY)

        if data["apply"]:
            evaluation.task_objectives = plan["directions"]
            evaluation.save(update_fields=["task_objectives", "updated_at"])

        return Response(
            {
                "evaluationId": evaluation.id,
                "applied": data["apply"],
                "strategySummary": plan["strategy_summary"],
                "protectedAttributes": plan["protected_attributes"],
                "targetInfo": plan["target_info"],
                "testDirections": plan["test_directions"],
                "techniques": plan["techniques"],
                "directions": plan["directions"],
                "rejected": plan["rejected"],
                "coverage": plan["coverage"],
                "gapFill": plan["gap_fill"],
            }
        )
