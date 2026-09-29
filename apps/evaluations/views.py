# pyright: reportAttributeAccessIssue=false
from django.db import connection
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView
from rest_framework.response import Response
from rest_framework import status

from .models import Evaluation
from .serializers import EvaluationSerializer
from .task_input import TaskInputError, parse_task_file


class EvaluationListCreateView(ListCreateAPIView):
    queryset = Evaluation.objects.all()
    serializer_class = EvaluationSerializer

    def create(self, request, *args, **kwargs):
        # Convert multipart QueryDict to a plain dict before inserting the
        # parsed list; assigning a list directly to QueryDict adds one nesting level.
        data = request.data.dict() if hasattr(request.data, "dict") else request.data.copy()
        upload = request.FILES.get("task_file")
        if upload:
            try:
                data["taskObjectives"] = parse_task_file(upload)
                if not data.get("useCaseDescription"):
                    data["useCaseDescription"] = "Uploaded task objectives"
            except TaskInputError as exc:
                return Response({"task_file": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
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
