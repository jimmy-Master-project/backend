# pyright: reportAttributeAccessIssue=false
from django.db import connection
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView

from .models import Evaluation
from .serializers import EvaluationSerializer


class EvaluationListCreateView(ListCreateAPIView):
    queryset = Evaluation.objects.all()
    serializer_class = EvaluationSerializer


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
