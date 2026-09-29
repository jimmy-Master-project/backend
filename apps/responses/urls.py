from django.urls import path

from .views import EvaluationResponseListView, RunResponseDeleteView, RunResponseListView

urlpatterns = [
    path(
        "evaluations/<int:evaluation_id>/responses",
        EvaluationResponseListView.as_view(),
        name="evaluation-responses-no-slash",
    ),
    path(
        "evaluations/<int:evaluation_id>/responses/",
        EvaluationResponseListView.as_view(),
        name="evaluation-responses",
    ),
    path(
        "runs/<int:run_id>/responses",
        RunResponseListView.as_view(),
        name="run-responses-no-slash",
    ),
    path(
        "runs/<int:run_id>/responses/",
        RunResponseListView.as_view(),
        name="run-responses",
    ),
    path(
        "runs/<int:run_id>/responses/<int:response_id>",
        RunResponseDeleteView.as_view(),
        name="run-response-delete-no-slash",
    ),
    path(
        "runs/<int:run_id>/responses/<int:response_id>/",
        RunResponseDeleteView.as_view(),
        name="run-response-delete",
    ),
]
