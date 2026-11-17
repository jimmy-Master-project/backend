from django.urls import path

from .views import EvaluationDetailView, EvaluationListCreateView, EvaluationTestPlanView

urlpatterns = [
    path("evaluations", EvaluationListCreateView.as_view(), name="evaluation-list-no-slash"),
    path("evaluations/", EvaluationListCreateView.as_view(), name="evaluation-list"),
    path(
        "evaluations/<int:pk>",
        EvaluationDetailView.as_view(),
        name="evaluation-detail-no-slash",
    ),
    path(
        "evaluations/<int:pk>/",
        EvaluationDetailView.as_view(),
        name="evaluation-detail",
    ),
    path(
        "evaluations/<int:pk>/test-plan",
        EvaluationTestPlanView.as_view(),
        name="evaluation-test-plan-no-slash",
    ),
    path(
        "evaluations/<int:pk>/test-plan/",
        EvaluationTestPlanView.as_view(),
        name="evaluation-test-plan",
    ),
]
