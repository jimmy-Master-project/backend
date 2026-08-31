from django.urls import path

from .views import EvaluationDetailView, EvaluationListCreateView

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
]
