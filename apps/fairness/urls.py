# pyright: reportMissingImports=false
from django.urls import path

from apps.fairness.views import RunLangFairMetricsView

urlpatterns = [
    path(
        "runs/<int:run_id>/fairness-metrics",
        RunLangFairMetricsView.as_view(),
        name="run-fairness-metrics-no-slash",
    ),
    path(
        "runs/<int:run_id>/fairness-metrics/",
        RunLangFairMetricsView.as_view(),
        name="run-fairness-metrics",
    ),
]
