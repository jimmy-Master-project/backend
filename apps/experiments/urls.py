from django.urls import re_path

from .views import (
    ExperimentDetailView,
    ExperimentFileView,
    ExperimentListView,
    ExperimentStartView,
    ExperimentStopView,
    TargetsView,
)

urlpatterns = [
    re_path(r"^experiments/?$", ExperimentListView.as_view(), name="experiment-list"),
    re_path(r"^experiments/targets/?$", TargetsView.as_view(), name="experiment-targets"),
    re_path(r"^experiments/(?P<experiment_id>\d+)/?$", ExperimentDetailView.as_view(), name="experiment-detail"),
    re_path(r"^experiments/(?P<experiment_id>\d+)/start/?$", ExperimentStartView.as_view(), name="experiment-start"),
    re_path(r"^experiments/(?P<experiment_id>\d+)/stop/?$", ExperimentStopView.as_view(), name="experiment-stop"),
    re_path(r"^experiments/(?P<experiment_id>\d+)/files/(?P<name>[\w.]+)$", ExperimentFileView.as_view(), name="experiment-file"),
]
