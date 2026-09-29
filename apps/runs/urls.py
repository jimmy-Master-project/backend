from django.urls import path

from .views import LatestRunStatusView, RunStatusView, StartRunView

urlpatterns = [
    path(
        "evaluations/<int:evaluation_id>/runs",
        StartRunView.as_view(),
        name="run-start-no-slash",
    ),
    path(
        "evaluations/<int:evaluation_id>/runs/",
        StartRunView.as_view(),
        name="run-start",
    ),
    path("runs/<int:run_id>/status", RunStatusView.as_view(), name="run-status-no-slash"),
    path("runs/<int:run_id>/status/", RunStatusView.as_view(), name="run-status"),
    path(
        "evaluations/<int:evaluation_id>/runs/latest/",
        LatestRunStatusView.as_view(),
        name="latest-run-status",
    ),
]
