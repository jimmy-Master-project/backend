from django.urls import path

from .views import LatestRunStatusView, RunDetailView, RunStatusView, StartRunView

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
    path("runs/<int:run_id>", RunDetailView.as_view(), name="run-detail-no-slash"),
    path("runs/<int:run_id>/", RunDetailView.as_view(), name="run-detail"),
    path("runs/<int:run_id>/status", RunStatusView.as_view(), name="run-status-no-slash"),
    path("runs/<int:run_id>/status/", RunStatusView.as_view(), name="run-status"),
    path(
        "evaluations/<int:evaluation_id>/runs/latest/",
        LatestRunStatusView.as_view(),
        name="latest-run-status",
    ),
]
