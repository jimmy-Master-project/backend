from django.urls import path

from .views import RunStatusView, StartRunView

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
]
