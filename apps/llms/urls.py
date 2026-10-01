from django.urls import path

from .views import LLMConfigTestConnectionView, LLMConfigView, LLMModelListView

urlpatterns = [
    path(
        "evaluations/<int:evaluation_id>/llm-config/models/",
        LLMModelListView.as_view(),
        name="llm-config-models",
    ),
    path(
        "evaluations/<int:evaluation_id>/llm-config/test-connection",
        LLMConfigTestConnectionView.as_view(),
        name="llm-config-test-connection-no-slash",
    ),
    path(
        "evaluations/<int:evaluation_id>/llm-config/test-connection/",
        LLMConfigTestConnectionView.as_view(),
        name="llm-config-test-connection",
    ),
    path(
        "evaluations/<int:evaluation_id>/llm-config",
        LLMConfigView.as_view(),
        name="llm-config-no-slash",
    ),
    path(
        "evaluations/<int:evaluation_id>/llm-config/",
        LLMConfigView.as_view(),
        name="llm-config",
    ),
]
