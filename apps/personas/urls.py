# pyright: reportMissingImports=false
from django.urls import path

from .views import PersonaAttributesView, PersonaDetailView, PersonaListView, PersonaSampleView

urlpatterns = [
    path("persona/attributes", PersonaAttributesView.as_view(), name="persona-attributes-no-slash"),
    path("persona/attributes/", PersonaAttributesView.as_view(), name="persona-attributes"),
    path("persona/personas", PersonaListView.as_view(), name="persona-list-no-slash"),
    path("persona/personas/", PersonaListView.as_view(), name="persona-list"),
    path("persona/personas/<str:persona_id>", PersonaDetailView.as_view(), name="persona-detail-no-slash"),
    path("persona/personas/<str:persona_id>/", PersonaDetailView.as_view(), name="persona-detail"),
    path("persona/sample", PersonaSampleView.as_view(), name="persona-sample-no-slash"),
    path("persona/sample/", PersonaSampleView.as_view(), name="persona-sample"),
]
