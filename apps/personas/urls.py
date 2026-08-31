# pyright: reportMissingImports=false
from django.urls import path

from .views import PersonaAttributesView, PersonaSampleView

urlpatterns = [
    path("persona/attributes", PersonaAttributesView.as_view(), name="persona-attributes-no-slash"),
    path("persona/attributes/", PersonaAttributesView.as_view(), name="persona-attributes"),
    path("persona/sample", PersonaSampleView.as_view(), name="persona-sample-no-slash"),
    path("persona/sample/", PersonaSampleView.as_view(), name="persona-sample"),
]
