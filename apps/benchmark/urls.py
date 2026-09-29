from django.urls import path

from .views import DialogSumValidationView


urlpatterns = [
    path("benchmark/dialogsum", DialogSumValidationView.as_view()),
    path("benchmark/dialogsum/", DialogSumValidationView.as_view()),
]
