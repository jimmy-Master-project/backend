from django.urls import path

from .views import BoldValidationView


urlpatterns = [
    path("benchmark/bold", BoldValidationView.as_view()),
    path("benchmark/bold/", BoldValidationView.as_view()),
]
