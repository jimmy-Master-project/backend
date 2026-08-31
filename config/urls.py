from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("apps.evaluations.urls")),
    path("api/", include("apps.prompts.urls")),
    path("api/", include("apps.llms.urls")),
    path("api/", include("apps.runs.urls")),
    path("api/", include("apps.responses.urls")),
    path("api/", include("apps.fairness.urls")),
    path("api/", include("apps.personas.urls")),
    path("api/", include("apps.benchmark.urls")),
]
