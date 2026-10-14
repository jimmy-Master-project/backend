from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path


def health(request):
    # Cheap readiness probe for start_services.sh; touches no database.
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/health/", health, name="health"),
    path("api/", include("apps.evaluations.urls")),
    path("api/", include("apps.prompts.urls")),
    path("api/", include("apps.llms.urls")),
    path("api/", include("apps.runs.urls")),
    path("api/", include("apps.responses.urls")),
    path("api/", include("apps.fairness.urls")),
    path("api/", include("apps.personas.urls")),
]
