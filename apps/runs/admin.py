from django.contrib import admin

from apps.runs.models import EvaluationRun


@admin.register(EvaluationRun)
class EvaluationRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "evaluation",
        "prompt_population",
        "status",
        "completed_prompts",
        "total_prompts",
        "completed_generations",
        "total_generations",
        "started_at",
        "finished_at",
    )
    list_filter = ("status",)
    search_fields = ("evaluation__name", "error_message")
    readonly_fields = ("started_at", "finished_at")
