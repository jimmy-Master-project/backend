from django.contrib import admin

from apps.llms.models import LLMConfig


@admin.register(LLMConfig)
class LLMConfigAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "evaluation",
        "provider",
        "model_name",
        "base_url",
        "temperature",
        "generations_per_prompt",
    )
    list_filter = ("provider",)
    search_fields = ("evaluation__name", "model_name", "base_url")
    readonly_fields = ("created_at", "updated_at")
