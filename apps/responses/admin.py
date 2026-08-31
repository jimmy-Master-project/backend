from django.contrib import admin

from apps.responses.models import LLMResponse


@admin.register(LLMResponse)
class LLMResponseAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "run",
        "prompt",
        "generation_index",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "created_at",
    )
    list_filter = ("generation_index",)
    search_fields = ("response_text", "prompt__text", "run__evaluation__name")
    readonly_fields = ("created_at",)
