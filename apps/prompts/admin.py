from django.contrib import admin

from apps.prompts.models import Prompt, PromptPopulation


class PromptInline(admin.TabularInline):
    model = Prompt
    extra = 0
    fields = ("sequence", "category", "text")


@admin.register(PromptPopulation)
class PromptPopulationAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "evaluation",
        "provider",
        "model_name",
        "status",
        "requested_prompt_count",
        "actual_prompt_count",
        "created_at",
    )
    list_filter = ("provider", "status")
    search_fields = ("evaluation__name", "model_name", "additional_instruction")
    readonly_fields = ("created_at",)
    inlines = [PromptInline]


@admin.register(Prompt)
class PromptAdmin(admin.ModelAdmin):
    list_display = ("id", "population", "sequence", "category", "created_at")
    list_filter = ("category",)
    search_fields = ("text", "category", "population__evaluation__name")
    readonly_fields = ("created_at",)
