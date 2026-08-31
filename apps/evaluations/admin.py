from django.contrib import admin

from apps.evaluations.models import Evaluation


@admin.register(Evaluation)
class EvaluationAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "task_type", "status", "created_at", "updated_at")
    list_filter = ("task_type", "status")
    search_fields = ("name", "use_case_description")
    readonly_fields = ("created_at", "updated_at")
