from typing import Any, cast

from django.db import models


class Evaluation(models.Model):
    class Language(models.TextChoices):
        ENGLISH = "en", "English"
        AUTO = "auto", "Auto detect"

    class TaskType(models.TextChoices):
        TEXT_GENERATION = "text_generation", "Text Generation"
        CLASSIFICATION = "classification", "Classification"
        RECOMMENDATION = "recommendation", "Recommendation"
        OTHER = "other", "Other"

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        GENERATING_PROMPTS = "generating_prompts", "Generating Prompts"
        PROMPT_READY = "prompt_ready", "Prompt Ready"
        READY = "ready", "Ready"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    name = models.CharField(max_length=255)
    use_case_description = models.TextField()
    task_type = models.CharField(max_length=32, choices=TaskType.choices)
    language = models.CharField(max_length=8, choices=Language.choices, default=Language.ENGLISH)
    classification_labels = models.JSONField(default=list, blank=True)
    recommendation_items = models.JSONField(default=list, blank=True)
    task_objectives = models.JSONField(default=list, blank=True)
    status = models.CharField(
        max_length=32, choices=Status.choices, default=Status.DRAFT
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def get_classification_labels(self) -> list[dict[str, Any]]:
        if self.task_type != self.TaskType.CLASSIFICATION:
            return []
        labels = cast(list[dict[str, Any]], self.classification_labels or [])
        if labels:
            return labels
        return [
            {"value": 0, "name": "reject", "description": "不通過"},
            {"value": 1, "name": "approve", "description": "通過"},
        ]

    def get_recommendation_items(self) -> list[dict[str, Any]]:
        if self.task_type != self.TaskType.RECOMMENDATION:
            return []
        return cast(list[dict[str, Any]], self.recommendation_items or [])

    def __str__(self) -> str:
        return str(self.name)
