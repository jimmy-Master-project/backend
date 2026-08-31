# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
from django.db import models

from apps.evaluations.models import Evaluation


class PromptPopulation(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        GENERATING = "generating", "Generating"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    evaluation = models.ForeignKey(
        Evaluation, related_name="prompt_populations", on_delete=models.CASCADE
    )
    name = models.CharField(max_length=255, blank=True, default="")
    provider = models.CharField(max_length=64)
    model_name = models.CharField(max_length=128)
    temperature = models.FloatField(default=0.8)
    requested_prompt_count = models.PositiveIntegerField()
    actual_prompt_count = models.PositiveIntegerField(default=0)
    additional_instruction = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=32, choices=Status.choices, default=Status.PENDING
    )
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"PromptPopulation #{self.id} ({self.evaluation_id})"


class Prompt(models.Model):
    population = models.ForeignKey(
        PromptPopulation, related_name="prompts", on_delete=models.CASCADE
    )
    text = models.TextField()
    category = models.CharField(max_length=128, blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    sequence = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["sequence"]

    def __str__(self):
        return f"Prompt #{self.id}"
