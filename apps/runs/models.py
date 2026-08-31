from django.db import models

from apps.evaluations.models import Evaluation
from apps.prompts.models import PromptPopulation


class EvaluationRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    evaluation = models.ForeignKey(
        Evaluation, related_name="runs", on_delete=models.CASCADE
    )
    prompt_population = models.ForeignKey(
        PromptPopulation, related_name="runs", on_delete=models.CASCADE
    )
    status = models.CharField(
        max_length=32, choices=Status.choices, default=Status.PENDING
    )
    total_prompts = models.PositiveIntegerField(default=0)
    completed_prompts = models.PositiveIntegerField(default=0)
    total_generations = models.PositiveIntegerField(default=0)
    completed_generations = models.PositiveIntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-id"]

    def __str__(self):
        return f"EvaluationRun #{self.id}"
