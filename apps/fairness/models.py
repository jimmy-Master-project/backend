from django.db import models

from apps.runs.models import EvaluationRun


class LangFairMetricResult(models.Model):
    """Cached LangFair computation for one (run, cache_key).

    ``cache_key`` hashes the response snapshot plus every computation input
    (suites, options, language, evaluation labels/items, langfair version), so
    a row is reused only when recomputing would give the same answer.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    run = models.ForeignKey(
        EvaluationRun, related_name="langfair_results", on_delete=models.CASCADE
    )
    cache_key = models.CharField(max_length=64)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True, default="")
    result = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["run", "cache_key"], name="unique_langfair_result_cache"
            )
        ]
