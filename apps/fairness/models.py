from django.db import models

from apps.runs.models import EvaluationRun


class LangFairMetricResult(models.Model):
    run = models.ForeignKey(
        EvaluationRun, related_name="langfair_results", on_delete=models.CASCADE
    )
    cache_key = models.CharField(max_length=64)
    result = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["run", "cache_key"], name="unique_langfair_result_cache"
            )
        ]
