from django.db import models

from apps.prompts.models import Prompt
from apps.runs.models import EvaluationRun


class LLMResponse(models.Model):
    run = models.ForeignKey(
        EvaluationRun, related_name="responses", on_delete=models.CASCADE
    )
    prompt = models.ForeignKey(
        Prompt, related_name="responses", on_delete=models.CASCADE
    )
    generation_index = models.PositiveIntegerField()
    response_text = models.TextField()
    input_tokens = models.PositiveIntegerField(null=True, blank=True)
    output_tokens = models.PositiveIntegerField(null=True, blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["prompt__sequence", "generation_index"]

    def __str__(self):
        return f"LLMResponse #{self.id}"
