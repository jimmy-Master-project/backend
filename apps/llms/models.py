# pyright: reportArgumentType=false
from django.db import models

from apps.evaluations.models import Evaluation


class LLMConfig(models.Model):
    evaluation = models.OneToOneField(
        Evaluation, related_name="llm_config", on_delete=models.CASCADE
    )
    provider = models.CharField(max_length=64)
    model_name = models.CharField(max_length=128)
    base_url = models.CharField(max_length=255, blank=True, default="")
    api_key = models.CharField(max_length=512, blank=True, default="")
    temperature = models.FloatField(default=0.7)
    top_p = models.FloatField(default=1.0)
    max_tokens = models.PositiveIntegerField(default=512)
    generations_per_prompt = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"LLMConfig({self.provider}:{self.model_name})"
