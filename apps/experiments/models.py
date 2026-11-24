"""One experiment = one target chatbot x every group (A1-A4, B1-B5).

Stored step by step so a multi-day run can resume after a restart:
personas -> base questions per group -> identity versions -> conversations ->
metrics / exported files.
"""
from django.db import models

from apps.evaluations.models import Evaluation


class Experiment(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        SAMPLING = "sampling", "Sampling personas"
        GENERATING = "generating", "Generating questions"
        QUERYING = "querying", "Querying target"
        COMPUTING = "computing", "Computing metrics"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        STOPPED = "stopped", "Stopped"

    evaluation = models.ForeignKey(Evaluation, on_delete=models.PROTECT, related_name="experiments")
    name = models.CharField(max_length=255, blank=True)
    target_provider = models.CharField(max_length=64)
    target_base_url = models.CharField(max_length=512, blank=True)
    target_model = models.CharField(max_length=255, blank=True)
    questions_per_group = models.PositiveIntegerField(default=50)
    groups = models.JSONField(default=list)
    config = models.JSONField(default=dict, blank=True)
    personas = models.JSONField(default=list, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    error_message = models.TextField(blank=True)
    progress = models.JSONField(default=dict, blank=True)
    output_dir = models.CharField(max_length=1024, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    heartbeat_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-id"]


class GroupGeneration(models.Model):
    """Process data of one group (directions, LLM calls, time) for the RQ tables."""

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name="generations")
    group = models.CharField(max_length=8)
    status = models.CharField(max_length=16, default="pending")
    directions = models.JSONField(default=list, blank=True)
    stats = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        unique_together = [("experiment", "group")]


class BaseQuestion(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name="questions")
    group = models.CharField(max_length=8)
    index = models.PositiveIntegerField()
    persona_id = models.CharField(max_length=255, blank=True)
    identity = models.JSONField(default=dict)
    text = models.TextField(blank=True)
    scenario = models.CharField(max_length=64, blank=True)
    valid = models.BooleanField(default=True)
    invalid_reason = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = [("experiment", "group", "index")]
        ordering = ["group", "index"]


class QuestionVersion(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        ANSWERED = "answered", "Answered"
        FOLLOW_UP_EXCLUDED = "follow_up_excluded", "Still asking after max turns"
        ERROR = "error", "Error"

    question = models.ForeignKey(BaseQuestion, on_delete=models.CASCADE, related_name="versions")
    code = models.CharField(max_length=255)
    gender = models.CharField(max_length=32, blank=True)
    age = models.CharField(max_length=32, blank=True)
    race = models.CharField(max_length=128, blank=True)
    text = models.TextField()
    status = models.CharField(max_length=24, choices=Status.choices, default=Status.PENDING)
    turns = models.JSONField(default=list, blank=True)
    final_answer = models.TextField(blank=True)
    error = models.TextField(blank=True)
    answered_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [("question", "code")]
