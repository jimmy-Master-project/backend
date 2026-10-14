from rest_framework import serializers

from apps.llms.models import LLMConfig

from .models import EvaluationRun


def llm_snapshot(llm_config: LLMConfig | None, external: bool = False) -> dict:
    """Target LLM settings worth keeping on a run (no credentials)."""
    if llm_config is None:
        return {"external": external}
    return {
        "provider": llm_config.provider,
        "modelName": llm_config.model_name,
        "baseUrl": llm_config.base_url,
        "temperature": llm_config.temperature,
        "topP": llm_config.top_p,
        "maxTokens": llm_config.max_tokens,
        "generationsPerPrompt": llm_config.generations_per_prompt,
        "external": external,
    }


class StartRunSerializer(serializers.Serializer):
    promptPopulationId = serializers.IntegerField(required=False, source="prompt_population_id")
    external = serializers.BooleanField(default=False)
    resumeRunId = serializers.IntegerField(required=False, source="resume_run_id")
    label = serializers.CharField(required=False, allow_blank=True, max_length=200, default="")


class RunUpdateSerializer(serializers.Serializer):
    label = serializers.CharField(allow_blank=True, max_length=200)


class RunStatusSerializer(serializers.ModelSerializer):
    totalPrompts = serializers.IntegerField(source="total_prompts", read_only=True)
    completedPrompts = serializers.IntegerField(source="completed_prompts", read_only=True)
    totalGenerations = serializers.IntegerField(source="total_generations", read_only=True)
    completedGenerations = serializers.IntegerField(source="completed_generations", read_only=True)
    failedGenerations = serializers.SerializerMethodField()
    progress = serializers.SerializerMethodField()
    errorMessage = serializers.CharField(source="error_message", read_only=True)
    heartbeatAt = serializers.DateTimeField(source="heartbeat_at", read_only=True)
    startedAt = serializers.DateTimeField(source="started_at", read_only=True)
    finishedAt = serializers.DateTimeField(source="finished_at", read_only=True)
    llmSnapshot = serializers.JSONField(source="llm_snapshot", read_only=True)
    promptPopulationId = serializers.IntegerField(source="prompt_population_id", read_only=True)

    class Meta:
        model = EvaluationRun
        fields = [
            "id",
            "status",
            "totalPrompts",
            "completedPrompts",
            "totalGenerations",
            "completedGenerations",
            "failedGenerations",
            "progress",
            "errorMessage",
            "external",
            "heartbeatAt",
            "startedAt",
            "finishedAt",
            "label",
            "llmSnapshot",
            "promptPopulationId",
        ]

    def get_progress(self, obj):
        if not obj.total_generations:
            return 0.0
        return round(obj.completed_generations / obj.total_generations, 4)

    def get_failedGenerations(self, obj):
        # Generations whose target LLM call failed after retries.
        annotated = getattr(obj, "failed_generations", None)
        if annotated is not None:
            return annotated
        return obj.responses.filter(error__isnull=False).count()
