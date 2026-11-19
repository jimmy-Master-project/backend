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
        "systemPrompt": llm_config.system_prompt,
        "external": external,
    }


class StartRunSerializer(serializers.Serializer):
    promptPopulationId = serializers.IntegerField(required=False, source="prompt_population_id")
    external = serializers.BooleanField(default=False)
    label = serializers.CharField(required=False, allow_blank=True, max_length=200, default="")

    def validate(self, attrs):
        # Resuming was removed on purpose (experiment protocol): mixing responses
        # from two executions of a run, possibly days apart, could bias results.
        # Reject instead of silently starting a new run.
        if "resumeRunId" in self.initial_data:
            raise serializers.ValidationError(
                {"resumeRunId": "Resuming runs is not supported. Start a new run instead."}
            )
        return attrs


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
    condition = serializers.SerializerMethodField()

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
            "condition",
        ]

    def get_condition(self, obj):
        """Experiment condition of the population this run sent (for comparing runs)."""
        population = obj.prompt_population
        metadata = population.metadata if isinstance(population.metadata, dict) else {}
        persona = metadata.get("persona") if isinstance(metadata.get("persona"), dict) else {}
        raw_objectives = metadata.get("task_objectives_raw") or []
        strategies = sorted(
            {str(item["strategy"]) for item in raw_objectives if isinstance(item, dict) and item.get("strategy")}
        )
        objectives = metadata.get("task_objectives")
        return {
            "populationName": population.name,
            "personaBased": bool(persona.get("enabled")),
            "personaSetName": persona.get("persona_set_name") or "",
            "generationMode": str((metadata.get("prompt_generator") or {}).get("mode") or ""),
            "objectiveCount": len(objectives) if isinstance(objectives, list) else None,
            "directionStrategies": strategies,
        }

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
