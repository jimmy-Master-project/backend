from rest_framework import serializers

from .models import EvaluationRun


class StartRunSerializer(serializers.Serializer):
    promptPopulationId = serializers.IntegerField(required=False, source="prompt_population_id")
    external = serializers.BooleanField(default=False)
    resumeRunId = serializers.IntegerField(required=False, source="resume_run_id")


class RunStatusSerializer(serializers.ModelSerializer):
    totalPrompts = serializers.IntegerField(source="total_prompts", read_only=True)
    completedPrompts = serializers.IntegerField(source="completed_prompts", read_only=True)
    totalGenerations = serializers.IntegerField(source="total_generations", read_only=True)
    completedGenerations = serializers.IntegerField(source="completed_generations", read_only=True)
    failedGenerations = serializers.SerializerMethodField()
    progress = serializers.SerializerMethodField()
    errorMessage = serializers.CharField(source="error_message", read_only=True)
    heartbeatAt = serializers.DateTimeField(source="heartbeat_at", read_only=True)

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
        ]

    def get_progress(self, obj):
        if not obj.total_generations:
            return 0.0
        return round(obj.completed_generations / obj.total_generations, 4)

    def get_failedGenerations(self, obj):
        # Generations whose target LLM call failed after retries.
        return obj.responses.filter(error__isnull=False).count()
