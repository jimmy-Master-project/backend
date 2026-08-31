from rest_framework import serializers

from .models import EvaluationRun


class StartRunSerializer(serializers.Serializer):
    prompt_population_id = serializers.IntegerField(required=False)


class RunStatusSerializer(serializers.ModelSerializer):
    progress = serializers.SerializerMethodField()

    class Meta:
        model = EvaluationRun
        fields = [
            "id",
            "status",
            "total_prompts",
            "completed_prompts",
            "total_generations",
            "completed_generations",
            "progress",
            "error_message",
        ]

    def get_progress(self, obj):
        if not obj.total_generations:
            return 0.0
        return round(obj.completed_generations / obj.total_generations, 4)
