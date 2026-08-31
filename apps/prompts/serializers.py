from django.conf import settings
from rest_framework import serializers

from .models import Prompt, PromptPopulation


class PromptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prompt
        fields = ["id", "text", "category", "metadata", "sequence", "created_at"]
        read_only_fields = ["id", "sequence", "created_at"]


class PromptCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prompt
        fields = ["text", "category"]


class PromptPopulationSerializer(serializers.ModelSerializer):
    evaluation_id = serializers.IntegerField(source="evaluation.id", read_only=True)

    class Meta:
        model = PromptPopulation
        fields = [
            "id",
            "evaluation_id",
            "name",
            "provider",
            "model_name",
            "temperature",
            "requested_prompt_count",
            "actual_prompt_count",
            "additional_instruction",
            "metadata",
            "status",
            "error_message",
            "created_at",
        ]
        read_only_fields = fields


class GeneratePromptPopulationSerializer(serializers.Serializer):
    provider = serializers.CharField(required=False, allow_blank=True, default="")
    model = serializers.CharField(required=False, allow_blank=True, default="")
    model_name = serializers.CharField(required=False, allow_blank=True, default="")
    base_url = serializers.CharField(required=False, allow_blank=True, default="")
    task = serializers.CharField(required=False, allow_blank=True, default="")
    population_size = serializers.IntegerField(required=False, min_value=1)
    promptCount = serializers.IntegerField(required=False, min_value=1)
    prompt_count = serializers.IntegerField(required=False, min_value=1)
    temperature = serializers.FloatField(required=False, default=0.8)
    additionalInstructions = serializers.CharField(
        required=False, allow_blank=True, default=""
    )
    additional_instruction = serializers.CharField(
        required=False, allow_blank=True, default=""
    )
    persona = serializers.DictField(required=False, default=dict)

    def validate(self, attrs):
        attrs["provider"] = attrs.get("provider") or settings.PROMPT_AGENT_PROVIDER
        attrs["model_name"] = (
            attrs.get("model_name") or attrs.get("model") or settings.PROMPT_AGENT_MODEL_NAME
        )
        attrs["prompt_count"] = (
            attrs.get("prompt_count") or attrs.get("promptCount") or attrs.get("population_size")
        )
        attrs["additional_instruction"] = (
            attrs.get("additional_instruction") or attrs.get("additionalInstructions") or ""
        )
        if not attrs["model_name"]:
            raise serializers.ValidationError(
                {"model_name": "This field is required unless PROMPT_AGENT_MODEL_NAME is set."}
            )
        if not attrs["prompt_count"]:
            raise serializers.ValidationError({"prompt_count": "This field is required."})
        return attrs
