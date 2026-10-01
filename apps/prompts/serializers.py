from django.conf import settings
from rest_framework import serializers

from .models import Prompt, PromptPopulation


class PromptSerializer(serializers.ModelSerializer):
    createdAt = serializers.DateTimeField(source="created_at", read_only=True)

    class Meta:
        model = Prompt
        # `metadata` is the raw internal (snake_case) JSON blob, exposed as-is.
        fields = ["id", "text", "category", "metadata", "sequence", "createdAt"]
        read_only_fields = ["id", "sequence"]


class PromptCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prompt
        fields = ["text", "category"]


class PromptPopulationSerializer(serializers.ModelSerializer):
    evaluationId = serializers.IntegerField(source="evaluation.id", read_only=True)
    modelName = serializers.CharField(source="model_name", read_only=True)
    requestedPromptCount = serializers.IntegerField(source="requested_prompt_count", read_only=True)
    actualPromptCount = serializers.IntegerField(source="actual_prompt_count", read_only=True)
    additionalInstruction = serializers.CharField(source="additional_instruction", read_only=True)
    errorMessage = serializers.CharField(source="error_message", read_only=True)
    createdAt = serializers.DateTimeField(source="created_at", read_only=True)

    class Meta:
        model = PromptPopulation
        # `metadata` is the raw internal (snake_case) JSON blob, exposed as-is.
        fields = [
            "id",
            "evaluationId",
            "name",
            "provider",
            "modelName",
            "temperature",
            "requestedPromptCount",
            "actualPromptCount",
            "additionalInstruction",
            "metadata",
            "status",
            "errorMessage",
            "createdAt",
        ]
        read_only_fields = ["id", "name", "provider", "temperature", "metadata", "status"]


# Upper bound per objective (x2 for counterfactual pairs); each prompt costs an LLM call.
MAX_PROMPT_COUNT = 500


class PersonaConfigSerializer(serializers.Serializer):
    """camelCase request body -> snake_case validated data (as stored in metadata)."""

    enabled = serializers.BooleanField(default=False)
    pool = serializers.CharField(required=False, allow_blank=True)
    selectedAttributes = serializers.ListField(
        child=serializers.CharField(), required=False, default=list, source="selected_attributes"
    )
    # Keys are persona attribute names (data), not field names.
    filters = serializers.DictField(required=False, default=dict)
    stratify = serializers.JSONField(required=False, allow_null=True, default=None)
    seed = serializers.IntegerField(required=False, allow_null=True, default=None)
    counterfactualEnabled = serializers.BooleanField(
        required=False, default=False, source="counterfactual_enabled"
    )


class GeneratePromptPopulationSerializer(serializers.Serializer):
    provider = serializers.CharField(required=False, allow_blank=True, default="")
    modelName = serializers.CharField(required=False, allow_blank=True, default="", source="model_name")
    baseUrl = serializers.CharField(required=False, allow_blank=True, default="", source="base_url")
    task = serializers.CharField(required=False, allow_blank=True, default="")
    promptCount = serializers.IntegerField(min_value=1, max_value=MAX_PROMPT_COUNT, source="prompt_count")
    temperature = serializers.FloatField(required=False, default=0.8)
    additionalInstruction = serializers.CharField(
        required=False, allow_blank=True, default="", source="additional_instruction"
    )
    persona = PersonaConfigSerializer(required=False)

    def validate(self, attrs):
        attrs["provider"] = attrs.get("provider") or settings.PROMPT_AGENT_PROVIDER
        attrs["model_name"] = attrs.get("model_name") or settings.PROMPT_AGENT_MODEL_NAME
        attrs["persona"] = dict(attrs.get("persona") or {})
        if not attrs["model_name"]:
            raise serializers.ValidationError(
                {"modelName": "This field is required unless PROMPT_AGENT_MODEL_NAME is set."}
            )
        return attrs
