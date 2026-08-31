from django.conf import settings
from rest_framework import serializers

from .models import LLMConfig


class LLMConfigSerializer(serializers.ModelSerializer):
    modelName = serializers.CharField(
        source="model_name", required=False, allow_blank=True
    )
    baseUrl = serializers.CharField(source="base_url", required=False, allow_blank=True)
    topP = serializers.FloatField(source="top_p", required=False)
    maxTokens = serializers.IntegerField(source="max_tokens", required=False, min_value=1)
    generationsPerPrompt = serializers.IntegerField(
        source="generations_per_prompt", required=False, min_value=1
    )
    createdAt = serializers.DateTimeField(source="created_at", read_only=True)
    updatedAt = serializers.DateTimeField(source="updated_at", read_only=True)

    class Meta:
        model = LLMConfig
        fields = [
            "id",
            "provider",
            "modelName",
            "baseUrl",
            "temperature",
            "topP",
            "maxTokens",
            "generationsPerPrompt",
            "createdAt",
            "updatedAt",
        ]
        read_only_fields = ["id", "createdAt", "updatedAt"]

    def validate(self, attrs):
        attrs["model_name"] = attrs.get("model_name") or settings.TARGET_LLM_MODEL_NAME
        attrs["base_url"] = attrs.get("base_url") or settings.TARGET_LLM_BASE_URL
        if not attrs["model_name"]:
            raise serializers.ValidationError(
                {"modelName": "This field is required unless TARGET_LLM_MODEL_NAME is set."}
            )
        return attrs
