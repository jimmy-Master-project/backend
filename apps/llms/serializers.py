from django.conf import settings
from rest_framework import serializers

from .models import LLMConfig


class LLMConfigSerializer(serializers.ModelSerializer):
    modelName = serializers.CharField(
        source="model_name", required=False, allow_blank=True
    )
    baseUrl = serializers.CharField(source="base_url", required=False, allow_blank=True)
    apiKey = serializers.CharField(
        source="api_key", required=False, allow_blank=True, write_only=True
    )
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
            "apiKey",
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
        if attrs.get("provider", getattr(self.instance, "provider", None)) == "custom" and not attrs["base_url"]:
            raise serializers.ValidationError(
                {"baseUrl": "This field is required for the custom provider."}
            )
        return attrs

    def update(self, instance, validated_data):
        # An omitted write-only key means "keep the existing credential".
        if "api_key" not in validated_data:
            validated_data["api_key"] = instance.api_key
        return super().update(instance, validated_data)
