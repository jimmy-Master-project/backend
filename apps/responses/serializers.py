from rest_framework import serializers

from .models import LLMResponse


class ResponsePromptSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    text = serializers.CharField()
    category = serializers.CharField()


class LLMResponseSerializer(serializers.ModelSerializer):
    prompt = ResponsePromptSerializer()
    class Meta:
        model = LLMResponse
        fields = [
            "id",
            "prompt",
            "generation_index",
            "response_text",
            "input_tokens",
            "output_tokens",
            "latency_ms",
            "created_at",
        ]


class ExternalResponseItemSerializer(serializers.Serializer):
    promptId = serializers.IntegerField(min_value=1)
    response = serializers.CharField(allow_blank=False)
    generationIndex = serializers.IntegerField(default=1, min_value=1)
    inputTokens = serializers.IntegerField(required=False, allow_null=True, min_value=0)
    outputTokens = serializers.IntegerField(required=False, allow_null=True, min_value=0)
    latencyMs = serializers.IntegerField(required=False, allow_null=True, min_value=0)


class ExternalResponsesSerializer(serializers.Serializer):
    responses = ExternalResponseItemSerializer(many=True, allow_empty=False)
    calculate = serializers.BooleanField(default=False)
    suites = serializers.ListField(required=False, child=serializers.CharField())
    options = serializers.JSONField(required=False)
