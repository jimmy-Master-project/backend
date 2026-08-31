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
