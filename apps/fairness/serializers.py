from rest_framework import serializers


SUPPORTED_SUITES = [
    "toxicity",
    "stereotype",
    "classification",
    "counterfactual",
    "recommendation",
]


class LangFairRunMetricsSerializer(serializers.Serializer):
    suites = serializers.ListField(
        child=serializers.ChoiceField(choices=SUPPORTED_SUITES),
        required=False,
        allow_empty=False,
    )
    options = serializers.JSONField(required=False)

    def validate_options(self, value):
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise serializers.ValidationError("options must be an object.")
        return value
