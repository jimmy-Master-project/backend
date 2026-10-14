from rest_framework import serializers

from apps.prompts.serializers import validate_protected_attributes


class PersonaListQuerySerializer(serializers.Serializer):
    offset = serializers.IntegerField(min_value=0, default=0)
    limit = serializers.IntegerField(min_value=1, max_value=200, default=50)
    pool = serializers.CharField(required=False, allow_blank=True)
    # Comma-separated persona ids: fetch these cards instead of a page.
    ids = serializers.CharField(required=False, allow_blank=True)

    def validate_ids(self, value):
        ids = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
        if len(ids) > 500:
            raise serializers.ValidationError("At most 500 persona ids per request.")
        return ids


class PersonaSampleRequestSerializer(serializers.Serializer):
    sampleSize = serializers.IntegerField(min_value=1, max_value=1000, source="sample_size")
    selectedAttributes = serializers.ListField(
        child=serializers.CharField(allow_blank=False),
        allow_empty=False,
        source="selected_attributes",
        validators=[validate_protected_attributes],
    )
    pool = serializers.CharField(required=False, allow_blank=True)
    # Keys are persona attribute names (data), not field names.
    filters = serializers.DictField(required=False, default=dict)
    stratify = serializers.JSONField(required=False, allow_null=True, default=None)
    seed = serializers.IntegerField(required=False, allow_null=True)
