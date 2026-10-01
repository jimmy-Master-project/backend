from rest_framework import serializers


class PersonaSampleRequestSerializer(serializers.Serializer):
    sampleSize = serializers.IntegerField(min_value=1, max_value=1000, source="sample_size")
    selectedAttributes = serializers.ListField(
        child=serializers.CharField(allow_blank=False),
        allow_empty=False,
        source="selected_attributes",
    )
    pool = serializers.CharField(required=False, allow_blank=True)
    # Keys are persona attribute names (data), not field names.
    filters = serializers.DictField(required=False, default=dict)
    stratify = serializers.JSONField(required=False, allow_null=True, default=None)
    seed = serializers.IntegerField(required=False, allow_null=True)
