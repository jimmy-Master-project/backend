from rest_framework import serializers


class PersonaSampleRequestSerializer(serializers.Serializer):
    sample_size = serializers.IntegerField(min_value=1, max_value=1000)
    selected_attributes = serializers.ListField(
        child=serializers.CharField(allow_blank=False),
        allow_empty=False,
    )
    pool = serializers.CharField(required=False, allow_blank=True)
    filters = serializers.DictField(required=False, default=dict)
    stratify = serializers.JSONField(required=False, allow_null=True, default=None)
    seed = serializers.IntegerField(required=False, allow_null=True)
