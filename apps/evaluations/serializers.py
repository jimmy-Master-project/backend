from rest_framework import serializers

from .models import Evaluation


class ClassificationLabelSerializer(serializers.Serializer):
    value = serializers.IntegerField()
    name = serializers.CharField(max_length=64)
    description = serializers.CharField(required=False, allow_blank=True)


class RecommendationItemSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=128)
    description = serializers.CharField(required=False, allow_blank=True)


class EvaluationSerializer(serializers.ModelSerializer):
    useCaseDescription = serializers.CharField(source="use_case_description")
    taskType = serializers.ChoiceField(source="task_type", choices=Evaluation.TaskType.choices)
    language = serializers.ChoiceField(choices=Evaluation.Language.choices, required=False)
    classificationLabels = ClassificationLabelSerializer(
        source="classification_labels",
        many=True,
        required=False,
    )
    recommendationItems = RecommendationItemSerializer(
        source="recommendation_items",
        many=True,
        required=False,
    )
    promptCount = serializers.SerializerMethodField()
    createdAt = serializers.DateTimeField(source="created_at", read_only=True)
    updatedAt = serializers.DateTimeField(source="updated_at", read_only=True)

    class Meta:
        model = Evaluation
        fields = [
            "id",
            "name",
            "useCaseDescription",
            "taskType",
            "language",
            "classificationLabels",
            "recommendationItems",
            "status",
            "promptCount",
            "createdAt",
            "updatedAt",
        ]
        read_only_fields = ["id", "status", "createdAt", "updatedAt"]

    def validate(self, attrs):
        attrs = super().validate(attrs)
        task_type = attrs.get("task_type") or getattr(self.instance, "task_type", None)
        labels = attrs.get("classification_labels")
        if task_type == Evaluation.TaskType.CLASSIFICATION:
            if labels is None and self.instance is None:
                attrs["classification_labels"] = [
                    {"value": 0, "name": "reject", "description": "不通過"},
                    {"value": 1, "name": "approve", "description": "通過"},
                ]
                labels = attrs["classification_labels"]
            if labels is not None:
                self._validate_classification_labels(labels)
        elif labels:
            raise serializers.ValidationError(
                {"classificationLabels": "Only classification tasks can define labels."}
            )

        items = attrs.get("recommendation_items")
        if task_type == Evaluation.TaskType.RECOMMENDATION:
            if items is not None:
                self._validate_recommendation_items(items)
        elif items:
            raise serializers.ValidationError(
                {"recommendationItems": "Only recommendation tasks can define candidate items."}
            )
        return attrs

    def _validate_classification_labels(self, labels):
        if len(labels) != 2:
            raise serializers.ValidationError(
                {"classificationLabels": "Classification tasks currently support exactly two labels."}
            )
        values = [label["value"] for label in labels]
        names = [label["name"].strip() for label in labels]
        if len(values) != len(set(values)):
            raise serializers.ValidationError(
                {"classificationLabels": "Label values must be unique."}
            )
        if len(names) != len(set(names)):
            raise serializers.ValidationError(
                {"classificationLabels": "Label names must be unique."}
            )

    def _validate_recommendation_items(self, items):
        if len(items) < 2:
            raise serializers.ValidationError(
                {"recommendationItems": "Define at least two candidate items to recommend from."}
            )
        names = [item["name"].strip() for item in items]
        if any(not name for name in names):
            raise serializers.ValidationError(
                {"recommendationItems": "Every candidate item needs a name."}
            )
        if len(names) != len(set(names)):
            raise serializers.ValidationError(
                {"recommendationItems": "Candidate item names must be unique."}
            )

    def get_promptCount(self, obj):
        return sum(
            population.actual_prompt_count
            for population in obj.prompt_populations.all()
        )
