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
    useCaseDescription = serializers.CharField(
        source="use_case_description", required=False, allow_blank=True
    )
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
    testDirections = serializers.ListField(
        child=serializers.CharField(allow_blank=True), source="test_directions", required=False
    )
    taskObjectives = serializers.ListField(source="task_objectives", required=False)
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
            "testDirections",
            "taskObjectives",
            "status",
            "promptCount",
            "createdAt",
            "updatedAt",
        ]
        read_only_fields = ["id", "status", "createdAt", "updatedAt"]

    def validate(self, attrs):
        attrs = super().validate(attrs)
        # The initial prompt agent needs both user-defined inputs: the target
        # LLM description and at least one test direction. They are required
        # when creating an evaluation; a partial update only checks the fields
        # it sends, so e.g. PATCH {"taskObjectives": []} still works on
        # evaluations created before test directions existed.
        if "use_case_description" in attrs or self.instance is None:
            attrs["use_case_description"] = (attrs.get("use_case_description") or "").strip()
            if not attrs["use_case_description"]:
                raise serializers.ValidationError(
                    {"useCaseDescription": "Describe the target LLM (its purpose, service scope and policies)."}
                )
        if "test_directions" in attrs or self.instance is None:
            directions = [
                " ".join(str(item).split()) for item in attrs.get("test_directions") or []
            ]
            attrs["test_directions"] = list(dict.fromkeys(item for item in directions if item))
            if not attrs["test_directions"]:
                raise serializers.ValidationError(
                    {"testDirections": "Define at least one test direction."}
                )
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


class TestPlanTargetInfoSerializer(serializers.Serializer):
    knownPolicies = serializers.CharField(required=False, allow_blank=True, default="", source="known_policies")
    systemPromptExcerpt = serializers.CharField(
        required=False, allow_blank=True, default="", source="system_prompt_excerpt"
    )
    userPopulation = serializers.CharField(required=False, allow_blank=True, default="", source="user_population")
    riskPriorities = serializers.CharField(required=False, allow_blank=True, default="", source="risk_priorities")
    whenStages = serializers.ListField(
        child=serializers.CharField(), required=False, default=list, source="when_stages"
    )


class GenerateTestPlanSerializer(serializers.Serializer):
    provider = serializers.CharField(required=False, allow_blank=True, default="")
    modelName = serializers.CharField(required=False, allow_blank=True, default="", source="model_name")
    baseUrl = serializers.CharField(required=False, allow_blank=True, default="", source="base_url")
    temperature = serializers.FloatField(required=False, default=0.7)
    targetInfo = TestPlanTargetInfoSerializer(required=False, source="target_info")
    protectedAttributes = serializers.ListField(
        child=serializers.ChoiceField(choices=["gender", "race_ethnicity"]),
        required=False,
        default=list,
        source="protected_attributes",
    )
    totalDirections = serializers.IntegerField(
        required=False, default=12, min_value=1, max_value=50, source="total_directions"
    )
    apply = serializers.BooleanField(required=False, default=True)

    def validate(self, attrs):
        from django.conf import settings

        attrs["provider"] = attrs.get("provider") or settings.PROMPT_AGENT_PROVIDER
        attrs["model_name"] = attrs.get("model_name") or settings.PROMPT_AGENT_MODEL_NAME
        if not attrs["model_name"]:
            raise serializers.ValidationError(
                {"modelName": "This field is required unless PROMPT_AGENT_MODEL_NAME is set."}
            )
        return attrs
