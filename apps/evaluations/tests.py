# pyright: reportMissingImports=false
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from .models import Evaluation


class RecommendationItemsSerializerTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _payload(self, recommendation_items=None, task_type="recommendation"):
        payload = {
            "name": "Product recs",
            "useCaseDescription": "Recommend the best products for a fitness enthusiast.",
            "taskType": task_type,
        }
        if recommendation_items is not None:
            payload["recommendationItems"] = recommendation_items
        return payload

    def test_creates_recommendation_evaluation_with_candidate_items(self):
        response = self.client.post(
            "/api/evaluations",
            self._payload(
                [
                    {"name": "高蛋白燕麥飲", "description": "低糖、高蛋白"},
                    {"name": "瑜伽墊", "description": "適合居家運動"},
                ]
            ),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        evaluation = Evaluation.objects.get(pk=response.data["id"])
        self.assertEqual(
            evaluation.get_recommendation_items(),
            [
                {"name": "高蛋白燕麥飲", "description": "低糖、高蛋白"},
                {"name": "瑜伽墊", "description": "適合居家運動"},
            ],
        )

    def test_rejects_fewer_than_two_candidate_items(self):
        response = self.client.post(
            "/api/evaluations",
            self._payload([{"name": "高蛋白燕麥飲", "description": ""}]),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("recommendationItems", response.data)

    def test_rejects_duplicate_candidate_item_names(self):
        response = self.client.post(
            "/api/evaluations",
            self._payload(
                [
                    {"name": "高蛋白燕麥飲", "description": "A"},
                    {"name": "高蛋白燕麥飲", "description": "B"},
                ]
            ),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("recommendationItems", response.data)

    def test_rejects_candidate_items_on_non_recommendation_tasks(self):
        response = self.client.post(
            "/api/evaluations",
            self._payload(
                [
                    {"name": "A", "description": ""},
                    {"name": "B", "description": ""},
                ],
                task_type="text_generation",
            ),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("recommendationItems", response.data)

    def test_recommendation_task_without_items_is_allowed(self):
        response = self.client.post(
            "/api/evaluations",
            self._payload(None),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        evaluation = Evaluation.objects.get(pk=response.data["id"])
        self.assertEqual(evaluation.get_recommendation_items(), [])


class GetRecommendationItemsModelMethodTests(TestCase):
    def test_returns_empty_for_non_recommendation_task_type_even_if_items_set(self):
        evaluation = Evaluation(
            name="x",
            use_case_description="x",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
            recommendation_items=[{"name": "A", "description": ""}],
        )

        self.assertEqual(evaluation.get_recommendation_items(), [])
