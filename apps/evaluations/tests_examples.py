import json
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APITestCase

from .models import Evaluation
from .task_input import parse_task_file

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "task_files"

CLASSIFICATION_LABELS = [
    {"value": 0, "name": "reject", "description": "不核准"},
    {"value": 1, "name": "approve", "description": "核准"},
]
RECOMMENDATION_ITEMS = [
    {"name": "台幣定存", "description": "保本、固定利率，流動性較低"},
    {"name": "投資等級債券基金", "description": "風險中低，穩定配息"},
]


def _upload(name):
    return SimpleUploadedFile(name, (EXAMPLES_DIR / name).read_bytes())


class ExampleTaskFileTests(APITestCase):
    def test_every_example_file_parses(self):
        files = sorted(p.name for p in EXAMPLES_DIR.iterdir() if p.suffix in {".json", ".xlsx"})
        self.assertTrue(files)
        for name in files:
            with self.subTest(name=name):
                objectives = parse_task_file(_upload(name))
                self.assertTrue(objectives)
                self.assertTrue(all(item["objective"] for item in objectives))

    def test_xlsx_matches_json(self):
        self.assertEqual(
            parse_task_file(_upload("customer_service_text_generation.xlsx")),
            parse_task_file(_upload("customer_service_text_generation.json")),
        )

    def _create(self, name, task_type, **extra):
        # Mirror the frontend: multipart form, non-string values JSON-encoded.
        data = {"name": name, "taskType": task_type, "language": "auto", "taskFile": _upload(name)}
        data.update({key: json.dumps(value, ensure_ascii=False) for key, value in extra.items()})
        return self.client.post("/api/evaluations/", data, format="multipart")

    def test_create_text_generation_evaluation(self):
        response = self._create("customer_service_text_generation.xlsx", "text_generation")
        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(len(response.data["taskObjectives"]), 10)

    def test_create_classification_evaluation(self):
        response = self._create(
            "loan_review_classification.json",
            "classification",
            classificationLabels=CLASSIFICATION_LABELS,
        )
        self.assertEqual(response.status_code, 201, response.data)
        evaluation = Evaluation.objects.get(pk=response.data["id"])
        self.assertEqual(evaluation.classification_labels, CLASSIFICATION_LABELS)

    def test_create_recommendation_evaluation(self):
        response = self._create(
            "wealth_product_recommendation.json",
            "recommendation",
            recommendationItems=RECOMMENDATION_ITEMS,
        )
        self.assertEqual(response.status_code, 201, response.data)
        evaluation = Evaluation.objects.get(pk=response.data["id"])
        self.assertEqual(evaluation.recommendation_items, RECOMMENDATION_ITEMS)
