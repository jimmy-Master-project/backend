import io
import json
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from openpyxl import Workbook
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


AIRLINE_EXAMPLE = "airline_customer_service_text_generation.json"


def _upload(name):
    return SimpleUploadedFile(name, (EXAMPLES_DIR / name).read_bytes())


def _json_upload(name, objectives):
    return SimpleUploadedFile(name, json.dumps(objectives, ensure_ascii=False).encode("utf-8"))


def _xlsx_upload(name, objectives):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["objective"])
    for item in objectives:
        sheet.append([item["objective"]])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return SimpleUploadedFile(name, buffer.getvalue())


class ExampleTaskFileTests(APITestCase):
    def test_every_example_file_parses(self):
        files = sorted(p.name for p in EXAMPLES_DIR.iterdir() if p.suffix in {".json", ".xlsx"})
        self.assertIn(AIRLINE_EXAMPLE, files)
        for name in files:
            with self.subTest(name=name):
                objectives = parse_task_file(_upload(name))
                self.assertTrue(objectives)
                self.assertTrue(all(item["objective"] for item in objectives))

    def test_xlsx_matches_json(self):
        objectives = parse_task_file(_upload(AIRLINE_EXAMPLE))
        self.assertEqual(parse_task_file(_xlsx_upload("airline.xlsx", objectives)), objectives)

    def _create(self, upload, task_type, **extra):
        # Mirror the frontend: multipart form, non-string values JSON-encoded.
        data = {"name": upload.name, "useCaseDescription": "Target LLM under test.",
                "taskType": task_type, "language": "auto", "taskFile": upload}
        data.update({key: json.dumps(value, ensure_ascii=False) for key, value in extra.items()})
        return self.client.post("/api/evaluations/", data, format="multipart")

    def test_create_text_generation_evaluation(self):
        response = self._create(_upload(AIRLINE_EXAMPLE), "text_generation")
        self.assertEqual(response.status_code, 201, response.data)
        # The uploaded file supplies the test directions; objectives come later from the planner.
        self.assertEqual(len(response.data["testDirections"]), 10)
        self.assertEqual(response.data["taskObjectives"], [])

    def test_uploaded_test_directions_are_appended_to_typed_ones(self):
        response = self._create(
            _json_upload("seeds.json", ["Test refund requests."]), "text_generation",
            testDirections=["Test lost baggage."],
        )
        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["testDirections"], ["Test lost baggage.", "Test refund requests."])

    def test_upload_still_requires_target_description(self):
        data = {"name": "x", "taskType": "text_generation", "taskFile": _json_upload("s.json", ["Test refunds."])}
        response = self.client.post("/api/evaluations/", data, format="multipart")
        self.assertEqual(response.status_code, 400)
        self.assertIn("useCaseDescription", response.data)

    def test_create_classification_evaluation(self):
        response = self._create(
            _json_upload("loan_review_classification.json", ["Review a small-business loan application."]),
            "classification",
            classificationLabels=CLASSIFICATION_LABELS,
        )
        self.assertEqual(response.status_code, 201, response.data)
        evaluation = Evaluation.objects.get(pk=response.data["id"])
        self.assertEqual(evaluation.classification_labels, CLASSIFICATION_LABELS)

    def test_create_recommendation_evaluation(self):
        response = self._create(
            _json_upload("wealth_product_recommendation.json", ["Recommend a savings product."]),
            "recommendation",
            recommendationItems=RECOMMENDATION_ITEMS,
        )
        self.assertEqual(response.status_code, 201, response.data)
        evaluation = Evaluation.objects.get(pk=response.data["id"])
        self.assertEqual(evaluation.recommendation_items, RECOMMENDATION_ITEMS)
