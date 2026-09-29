from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from .task_input import parse_task_file


class TaskInputTests(SimpleTestCase):
    def test_reads_json_objective_list(self):
        upload = SimpleUploadedFile(
            "tasks.json",
            b'[{"objective":"Ask about login", "task_type":"question"}, "Open an account"]',
        )
        self.assertEqual(
            parse_task_file(upload),
            [
                {"objective": "Ask about login"},
                {"objective": "Open an account"},
            ],
        )
