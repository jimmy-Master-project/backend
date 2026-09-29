from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings

from .clients.custom import CustomLLMClient


class CustomLLMClientTests(SimpleTestCase):
    @override_settings(TARGET_LLM_TIMEOUT_SECONDS=7)
    @patch("apps.llms.clients.custom.httpx.post")
    def test_posts_stable_payload_and_reads_response(self, post):
        response = Mock()
        response.json.return_value = {"response": "hello"}
        post.return_value = response

        result = CustomLLMClient(
            model_name="robot-v1",
            base_url="https://robot.example/generate",
            api_key="secret",
        ).generate("say hello")

        post.assert_called_once()
        request = post.call_args.kwargs
        self.assertEqual(request["json"]["prompt"], "say hello")
        self.assertEqual(request["json"]["model"], "robot-v1")
        self.assertEqual(request["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(request["timeout"], 7)
        self.assertEqual(result["text"], "hello")
