from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings

import httpx

from .clients.custom import CustomLLMClient
from .clients.openai_compatible import OpenAICompatibleLLMClient


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

    @override_settings(TARGET_LLM_MAX_RETRIES=2)
    @patch("apps.llms.clients.custom.time.sleep")
    @patch("apps.llms.clients.custom.httpx.post")
    def test_retries_transient_errors_with_backoff(self, post, sleep):
        request = httpx.Request("POST", "https://robot.example/generate")
        busy = httpx.Response(503, request=request)
        ok = Mock()
        ok.json.return_value = {"text": "hello"}
        post.side_effect = [httpx.ReadTimeout("slow", request=request), busy, ok]

        result = CustomLLMClient(
            model_name="robot-v1", base_url="https://robot.example/generate"
        ).generate("say hello")

        self.assertEqual(result["text"], "hello")
        self.assertEqual(post.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])

    @override_settings(TARGET_LLM_MAX_RETRIES=2)
    @patch("apps.llms.clients.custom.time.sleep")
    @patch("apps.llms.clients.custom.httpx.post")
    def test_does_not_retry_client_errors(self, post, sleep):
        request = httpx.Request("POST", "https://robot.example/generate")
        post.return_value = httpx.Response(400, request=request)

        with self.assertRaises(httpx.HTTPStatusError):
            CustomLLMClient(
                model_name="robot-v1", base_url="https://robot.example/generate"
            ).generate("say hello")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()


class OpenAICompatibleLLMClientTests(SimpleTestCase):
    @override_settings(TARGET_LLM_TIMEOUT_SECONDS=9, TARGET_LLM_MAX_RETRIES=4)
    def test_sends_only_the_prompt(self):
        client = OpenAICompatibleLLMClient(model_name="target", base_url="http://x/v1")
        self.assertEqual(client.client.timeout, 9)
        self.assertEqual(client.client.max_retries, 4)
        completion = Mock()
        completion.choices = [Mock()]
        completion.choices[0].message.content = "answer"
        with patch.object(
            client.client.chat.completions, "create", return_value=completion
        ) as create:
            result = client.generate("the prompt")

        self.assertEqual(result["text"], "answer")
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "the prompt"}])
        self.assertNotIn("Cache-Control", kwargs["extra_headers"])
        self.assertNotIn("Pragma", kwargs["extra_headers"])
