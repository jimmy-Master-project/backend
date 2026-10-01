from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings

from .clients.openai_compatible import OpenAICompatibleLLMClient
from .serializers import LLMConfigSerializer


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


class LLMConfigSerializerProviderTests(SimpleTestCase):
    def test_accepts_openai_providers(self):
        for provider in ("openai", "openai_compatible"):
            with self.subTest(provider=provider):
                serializer = LLMConfigSerializer(data={"provider": provider, "modelName": "gpt-4o-mini"})
                self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_rejects_other_providers(self):
        for provider in ("custom", "gemini"):
            with self.subTest(provider=provider):
                serializer = LLMConfigSerializer(data={"provider": provider, "modelName": "x"})
                self.assertFalse(serializer.is_valid())
                self.assertIn("provider", serializer.errors)
