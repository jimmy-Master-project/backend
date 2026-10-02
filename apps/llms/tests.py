from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings
from rest_framework.test import APITestCase

from apps.evaluations.models import Evaluation

from .clients.openai_compatible import OpenAICompatibleLLMClient
from .models import LLMConfig
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
        for provider in ("openai", "openai_compatible", "evaair", "virgin", "priceline"):
            with self.subTest(provider=provider):
                serializer = LLMConfigSerializer(data={"provider": provider, "modelName": "gpt-4o-mini"})
                self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_rejects_other_providers(self):
        for provider in ("custom", "gemini"):
            with self.subTest(provider=provider):
                serializer = LLMConfigSerializer(data={"provider": provider, "modelName": "x"})
                self.assertFalse(serializer.is_valid())
                self.assertIn("provider", serializer.errors)


class LLMModelListViewTests(APITestCase):
    def setUp(self):
        self.evaluation = Evaluation.objects.create(
            name="e", use_case_description="d", task_type=Evaluation.TaskType.TEXT_GENERATION
        )
        self.url = f"/api/evaluations/{self.evaluation.id}/llm-config/models/"

    @patch("apps.llms.clients.openai_compatible.OpenAI")
    def test_lists_chat_models_only(self, openai_cls):
        openai_cls.return_value.models.list.return_value = [
            Mock(id=model_id)
            for model_id in ("gpt-4o-mini", "text-embedding-3-small", "gpt-4o", "whisper-1", "local-llama")
        ]
        response = self.client.post(self.url, {"provider": "openai", "baseUrl": "https://api.openai.com/v1"}, format="json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, {"models": ["gpt-4o", "gpt-4o-mini", "local-llama"], "error": None})
        self.assertEqual(openai_cls.call_args.kwargs["max_retries"], 0)

    @patch("apps.llms.clients.openai_compatible.OpenAI")
    def test_unreachable_endpoint_is_not_an_error(self, openai_cls):
        openai_cls.return_value.models.list.side_effect = ConnectionError("Connection refused")
        response = self.client.post(self.url, {"provider": "openai_compatible", "baseUrl": "http://127.0.0.1:9/v1"}, format="json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["models"], [])
        self.assertIn("Connection refused", response.data["error"])

    @patch("apps.llms.clients.openai_compatible.OpenAI")
    def test_falls_back_to_stored_api_key(self, openai_cls):
        openai_cls.return_value.models.list.return_value = []
        LLMConfig.objects.create(
            evaluation=self.evaluation, provider="openai", model_name="gpt-4o-mini", api_key="sk-stored"
        )
        self.client.post(self.url, {"provider": "openai", "baseUrl": "https://api.openai.com/v1"}, format="json")
        self.assertEqual(openai_cls.call_args.kwargs["api_key"], "sk-stored")

    def test_rejects_unknown_provider(self):
        response = self.client.post(self.url, {"provider": "custom"}, format="json")
        self.assertEqual(response.status_code, 400)


class EvaAirProviderTests(APITestCase):
    @override_settings(TARGET_LLM_API_KEY="sk-openai")
    @patch("apps.llms.clients.openai_compatible.OpenAI")
    def test_evaair_never_receives_the_openai_key(self, openai_cls):
        from .clients.factory import build_llm_client

        evaluation = Evaluation.objects.create(
            name="e", use_case_description="d", task_type=Evaluation.TaskType.TEXT_GENERATION
        )
        config = LLMConfig.objects.create(
            evaluation=evaluation, provider="evaair", model_name="evaair", base_url="http://127.0.0.1:8768/v1"
        )
        build_llm_client(config)
        self.assertEqual(openai_cls.call_args.kwargs["api_key"], "not-needed")
        self.assertEqual(openai_cls.call_args.kwargs["base_url"], "http://127.0.0.1:8768/v1")

        openai_cls.return_value.models.list.return_value = [Mock(id="evaair")]
        response = self.client.post(
            f"/api/evaluations/{evaluation.id}/llm-config/models/",
            {"provider": "evaair", "baseUrl": "http://127.0.0.1:8768/v1"},
            format="json",
        )
        self.assertEqual(response.data["models"], ["evaair"])
        self.assertEqual(openai_cls.call_args.kwargs["api_key"], "not-needed")

    @override_settings(TARGET_LLM_API_KEY="sk-openai")
    @patch("apps.llms.clients.openai_compatible.OpenAI")
    def test_virgin_never_receives_the_openai_key(self, openai_cls):
        from .clients.factory import build_llm_client

        evaluation = Evaluation.objects.create(
            name="v", use_case_description="d", task_type=Evaluation.TaskType.TEXT_GENERATION
        )
        config = LLMConfig.objects.create(
            evaluation=evaluation, provider="virgin", model_name="virgin", base_url="http://127.0.0.1:8769/virgin/v1"
        )
        build_llm_client(config)
        self.assertEqual(openai_cls.call_args.kwargs["api_key"], "not-needed")
        self.assertEqual(openai_cls.call_args.kwargs["base_url"], "http://127.0.0.1:8769/virgin/v1")
