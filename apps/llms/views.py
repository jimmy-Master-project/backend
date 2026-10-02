# pyright: reportAttributeAccessIssue=false
from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.evaluations.models import Evaluation

from .clients.factory import CLIENTS, KEYLESS_PROVIDERS, build_llm_client
from .clients.openai_compatible import list_models
from .models import LLMConfig
from .serializers import LLMConfigSerializer


def build_default_llm_config(evaluation, save=False):
    config = LLMConfig(
        evaluation=evaluation,
        provider=settings.TARGET_LLM_PROVIDER,
        model_name=settings.TARGET_LLM_MODEL_NAME,
        base_url=settings.TARGET_LLM_BASE_URL,
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        generations_per_prompt=1,
    )
    if save:
        config.save()
    return config


class LLMConfigView(APIView):
    def get(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        config = LLMConfig.objects.filter(evaluation=evaluation).first()
        if config is None:
            return Response(
                {
                    "provider": settings.TARGET_LLM_PROVIDER,
                    "modelName": settings.TARGET_LLM_MODEL_NAME,
                    "baseUrl": settings.TARGET_LLM_BASE_URL,
                    "temperature": 0.7,
                    "topP": 0.9,
                    "maxTokens": 512,
                    "generationsPerPrompt": 1,
                }
            )
        return Response(LLMConfigSerializer(config).data)

    def put(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        config = LLMConfig.objects.filter(evaluation=evaluation).first()

        serializer = LLMConfigSerializer(instance=config, data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(evaluation=evaluation)

        evaluation.status = Evaluation.Status.READY
        evaluation.save(update_fields=["status", "updated_at"])

        return Response(serializer.data)


class LLMConfigTestConnectionView(APIView):
    def post(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        config = LLMConfig.objects.filter(evaluation=evaluation).first()
        if config is None:
            config = build_default_llm_config(evaluation)

        if not config.model_name:
            return Response(
                {"success": False, "message": "TARGET_LLM_MODEL_NAME is not set."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            client = build_llm_client(config)
            client.generate("Reply briefly: connection test successful.")
        except Exception as exc:
            return Response(
                {"success": False, "message": str(exc)},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        return Response({"success": True, "message": "Connection successful."})


class LLMModelListView(APIView):
    """List models offered by an OpenAI-style endpoint.

    Always answers 200: an unreachable endpoint or an endpoint without
    ``/models`` yields an empty list plus ``error``, so the model name can
    still be typed by hand.
    """

    def post(self, request, evaluation_id):
        evaluation = get_object_or_404(Evaluation, pk=evaluation_id)
        provider = request.data.get("provider") or settings.TARGET_LLM_PROVIDER
        if provider not in CLIENTS:
            return Response(
                {"provider": f"Unsupported provider. Choose one of: {', '.join(CLIENTS)}."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        base_url = (request.data.get("baseUrl") or "").strip() or settings.TARGET_LLM_BASE_URL
        api_key = request.data.get("apiKey") or ""
        if not api_key:
            stored = LLMConfig.objects.filter(evaluation=evaluation).first()
            api_key = (stored.api_key if stored else "") or (
                "" if provider in KEYLESS_PROVIDERS else settings.TARGET_LLM_API_KEY
            )
        try:
            models = list_models(base_url=base_url, api_key=api_key)
        except Exception as exc:
            return Response({"models": [], "error": _short_error(exc)})
        return Response({"models": models, "error": None})


def _short_error(exc):
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return message[:200]
