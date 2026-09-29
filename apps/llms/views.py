# pyright: reportAttributeAccessIssue=false
from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.evaluations.models import Evaluation

from .clients.factory import build_llm_client
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
