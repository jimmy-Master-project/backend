from django.conf import settings

from .openai_compatible import OpenAICompatibleLLMClient
from .custom import CustomLLMClient

CLIENTS = {
    "openai_compatible": OpenAICompatibleLLMClient,
    "openai": OpenAICompatibleLLMClient,
    "custom": CustomLLMClient,
}


class UnsupportedProviderError(Exception):
    pass


def build_llm_client(llm_config):
    client_cls = CLIENTS.get(llm_config.provider)
    if client_cls is None:
        raise UnsupportedProviderError(f"Unsupported provider: {llm_config.provider}")

    return client_cls(
        model_name=llm_config.model_name,
        base_url=llm_config.base_url or None,
        api_key=llm_config.api_key or settings.TARGET_LLM_API_KEY,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        max_tokens=llm_config.max_tokens,
        language=getattr(llm_config.evaluation, "language", getattr(settings, "LANGFAIR_DEFAULT_LANGUAGE", "en")),
    )
