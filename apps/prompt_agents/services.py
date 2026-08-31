from .builders import build_persona_prompt_agent_messages, build_prompt_agent_messages
from .clients.openai_compatible import OpenAICompatiblePromptAgentClient

CLIENTS = {
    "openai_compatible": OpenAICompatiblePromptAgentClient,
    "openai": OpenAICompatiblePromptAgentClient,
}


class PromptGenerationValidationError(Exception):
    pass


class UnsupportedProviderError(Exception):
    pass


class PromptGenerationService:
    def _build_client(self, provider, model_name, base_url, temperature):
        client_cls = CLIENTS.get(provider)
        if client_cls is None:
            raise UnsupportedProviderError(f"Unsupported provider: {provider}")
        return client_cls(model_name=model_name, base_url=base_url, temperature=temperature)

    def _validate(self, payload, prompt_count):
        if not isinstance(payload, dict) or "prompts" not in payload:
            raise PromptGenerationValidationError("Response is missing 'prompts' field.")

        prompts = payload["prompts"]
        if not isinstance(prompts, list):
            raise PromptGenerationValidationError("'prompts' must be an array.")

        seen_texts = set()
        cleaned = []
        for item in prompts:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)
            cleaned_item = {
                "text": text,
                "category": (item.get("category") or "").strip(),
            }
            if "expected_label" in item:
                cleaned_item["expected_label"] = item.get("expected_label")
            elif "expectedLabel" in item:
                cleaned_item["expected_label"] = item.get("expectedLabel")
            cleaned.append(cleaned_item)

        if not cleaned:
            raise PromptGenerationValidationError(
                "Prompt agent did not return any valid prompts."
            )

        return cleaned[:prompt_count]

    def generate_population(
        self,
        evaluation,
        provider,
        model_name,
        prompt_count,
        temperature,
        base_url=None,
        additional_instruction=None,
    ):
        client = self._build_client(provider, model_name, base_url, temperature)
        messages = build_prompt_agent_messages(
            evaluation, prompt_count, additional_instruction
        )
        payload = client.generate(messages)
        return self._validate(payload, prompt_count)

    def generate_for_personas(
        self,
        evaluation,
        provider,
        model_name,
        temperature,
        personas,
        base_url=None,
        additional_instruction=None,
        desired_labels=None,
    ):
        client = self._build_client(provider, model_name, base_url, temperature)
        generated = []
        for index, persona in enumerate(personas):
            desired_label = None
            if desired_labels:
                desired_label = desired_labels[index % len(desired_labels)]
            messages = build_persona_prompt_agent_messages(
                evaluation,
                persona["attributes"],
                additional_instruction,
                desired_label=desired_label,
            )
            payload = client.generate(messages)
            prompt = self._validate(payload, 1)[0]
            if desired_label is not None:
                prompt["expected_label"] = desired_label.get("value")
            generated.append(
                {
                    **prompt,
                    "persona_id": persona["persona_id"],
                    "persona_attributes": persona["attributes"],
                    "missing_attributes": persona.get("missing_attributes", []),
                }
            )
        return generated
