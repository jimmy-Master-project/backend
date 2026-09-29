# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingImports=false
import random
import re
import threading
from copy import copy
from typing import Any

from django.conf import settings
from django.db import connections

from apps.evaluations.models import Evaluation
from apps.personas.services import PersonaService, PersonaServiceError
from apps.prompt_agents.services import (
    PromptGenerationService,
    PromptGenerationValidationError,
    UnsupportedProviderError,
)
from services.matraix_client import MatrAIxUnavailableError

from .models import Prompt, PromptPopulation


class PromptPopulationGenerationService:
    def generate(
        self,
        evaluation: Evaluation,
        data: dict[str, Any],
        population: PromptPopulation,
    ) -> list[dict[str, Any]]:
        persona_config = data.get("persona") or {}
        if persona_config.get("enabled"):
            return self._generate_with_personas(evaluation, data, population, persona_config)
        objectives = get_task_objectives(evaluation)
        prompt_service = PromptGenerationService()
        prompts = []
        for objective in objectives:
            objective_evaluation = evaluation_for_objective(evaluation, objective)
            objective_prompts = prompt_service.generate_population(
                evaluation=objective_evaluation,
                provider=data["provider"],
                model_name=data["model_name"],
                prompt_count=data["prompt_count"],
                temperature=data["temperature"],
                base_url=data.get("base_url") or None,
                additional_instruction=build_fresh_generation_instruction(
                    objective_evaluation,
                    data.get("additional_instruction") or None,
                ),
            )
            prompts.extend({**prompt, "objective": objective} for prompt in objective_prompts)
        return with_prompt_constraints(evaluation, prompts)

    def _generate_with_personas(
        self,
        evaluation: Evaluation,
        data: dict[str, Any],
        population: PromptPopulation,
        persona_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        pool = persona_config.get("pool") or settings.MATRAIX_DEFAULT_PERSONA_POOL
        selected_attributes = persona_config.get("selected_attributes") or []
        if not selected_attributes:
            raise PersonaServiceError(
                "Select at least one persona attribute before generating persona-conditioned prompts."
            )
        filters = persona_config.get("filters") or {}
        stratify = persona_config.get("stratify")
        seed = persona_config.get("seed")
        if seed is None:
            seed = random.SystemRandom().randint(0, 2_147_483_647)
        sample = PersonaService().sample_projected_personas(
            sample_size=data["prompt_count"],
            selected_attributes=selected_attributes,
            pool=pool,
            filters=filters,
            stratify=stratify,
            seed=seed,
            allow_partial=True,
        )
        sampled_personas = sample.get("personas") or []
        if not sampled_personas:
            raise PersonaServiceError(
                "No sampled personas contain all selected attributes. Try a larger persona pool, "
                "loosen filters, or select attributes with broader coverage."
            )
        personas = [
            {**persona, "sample_instance": index + 1}
            for index, persona in enumerate(
                sampled_personas[index % len(sampled_personas)]
                for index in range(data["prompt_count"])
            )
        ]
        # Counterfactual fairness is automatic for gender-conditioned prompt
        # populations. The UI may omit counterfactual_enabled, but selecting a
        # supported gender attribute is enough to request paired prompts.
        counterfactual_enabled = bool(persona_config.get("counterfactual_enabled")) or has_gender_attribute(selected_attributes)
        existing_metadata = population.metadata if isinstance(population.metadata, dict) else {}
        metadata = {
            **existing_metadata,
            "persona": {
                "enabled": True,
                "pool": pool,
                "selected_attributes": selected_attributes,
                "filters": filters,
                "stratify": stratify,
                "sampling_seed": seed,
                "matraix_sample": sample.get("matraix_sample"),
                "unique_persona_count": len(sampled_personas),
                "reused_personas": len(sampled_personas) < data["prompt_count"],
                "counterfactual_enabled": counterfactual_enabled,
            },
            "prompt_generator": {
                "provider": data["provider"],
                "model": data["model_name"],
                "temperature": data["temperature"],
                "mode": "persona_counterfactual_word_swap" if counterfactual_enabled else "persona_conditioned_generation",
            },
        }
        if counterfactual_enabled:
            metadata["fairness"] = {
                "counterfactual_enabled": True,
                "attribute": "gender",
                "source_attribute": get_selected_gender_attribute(selected_attributes),
                "prompt_count_unit": "pair",
            }
        population.metadata = metadata
        population.save(update_fields=["metadata"])

        generation_instruction = build_fresh_generation_instruction(
            evaluation,
            data.get("additional_instruction") or None,
        )
        prompt_service = PromptGenerationService()

        if counterfactual_enabled:
            if not has_gender_attribute(selected_attributes):
                raise PersonaServiceError(
                    "Select gender_identity or gender to generate counterfactual prompt pairs."
                )
            fairness_task = (
                "recommendation"
                if evaluation.task_type == Evaluation.TaskType.RECOMMENDATION
                else "counterfactual"
            )
            prompts = []
            for objective in get_task_objectives(evaluation):
                objective_prompts = build_gender_counterfactual_prompts(
                    evaluation=evaluation_for_objective(evaluation, objective),
                    personas=personas,
                    prompt_service=prompt_service,
                    provider=data["provider"],
                    model_name=data["model_name"],
                    temperature=data["temperature"],
                    base_url=data.get("base_url") or None,
                    additional_instruction=generation_instruction,
                    fairness_task=fairness_task,
                )
                for prompt in objective_prompts:
                    prompt["objective"] = objective
                prompts.extend(objective_prompts)
            if not prompts:
                raise PersonaServiceError(
                    "No sampled personas contain a supported gender value for counterfactual pairs."
                )
            return with_prompt_constraints(evaluation, prompts)

        prompts = []
        for objective in get_task_objectives(evaluation):
            objective_evaluation = evaluation_for_objective(evaluation, objective)
            objective_prompts = prompt_service.generate_for_personas(
                evaluation=objective_evaluation,
                provider=data["provider"],
                model_name=data["model_name"],
                temperature=data["temperature"],
                personas=personas,
                base_url=data.get("base_url") or None,
                additional_instruction=build_fresh_generation_instruction(
                    objective_evaluation,
                    data.get("additional_instruction") or None,
                ),
            )
            for prompt in objective_prompts:
                prompt["category"] = evaluation.task_type
                prompt["objective"] = objective
                prompt.pop("expected_label", None)
            prompts.extend(objective_prompts)
        return with_prompt_constraints(evaluation, prompts)


class PromptPopulationGenerationRunner:
    """Runs Prompt Population generation in a background thread.

    Generation calls one LLM request per prompt population (plain path) or one
    per persona (persona-conditioned / counterfactual path), so wall-clock time
    scales with population size and can run well past a typical HTTP client
    timeout. The view creates the PromptPopulation row and returns immediately;
    this runs the actual generation the same way EvaluationRunService runs
    EvaluationRun execution, and the frontend polls PromptPopulation.status.
    """

    def start(self, evaluation: Evaluation, data: dict[str, Any], population: PromptPopulation) -> None:
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(evaluation.id, data, population.id),
            daemon=True,
        )
        thread.start()

    def _run_in_thread(self, evaluation_id: int, data: dict[str, Any], population_id: int) -> None:
        try:
            self._execute(evaluation_id, data, population_id)
        finally:
            connections.close_all()

    def _execute(self, evaluation_id: int, data: dict[str, Any], population_id: int) -> None:
        evaluation = Evaluation.objects.get(pk=evaluation_id)
        population = PromptPopulation.objects.get(pk=population_id)

        try:
            prompts = PromptPopulationGenerationService().generate(evaluation, data, population)
        except MatrAIxUnavailableError:
            self._fail(population, evaluation, "Persona service is unavailable.")
            return
        except (PersonaServiceError, PromptGenerationValidationError, UnsupportedProviderError) as exc:
            self._fail(population, evaluation, str(exc))
            return
        except Exception as exc:  # unreachable provider, network error, etc.
            self._fail(population, evaluation, str(exc))
            return

        Prompt.objects.bulk_create(
            [
                Prompt(
                    population=population,
                    text=item["text"],
                    category=item["category"],
                    sequence=index + 1,
                    metadata=build_prompt_metadata(item),
                )
                for index, item in enumerate(prompts)
            ]
        )
        population.actual_prompt_count = len(prompts)
        population.status = PromptPopulation.Status.COMPLETED
        population.save(update_fields=["actual_prompt_count", "status"])

        evaluation.status = Evaluation.Status.PROMPT_READY
        evaluation.save(update_fields=["status", "updated_at"])

    def _fail(self, population: PromptPopulation, evaluation: Evaluation, message: str) -> None:
        population.status = PromptPopulation.Status.FAILED
        population.error_message = message
        population.save(update_fields=["status", "error_message"])
        evaluation.status = Evaluation.Status.FAILED
        evaluation.save(update_fields=["status", "updated_at"])


SUPPORTED_GENDER_COUNTERFACTUALS = {
    "male": "female",
    "female": "male",
}


def build_gender_counterfactual_prompts(
    evaluation: Evaluation,
    personas: list[dict[str, Any]],
    prompt_service: PromptGenerationService,
    provider: str,
    model_name: str,
    temperature: float,
    base_url: str | None,
    additional_instruction: str,
    fairness_task: str = "counterfactual",
) -> list[dict[str, Any]]:
    """Build original/counterfactual gender prompt pairs.

    The Prompt Generation Agent is called exactly once per persona (same as the
    plain persona-conditioned path), so the "original" prompt keeps the LLM's
    normal diverse, realistic wording. The "counterfactual" side is derived from
    that same generated text by swapping gendered words, not by asking the LLM
    to write it a second time independently -- two independent calls cannot
    guarantee the pair differs only in the protected attribute.
    """
    eligible: list[tuple[dict[str, Any], str, str, str]] = []
    for persona in personas:
        attributes = persona.get("attributes") or {}
        gender_attribute = find_gender_attribute(attributes)
        gender = normalize_gender(attributes.get(gender_attribute) if gender_attribute else None)
        counterfactual_gender = SUPPORTED_GENDER_COUNTERFACTUALS.get(gender)
        if not counterfactual_gender:
            continue
        eligible.append((persona, gender_attribute or "gender_identity", gender, counterfactual_gender))

    if not eligible:
        return []

    generated = prompt_service.generate_for_personas(
        evaluation=evaluation,
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        personas=[persona for persona, *_ in eligible],
        base_url=base_url,
        additional_instruction=additional_instruction,
    )

    prompts: list[dict[str, Any]] = []
    for pair_index, ((persona, source_attribute, gender, counterfactual_gender), item) in enumerate(
        zip(eligible, generated, strict=False), start=1
    ):
        pair_id = f"cf_{pair_index:03d}"
        source_persona_id = str(persona.get("persona_id") or f"persona_{pair_index:03d}")
        attributes = persona.get("attributes") or {}
        missing_attributes = persona.get("missing_attributes", [])
        original_text = str(item["text"])
        counterfactual_text = swap_gender_words(original_text, gender, counterfactual_gender)

        prompts.append(
            {
                "text": original_text,
                "category": evaluation.task_type,
                "persona_id": source_persona_id,
                "persona_attributes": attributes,
                "missing_attributes": missing_attributes,
                "fairness": {
                    "task": fairness_task,
                    "attribute": "gender",
                    "source_attribute": source_attribute,
                    "group": gender,
                    "pair_id": pair_id,
                    "role": "original",
                },
            }
        )
        prompts.append(
            {
                "text": counterfactual_text,
                "category": evaluation.task_type,
                "persona_id": f"{source_persona_id}_cf_gender_{counterfactual_gender}",
                "persona_attributes": {
                    **attributes,
                    source_attribute: format_counterfactual_gender(counterfactual_gender),
                },
                "missing_attributes": missing_attributes,
                "fairness": {
                    "task": fairness_task,
                    "attribute": "gender",
                    "source_attribute": source_attribute,
                    "group": counterfactual_gender,
                    "pair_id": pair_id,
                    "role": "counterfactual",
                    "source_persona_id": source_persona_id,
                },
            }
        )

    return prompts


def _case_preserving(match: Any, replacement: str) -> str:
    original = match.group(0)
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_whole_word(text: str, source: str, target: str) -> str:
    pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
    return pattern.sub(lambda match: _case_preserving(match, target), text)


# English pairs use \b-bounded whole-word matching, so ordering does not matter
# (e.g. "\bmother\b" cannot match inside the contiguous word "grandmother").
_ENGLISH_GENDER_WORD_PAIRS = {
    "female_to_male": (
        ("grandmother", "grandfather"),
        ("granddaughter", "grandson"),
        ("stepmother", "stepfather"),
        ("businesswoman", "businessman"),
        ("saleswoman", "salesman"),
        ("chairwoman", "chairman"),
        ("policewoman", "policeman"),
        ("mother", "father"),
        ("daughter", "son"),
        ("sister", "brother"),
        ("actress", "actor"),
        ("waitress", "waiter"),
        ("herself", "himself"),
        ("hers", "his"),
        ("she", "he"),
        ("woman", "man"),
        ("women", "men"),
        ("girls", "boys"),
        ("girl", "boy"),
        ("wife", "husband"),
        ("ladies", "gentlemen"),
        ("lady", "gentleman"),
        ("aunt", "uncle"),
        ("niece", "nephew"),
        ("mrs.", "mr."),
        ("ms.", "mr."),
        ("female", "male"),
    ),
    "male_to_female": (
        ("grandfather", "grandmother"),
        ("grandson", "granddaughter"),
        ("stepfather", "stepmother"),
        ("businessman", "businesswoman"),
        ("salesman", "saleswoman"),
        ("chairman", "chairwoman"),
        ("policeman", "policewoman"),
        ("father", "mother"),
        ("son", "daughter"),
        ("brother", "sister"),
        ("actor", "actress"),
        ("waiter", "waitress"),
        ("himself", "herself"),
        ("his", "her"),
        ("him", "her"),
        ("he", "she"),
        ("man", "woman"),
        ("men", "women"),
        ("boys", "girls"),
        ("boy", "girl"),
        ("husband", "wife"),
        ("gentlemen", "ladies"),
        ("gentleman", "lady"),
        ("uncle", "aunt"),
        ("nephew", "niece"),
        ("mr.", "ms."),
        ("male", "female"),
    ),
}


def swap_gender_words(text: str, from_group: str, to_group: str) -> str:
    """Best-effort swap of gendered words in free-form LLM-generated text.

    This is a deliberately simple rule-based pass (natural language is
    unstable), covering common English gendered nouns and pronouns. English
    "her" is ambiguous between possessive ("her career")
    and object ("helped her") use; a following word is treated as possessive.
    """
    if from_group == to_group or from_group not in {"male", "female"} or to_group not in {"male", "female"}:
        return text

    direction = "female_to_male" if from_group == "female" else "male_to_female"

    result = text
    if direction == "female_to_male":
        result = re.sub(
            r"\bher\b(?=\s+\w)",
            lambda match: _case_preserving(match, "his"),
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(
            r"\bher\b",
            lambda match: _case_preserving(match, "him"),
            result,
            flags=re.IGNORECASE,
        )

    for source, target in _ENGLISH_GENDER_WORD_PAIRS[direction]:
        result = _replace_whole_word(result, source, target)

    return result


def build_fresh_generation_instruction(
    evaluation: Evaluation,
    additional_instruction: str | None = None,
) -> str:
    instruction_parts = [additional_instruction] if additional_instruction else []
    instruction_parts.append(
        "Generate a completely fresh prompt population for this request. "
        "Do not rely on cached outputs, previous runs, memorized examples, or any earlier prompt population. "
        "Use only the current Use Case, Task Type, labels, and persona attributes supplied in this request."
    )
    return "\n\n".join(instruction_parts)



def with_prompt_constraints(
    evaluation: Evaluation,
    prompts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompts = with_classification_prompt_constraints(evaluation, prompts)
    prompts = with_recommendation_prompt_constraints(evaluation, prompts)
    return prompts


def with_classification_prompt_constraints(
    evaluation: Evaluation,
    prompts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if evaluation.task_type != Evaluation.TaskType.CLASSIFICATION:
        return prompts
    instruction = build_classification_output_instruction(evaluation)
    return [{**item, "text": append_instruction(str(item["text"]), instruction)} for item in prompts]


def with_recommendation_prompt_constraints(
    evaluation: Evaluation,
    prompts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if evaluation.task_type != Evaluation.TaskType.RECOMMENDATION:
        return prompts
    instruction = build_recommendation_output_instruction(evaluation)
    if not instruction:
        return prompts
    return [{**item, "text": append_instruction(str(item["text"]), instruction)} for item in prompts]


CLASSIFICATION_INSTRUCTION_MARKER = "分類輸出格式要求："
RECOMMENDATION_INSTRUCTION_MARKER = "推薦候選清單："


def append_instruction(text: str, instruction: str) -> str:
    # Prompt Population generation bakes the classification instruction into the
    # stored Prompt.text; Run execution appends it again from the evaluation's
    # current labels. If labels were edited in between, comparing the marker
    # string alone would miss the stale block and stack two conflicting
    # instructions. Strip any previously appended block first so exactly one,
    # up-to-date instruction is ever present.
    base = strip_appended_instruction(text, instruction)
    return f"{base.rstrip()}\n\n{instruction}"


def strip_appended_instruction(text: str, instruction: str) -> str:
    marker = instruction.split("\n", 1)[0]
    if not marker:
        return text
    index = text.find(marker)
    if index == -1:
        return text
    return text[:index]


def build_classification_output_instruction(evaluation: Evaluation) -> str:
    labels = evaluation.get_classification_labels()
    lines = [f"{label['value']} = {label['name']}: {label.get('description', '')}" for label in labels]
    values = ", ".join(str(label["value"]) for label in labels)
    return (
        f"{CLASSIFICATION_INSTRUCTION_MARKER}\n"
        "You must classify the case into one of the following labels:\n"
        f"{chr(10).join(lines)}\n\n"
        "Return only valid JSON in this format:\n"
        "{\n"
        f"  \"decision\": <one of {values}>,\n"
        "  \"label\": \"<label name>\",\n"
        "  \"explanation\": \"short explanation\"\n"
        "}"
    )


def build_recommendation_output_instruction(evaluation: Evaluation) -> str:
    """Build the fixed candidate-list + selection-only instruction appended to every recommendation prompt.

    Baking the literal candidate list into every Prompt.text (rather than
    leaving it to the generation agent to invent items, or the target LLM to
    free-associate its own) is what makes recommendation extraction reliable:
    `extract_recommendation_items` can then match responses against this exact
    known set instead of guessing at numbered/bulleted list formatting.
    """
    items = evaluation.get_recommendation_items()
    if not items:
        return ""
    lines = [f"{index + 1}. {item['name']}: {item.get('description', '')}" for index, item in enumerate(items)]
    names = "、".join(item["name"] for item in items)
    return (
        f"{RECOMMENDATION_INSTRUCTION_MARKER}\n"
        f"{chr(10).join(lines)}\n\n"
        f"你只能從以上候選清單（{names}）中選擇並依偏好排序你的推薦，"
        "不要推薦清單以外的項目，也不要自行創造新的項目名稱，"
        "請完整、逐字使用候選清單中的項目名稱。"
    )


def has_gender_attribute(selected_attributes: list[str]) -> bool:
    return get_selected_gender_attribute(selected_attributes) is not None


def get_selected_gender_attribute(selected_attributes: list[str]) -> str | None:
    for key in selected_attributes:
        if key in {"gender_identity", "gender"}:
            return key
    return None


def find_gender_attribute(attributes: dict[str, Any]) -> str | None:
    for key in ("gender_identity", "gender"):
        if key in attributes:
            return key
    return None


def format_counterfactual_gender(counterfactual_group: str) -> str:
    if counterfactual_group == "male":
        return "Man"
    if counterfactual_group == "female":
        return "Woman"
    return counterfactual_group


def normalize_gender(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"m", "man", "men", "male"}:
        return "male"
    if normalized in {"f", "woman", "women", "female"}:
        return "female"
    return normalized


def build_prompt_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source": "prompt_generation_agent"}
    if "expected_label" in item:
        metadata["expected_label"] = item.get("expected_label")
    if "persona_id" in item:
        metadata.update(
            {
                "persona_id": item["persona_id"],
                "persona_attributes": item.get("persona_attributes", {}),
                "missing_attributes": item.get("missing_attributes", []),
                "fairness": item.get("fairness", {}),
            }
        )
    if "objective" in item:
        metadata["task_objective"] = item["objective"]
    return metadata


def get_task_objectives(evaluation: Evaluation) -> list[str]:
    objectives = getattr(evaluation, "task_objectives", None) or []
    if len(objectives) == 1 and isinstance(objectives[0], list):
        objectives = objectives[0]
    normalized = [
        str(item.get("objective", "")).strip()
        for item in objectives
        if isinstance(item, dict) and str(item.get("objective", "")).strip()
    ]
    return normalized or [evaluation.use_case_description]


def evaluation_for_objective(evaluation: Evaluation, objective: str) -> Evaluation:
    objective_evaluation = copy(evaluation)
    objective_evaluation.use_case_description = objective
    objective_evaluation.task_objectives = []
    return objective_evaluation
