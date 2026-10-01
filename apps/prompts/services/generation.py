# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingImports=false
import random
import threading
from datetime import timedelta
from typing import Any

from django.conf import settings
from django.db import connections
from django.utils import timezone

from apps.evaluations.models import Evaluation
from apps.personas.services import PersonaService, PersonaServiceError
from apps.prompt_agents.services import (
    PromptGenerationService,
    PromptGenerationValidationError,
    UnsupportedProviderError,
)
from services.matraix_client import MatrAIxUnavailableError

from ..models import Prompt, PromptPopulation
from .counterfactual import (
    build_gender_counterfactual_prompts,
    get_selected_gender_attribute,
    has_gender_attribute,
)
from .instructions import build_fresh_generation_instruction, with_prompt_constraints
from .metadata import build_prompt_metadata
from .objectives import evaluation_for_objective, get_task_objectives


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
            swap_stats: dict[str, Any] = {"identical_swap_skipped": 0}
            for objective_index, objective in enumerate(get_task_objectives(evaluation), start=1):
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
                    objective_index=objective_index,
                    stats=swap_stats,
                )
                for prompt in objective_prompts:
                    prompt["objective"] = objective
                prompts.extend(objective_prompts)
            skipped = int(swap_stats["identical_swap_skipped"])
            metadata["fairness"]["identical_swap_skipped"] = skipped
            population.metadata = metadata
            population.save(update_fields=["metadata"])
            if not prompts:
                if skipped:
                    raise PersonaServiceError(
                        "Gender word swap did not change any generated prompt, so no counterfactual "
                        f"pairs could be built ({skipped} identical pair(s) skipped). Ask the prompt "
                        "agent to mention the persona's gender explicitly."
                    )
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


IN_PROGRESS_STATUSES = (PromptPopulation.Status.PENDING, PromptPopulation.Status.GENERATING)
INTERRUPTED_MESSAGE = "Prompt generation was interrupted (server restarted). Generate again."
# Populations younger than this are never reaped, so a just-created row whose
# thread has not registered yet is not mistaken for an orphan.
ORPHAN_GRACE_SECONDS = 60

_live_populations: set[int] = set()
_live_populations_lock = threading.Lock()


def reap_orphaned_populations(evaluation_id: int | None = None) -> int:
    """Mark in-progress populations with no live worker thread in this process as FAILED.

    Same single-process assumption as ``apps.runs.services.reap_orphaned_runs``.
    Without this a server restart would leave a population "generating" forever
    and a client resuming the poll after a reload would never stop.
    """
    queryset = PromptPopulation.objects.filter(
        status__in=IN_PROGRESS_STATUSES,
        created_at__lt=timezone.now() - timedelta(seconds=ORPHAN_GRACE_SECONDS),
    )
    if evaluation_id is not None:
        queryset = queryset.filter(evaluation_id=evaluation_id)
    with _live_populations_lock:
        live = set(_live_populations)
    orphans = queryset.exclude(id__in=live)
    evaluation_ids = list(orphans.values_list("evaluation_id", flat=True).distinct())
    reaped = orphans.update(status=PromptPopulation.Status.FAILED, error_message=INTERRUPTED_MESSAGE)
    if reaped:
        Evaluation.objects.filter(
            id__in=evaluation_ids, status=Evaluation.Status.GENERATING_PROMPTS
        ).update(status=Evaluation.Status.FAILED, updated_at=timezone.now())
    return reaped


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
        with _live_populations_lock:
            _live_populations.add(population.id)
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
            with _live_populations_lock:
                _live_populations.discard(population_id)
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

