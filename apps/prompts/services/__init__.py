# ruff: noqa: F401
"""Prompt Population services.

Split into submodules; everything historically importable from
``apps.prompts.services`` is re-exported here so existing imports keep working.

Note for tests: patch collaborators where they are *used*, e.g.
``apps.prompts.services.generation.PromptGenerationService``.
"""
# pyright: reportMissingImports=false
import threading  # noqa: F401  (kept so ``apps.prompts.services.threading`` patch targets still resolve)

from apps.personas.services import PersonaService, PersonaServiceError  # noqa: F401
from apps.prompt_agents.services import (  # noqa: F401
    PromptGenerationService,
    PromptGenerationValidationError,
    UnsupportedProviderError,
)

from .counterfactual import (
    SUPPORTED_GENDER_COUNTERFACTUALS,
    build_gender_counterfactual_prompts,
    find_gender_attribute,
    format_counterfactual_gender,
    get_selected_gender_attribute,
    has_gender_attribute,
    normalize_gender,
    swap_gender_words,
)
from .generation import (
    PromptPopulationGenerationRunner,
    PromptPopulationGenerationService,
    reap_orphaned_populations,
)
from .instructions import (
    CLASSIFICATION_INSTRUCTION_MARKER,
    RECOMMENDATION_INSTRUCTION_MARKER,
    append_instruction,
    build_classification_output_instruction,
    build_fresh_generation_instruction,
    build_recommendation_output_instruction,
    strip_appended_instruction,
    with_classification_prompt_constraints,
    with_prompt_constraints,
    with_recommendation_prompt_constraints,
)
from .metadata import build_prompt_metadata
from .objectives import evaluation_for_objective, get_task_objectives

__all__ = [
    "CLASSIFICATION_INSTRUCTION_MARKER",
    "RECOMMENDATION_INSTRUCTION_MARKER",
    "SUPPORTED_GENDER_COUNTERFACTUALS",
    "PromptPopulationGenerationRunner",
    "PromptPopulationGenerationService",
    "append_instruction",
    "build_classification_output_instruction",
    "build_fresh_generation_instruction",
    "build_gender_counterfactual_prompts",
    "build_prompt_metadata",
    "build_recommendation_output_instruction",
    "evaluation_for_objective",
    "find_gender_attribute",
    "format_counterfactual_gender",
    "get_selected_gender_attribute",
    "get_task_objectives",
    "has_gender_attribute",
    "normalize_gender",
    "reap_orphaned_populations",
    "strip_appended_instruction",
    "swap_gender_words",
    "with_classification_prompt_constraints",
    "with_prompt_constraints",
    "with_recommendation_prompt_constraints",
]
