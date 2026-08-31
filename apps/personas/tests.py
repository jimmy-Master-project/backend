# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportOptionalSubscript=false
from unittest.mock import Mock, patch

from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.test import APIClient

from apps.evaluations.models import Evaluation
from apps.prompts.models import Prompt, PromptPopulation
from services.matraix_client import MatrAIxUnavailableError

from .services import PersonaService, PersonaServiceError


class PersonaServiceTests(TestCase):
    def test_sample_projected_personas_keeps_every_dimension_not_just_selected(self):
        # selected_attributes only names the diagnostic attribute(s) to gate
        # persona inclusion on; it must not discard the other dimensions the
        # persona pool returns, since prompt generation needs that fuller
        # context to avoid over-fitting on the diagnostic attribute alone.
        client = Mock()
        client.sample_personas.return_value = {"persona_ids": ["0042"]}
        client.get_persona.return_value = {
            "persona_id": "0042",
            "dimensions": {
                "age_bracket": "25-34",
                "gender": "Female",
                "education": "Bachelor",
                "income_level": "Medium",
            },
        }

        result = PersonaService(client).sample_projected_personas(
            sample_size=1,
            selected_attributes=["age_bracket", "gender"],
            pool="persona/datasets/matraix-persona-dev-sample",
        )

        self.assertEqual(result["personas"][0]["persona_id"], "0042")
        self.assertEqual(
            result["personas"][0]["attributes"],
            {
                "age_bracket": "25-34",
                "gender": "Female",
                "education": "Bachelor",
                "income_level": "Medium",
            },
        )
        self.assertEqual(result["personas"][0]["missing_attributes"], [])

    def test_sample_projected_personas_requires_selected_attributes(self):
        client = Mock()
        client.sample_personas.return_value = {"persona_ids": ["0042", "0093"]}
        client.get_persona.side_effect = [
            {"persona_id": "0042", "dimensions": {"age_bracket": "25-34"}},
            {"persona_id": "0093", "dimensions": {"gender": "Man"}},
        ]

        result = PersonaService(client).sample_projected_personas(
            sample_size=1,
            selected_attributes=["gender"],
            pool="persona/datasets/matraix-persona-dev-sample",
        )

        self.assertEqual(result["personas"][0]["persona_id"], "0093")
        self.assertEqual(result["personas"][0]["attributes"], {"gender": "Man"})

    def test_sample_projected_personas_rejects_short_samples(self):
        client = Mock()
        client.sample_personas.return_value = {"persona_ids": ["0042"]}

        with self.assertRaises(PersonaServiceError):
            PersonaService(client).sample_projected_personas(
                sample_size=2,
                selected_attributes=["gender"],
                pool="persona/datasets/matraix-persona-dev-sample",
            )


class PersonaApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    @patch("apps.personas.views.PersonaService")
    def test_sample_api_returns_projected_personas(self, service_cls):
        service_cls.return_value.sample_projected_personas.return_value = {
            "pool": "pool-a",
            "selected_attributes": ["gender"],
            "personas": [
                {
                    "persona_id": "0042",
                    "attributes": {"gender": "Female"},
                    "missing_attributes": [],
                }
            ],
        }

        response = self.client.post(
            "/api/persona/sample",
            {
                "sample_size": 1,
                "selected_attributes": ["gender"],
                "filters": {},
                "stratify": None,
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["personas"][0]["attributes"], {"gender": "Female"})

    @patch("apps.personas.views.PersonaService")
    def test_attributes_api_handles_matraix_unavailable(self, service_cls):
        service_cls.return_value.get_attributes.side_effect = MatrAIxUnavailableError()

        response = self.client.get("/api/persona/attributes")

        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data["error"], "matraix_unavailable")


class _ImmediateThread:
    """Drop-in threading.Thread replacement that runs the target synchronously.

    Generation runs in a real background thread in production (see
    PromptPopulationGenerationRunner) so the HTTP response returns before
    generation finishes. Under Django's transactional TestCase, a second
    thread/connection racing the still-open test transaction hits sqlite
    "database is locked" errors and can't see uncommitted rows anyway, so
    these tests replace the thread with one that runs inline -- the endpoint's
    behavior is otherwise unchanged (it just returns 202 with the population
    already generated by the time the client call returns).
    """

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


@patch("apps.prompts.services.threading.Thread", new=_ImmediateThread)
class PersonaPromptPopulationTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.evaluation = Evaluation.objects.create(
            name="Career advice",
            use_case_description="Ask an AI for career advice",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
        )

    @override_settings(PROMPT_AGENT_PROVIDER="openai_compatible", PROMPT_AGENT_MODEL_NAME="agent-model")
    @patch("apps.prompts.services.PromptGenerationService")
    @patch("apps.prompts.services.PersonaService")
    def test_generate_population_with_personas_stores_reproducibility_metadata(
        self,
        persona_service_cls,
        prompt_service_cls,
    ):
        # NOTE: gender_identity/gender triggers automatic counterfactual pairing
        # (see test_generate_population_with_gender_builds_template_counterfactual_pairs
        # below), so this test uses a non-gender attribute to exercise the plain
        # persona-conditioned generation path instead.
        persona_service_cls.return_value.sample_projected_personas.return_value = {
            "pool": "persona/datasets/matraix-persona-dev-sample",
            "selected_attributes": ["age_bracket", "risk_tolerance"],
            "filters": {},
            "stratify": None,
            "seed": 123,
            "matraix_sample": {"persona_ids": ["0042"]},
            "personas": [
                {
                    "persona_id": "0042",
                    "attributes": {"age_bracket": "25-34", "risk_tolerance": "medium"},
                    "missing_attributes": [],
                }
            ],
        }
        prompt_service_cls.return_value.generate_for_personas.return_value = [
            {
                "text": "How should I plan my next career move?",
                "category": "persona",
                "persona_id": "0042",
                "persona_attributes": {"age_bracket": "25-34", "risk_tolerance": "medium"},
                "missing_attributes": [],
            }
        ]

        response = self.client.post(
            f"/api/evaluations/{self.evaluation.id}/prompt-populations/generate",
            {
                "population_size": 1,
                "persona": {
                    "enabled": True,
                    "pool": "persona/datasets/matraix-persona-dev-sample",
                    "selected_attributes": ["age_bracket", "risk_tolerance"],
                    "filters": {},
                    "stratify": None,
                    "seed": 123,
                },
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        population = PromptPopulation.objects.get(pk=response.data["population_id"])
        prompt = Prompt.objects.get(population=population)
        self.assertEqual(population.metadata["persona"]["sampling_seed"], 123)
        self.assertEqual(population.metadata["prompt_generator"]["mode"], "persona_conditioned_generation")
        self.assertNotIn("fairness", population.metadata)
        self.assertEqual(prompt.metadata["persona_id"], "0042")
        self.assertEqual(prompt.metadata["persona_attributes"]["risk_tolerance"], "medium")

    @override_settings(PROMPT_AGENT_PROVIDER="openai_compatible", PROMPT_AGENT_MODEL_NAME="agent-model")
    @patch("apps.prompts.services.PromptGenerationService")
    @patch("apps.prompts.services.PersonaService")
    def test_generate_population_reuses_partial_persona_sample(
        self,
        persona_service_cls,
        prompt_service_cls,
    ):
        # Uses a non-gender attribute so this exercises persona sample reuse/cycling
        # rather than the automatic gender counterfactual pairing path.
        persona_service_cls.return_value.sample_projected_personas.return_value = {
            "pool": "persona/datasets/matraix-persona-dev-sample",
            "selected_attributes": ["age_bracket"],
            "filters": {},
            "stratify": None,
            "seed": 123,
            "matraix_sample": {"persona_ids": ["0042"]},
            "personas": [
                {
                    "persona_id": "0042",
                    "attributes": {"age_bracket": "25-34"},
                    "missing_attributes": [],
                }
            ],
        }
        prompt_service_cls.return_value.generate_for_personas.return_value = [
            {
                "text": f"Prompt {index}",
                "category": "persona",
                "persona_id": "0042",
                "persona_attributes": {"age_bracket": "25-34"},
                "missing_attributes": [],
            }
            for index in range(3)
        ]

        response = self.client.post(
            f"/api/evaluations/{self.evaluation.id}/prompt-populations/generate",
            {
                "population_size": 3,
                "persona": {
                    "enabled": True,
                    "selected_attributes": ["age_bracket"],
                    "seed": 123,
                },
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        population = PromptPopulation.objects.get(pk=response.data["population_id"])
        self.assertEqual(population.actual_prompt_count, 3)
        self.assertEqual(population.metadata["persona"]["unique_persona_count"], 1)
        self.assertTrue(population.metadata["persona"]["reused_personas"])
        persona_service_cls.return_value.sample_projected_personas.assert_called_once()
        sample_kwargs = persona_service_cls.return_value.sample_projected_personas.call_args.kwargs
        self.assertTrue(sample_kwargs["allow_partial"])
        generate_kwargs = prompt_service_cls.return_value.generate_for_personas.call_args.kwargs
        self.assertEqual(len(generate_kwargs["personas"]), 3)
        self.assertEqual(generate_kwargs["personas"][2]["sample_instance"], 3)

    @override_settings(PROMPT_AGENT_PROVIDER="openai_compatible", PROMPT_AGENT_MODEL_NAME="agent-model")
    @patch("apps.prompts.services.PromptGenerationService")
    @patch("apps.prompts.services.PersonaService")
    def test_persona_generation_does_not_store_expected_labels(
        self,
        persona_service_cls,
        prompt_service_cls,
    ):
        self.evaluation.task_type = Evaluation.TaskType.CLASSIFICATION
        self.evaluation.classification_labels = [
            {"value": 0, "name": "Reject", "description": "should reject"},
            {"value": 1, "name": "Approve", "description": "should approve"},
        ]
        self.evaluation.save(update_fields=["task_type", "classification_labels"])
        persona_service_cls.return_value.sample_projected_personas.return_value = {
            "pool": "persona/datasets/matraix-persona-dev-sample",
            "selected_attributes": ["age_bracket"],
            "filters": {},
            "stratify": None,
            "seed": 123,
            "matraix_sample": {"persona_ids": ["0042"]},
            "personas": [
                {
                    "persona_id": "0042",
                    "attributes": {"age_bracket": "25-34"},
                    "missing_attributes": [],
                }
            ],
        }
        prompt_service_cls.return_value.generate_for_personas.return_value = [
            {
                "text": "Should this applicant be approved?",
                "category": "persona",
                "expected_label": 1,
                "persona_id": "0042",
                "persona_attributes": {"age_bracket": "25-34"},
                "missing_attributes": [],
            }
        ]

        response = self.client.post(
            f"/api/evaluations/{self.evaluation.id}/prompt-populations/generate",
            {
                "population_size": 1,
                "persona": {
                    "enabled": True,
                    "selected_attributes": ["age_bracket"],
                    "seed": 123,
                },
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        prompt = Prompt.objects.get(population_id=response.data["population_id"])
        self.assertNotIn("expected_label", prompt.metadata)
        prompt_service_cls.return_value.generate_for_personas.assert_called_once()
        kwargs = prompt_service_cls.return_value.generate_for_personas.call_args.kwargs
        self.assertNotIn("desired_labels", kwargs)

    @override_settings(PROMPT_AGENT_PROVIDER="openai_compatible", PROMPT_AGENT_MODEL_NAME="agent-model")
    @patch("apps.prompts.services.PromptGenerationService")
    @patch("apps.prompts.services.PersonaService")
    def test_generate_population_with_gender_builds_word_swap_counterfactual_pairs(
        self,
        persona_service_cls,
        prompt_service_cls,
    ):
        # The counterfactual side must be derived from the LLM's own generated
        # text via a gender-word swap (one LLM call per persona), not from a
        # fixed template and not from a second independent LLM call.
        persona_service_cls.return_value.sample_projected_personas.return_value = {
            "pool": "persona/datasets/matraix-persona-dev-sample",
            "selected_attributes": ["gender_identity", "age_bracket"],
            "filters": {},
            "stratify": None,
            "seed": 123,
            "matraix_sample": {"persona_ids": ["0042"]},
            "personas": [
                {
                    "persona_id": "0042",
                    "attributes": {"gender_identity": "Woman", "age_bracket": "25-34"},
                    "missing_attributes": [],
                }
            ],
        }
        prompt_service_cls.return_value.generate_for_personas.return_value = [
            {
                "text": "身為一名女性工程師，我想請教長期投資建議。",
                "category": "persona",
                "persona_id": "0042",
                "persona_attributes": {"gender_identity": "Woman", "age_bracket": "25-34"},
                "missing_attributes": [],
            }
        ]

        response = self.client.post(
            f"/api/evaluations/{self.evaluation.id}/prompt-populations/generate",
            {
                "population_size": 1,
                "persona": {
                    "enabled": True,
                    "selected_attributes": ["gender_identity", "age_bracket"],
                    "seed": 123,
                },
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        population = PromptPopulation.objects.get(pk=response.data["population_id"])
        self.assertEqual(population.actual_prompt_count, 2)
        self.assertEqual(
            population.metadata["prompt_generator"]["mode"], "persona_counterfactual_word_swap"
        )

        # Exactly one LLM call for the one eligible persona, not one per pair side.
        prompt_service_cls.return_value.generate_for_personas.assert_called_once()
        generate_kwargs = prompt_service_cls.return_value.generate_for_personas.call_args.kwargs
        self.assertEqual(len(generate_kwargs["personas"]), 1)

        prompts = list(Prompt.objects.filter(population=population).order_by("sequence"))
        self.assertEqual(len(prompts), 2)
        original, counterfactual = prompts
        self.assertEqual(original.metadata["fairness"]["role"], "original")
        self.assertEqual(original.metadata["fairness"]["group"], "female")
        self.assertEqual(counterfactual.metadata["fairness"]["role"], "counterfactual")
        self.assertEqual(counterfactual.metadata["fairness"]["group"], "male")
        self.assertEqual(
            original.metadata["fairness"]["pair_id"],
            counterfactual.metadata["fairness"]["pair_id"],
        )

        # The original prompt is exactly what the LLM generated; the
        # counterfactual prompt is that same text with gender words swapped.
        self.assertEqual(original.text, "身為一名女性工程師，我想請教長期投資建議。")
        self.assertEqual(counterfactual.text, "身為一名男性工程師，我想請教長期投資建議。")
