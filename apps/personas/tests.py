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


class PersonaCardLookupTests(TestCase):
    def test_list_by_ids_builds_cards_from_single_lookups(self):
        client = Mock()
        client.get_persona.side_effect = lambda persona_id, pool: (
            {
                "personaId": persona_id,
                "name": "Noah Petrov",
                "source": "wiki",
                "dimensions": {"age_bracket": "65+", "gender_identity": "Woman", "fam_history": "Familiar"},
            }
            if persona_id == "p1"
            else (_ for _ in ()).throw(MatrAIxUnavailableError())
        )

        result = PersonaService(client=client).list_personas(limit=50, persona_ids=["p1", "gone"])

        client.list_personas.assert_not_called()
        self.assertEqual(len(result["personas"]), 1)
        card = result["personas"][0]
        self.assertEqual(card["name"], "Noah Petrov")
        self.assertEqual(card["summary"], {"age_bracket": "65+"})
        self.assertIn("gender_identity Woman", card["search_text"])


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
                "sampleSize": 1,
                "selectedAttributes": ["gender"],
                "filters": {},
                "stratify": None,
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["personas"][0]["attributes"], {"gender": "Female"})
        self.assertEqual(response.data["personas"][0]["personaId"], "0042")
        self.assertEqual(response.data["personas"][0]["missingAttributes"], [])
        self.assertEqual(response.data["selectedAttributes"], ["gender"])

    @patch("apps.personas.views.PersonaService")
    def test_attributes_api_handles_matraix_unavailable(self, service_cls):
        service_cls.return_value.get_attributes.side_effect = MatrAIxUnavailableError()

        response = self.client.get("/api/persona/attributes")

        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data["error"], "matraix_unavailable")


    @patch("apps.personas.views.PersonaService")
    def test_list_api_pages_persona_cards(self, service_cls):
        service_cls.return_value.list_personas.return_value = {
            "pool": "pool-a",
            "offset": 50,
            "limit": 50,
            "total": 10000,
            "personas": [
                {
                    "persona_id": "wiki-1",
                    "name": "Clara Rossi",
                    "source": "wiki",
                    "summary": {"region": "Western Europe"},
                    "search_text": "wiki-1 Clara Rossi gender_identity Woman",
                }
            ],
        }

        response = self.client.get("/api/persona/personas", {"offset": 50})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        service_cls.return_value.list_personas.assert_called_once()
        self.assertEqual(service_cls.return_value.list_personas.call_args.kwargs["limit"], 50)
        self.assertEqual(service_cls.return_value.list_personas.call_args.kwargs["offset"], 50)
        self.assertEqual(response.data["total"], 10000)
        self.assertEqual(response.data["personas"][0]["personaId"], "wiki-1")
        self.assertEqual(response.data["personas"][0]["summary"], {"region": "Western Europe"})

    @patch("apps.personas.views.PersonaService")
    def test_list_api_fetches_specific_ids(self, service_cls):
        service_cls.return_value.list_personas.return_value = {
            "pool": "pool-a", "offset": 0, "limit": 2, "total": 10000, "personas": [],
        }

        response = self.client.get("/api/persona/personas", {"ids": "b, a,b"})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(service_cls.return_value.list_personas.call_args.kwargs["persona_ids"], ["b", "a"])

    def test_list_api_rejects_oversized_pages(self):
        response = self.client.get("/api/persona/personas", {"limit": 1000})

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class ProtectedAttributeTests(TestCase):
    """Fairness labels are limited to what LangFair supports (gender, race/ethnicity)."""

    def test_persona_values_map_to_langfair_groups(self):
        from .protected_attributes import langfair_attribute, langfair_group

        self.assertEqual(langfair_attribute("gender"), "gender")
        self.assertEqual(langfair_attribute("demo_ethnicity_broad"), "race")
        self.assertIsNone(langfair_attribute("age_bracket"))
        self.assertEqual(langfair_group("gender_identity", "Woman"), "female")
        self.assertIsNone(langfair_group("gender_identity", "Non-binary"))
        self.assertEqual(langfair_group("demo_ethnicity_broad", "South Asian"), "asian")
        self.assertIsNone(langfair_group("demo_ethnicity_broad", "Middle Eastern"))

    def test_non_langfair_attribute_is_rejected(self):
        client = APIClient()
        evaluation = Evaluation.objects.create(name="x", use_case_description="y")

        response = client.post(
            f"/api/evaluations/{evaluation.id}/prompt-population/generate",
            {"promptCount": 1, "modelName": "m", "persona": {"enabled": True, "selectedAttributes": ["age_bracket"]}},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("age_bracket", str(response.data))

    @patch("apps.prompts.services.generation.PromptGenerationService")
    @patch("apps.prompts.services.generation.PersonaService")
    def test_any_attribute_can_filter_and_stratify_the_sample(self, persona_service_cls, prompt_service_cls):
        persona_service_cls.return_value.sample_projected_personas.return_value = {
            "personas": [{"persona_id": "p1", "attributes": {"gender_identity": "Woman"}, "missing_attributes": []}],
        }
        prompt_service_cls.return_value.generate_for_personas.return_value = []
        evaluation = Evaluation.objects.create(name="x", use_case_description="y")

        with patch("apps.prompts.services.threading.Thread", new=_ImmediateThread):
            response = APIClient().post(
                f"/api/evaluations/{evaluation.id}/prompt-population/generate",
                {
                    "promptCount": 1,
                    "modelName": "m",
                    "persona": {
                        "enabled": True,
                        # Fairness label: LangFair only. Sampling filters: any attribute.
                        "selectedAttributes": ["demo_ethnicity_broad"],
                        "filters": {"age_bracket": ["25-34"], "region": ["Western Europe"]},
                        "stratify": "age_bracket",
                    },
                },
                format="json",
            )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        kwargs = persona_service_cls.return_value.sample_projected_personas.call_args.kwargs
        self.assertEqual(kwargs["filters"], {"age_bracket": ["25-34"], "region": ["Western Europe"]})
        self.assertEqual(kwargs["stratify"], "age_bracket")

    @patch("apps.personas.views.PersonaService")
    def test_attributes_api_lists_supported_attributes(self, service_cls):
        service_cls.return_value.get_attributes.return_value = {"pool": "p", "attributes": [], "catalog": {}}

        response = APIClient().get("/api/persona/attributes")

        self.assertEqual(
            [(item["key"], item["attribute"]) for item in response.data["protectedAttributes"]],
            [("gender_identity", "gender"), ("demo_ethnicity_broad", "race")],
        )


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
    @patch("apps.prompts.services.generation.PromptGenerationService")
    @patch("apps.prompts.services.generation.PersonaService")
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
                "promptCount": 1,
                "persona": {
                    "enabled": True,
                    "pool": "persona/datasets/matraix-persona-dev-sample",
                    "selectedAttributes": ["demo_ethnicity_broad"],
                    "filters": {},
                    "stratify": None,
                    "seed": 123,
                },
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        population = PromptPopulation.objects.get(pk=response.data["id"])
        prompt = Prompt.objects.get(population=population)
        self.assertEqual(population.metadata["persona"]["sampling_seed"], 123)
        self.assertEqual(population.metadata["prompt_generator"]["mode"], "persona_conditioned_generation")
        self.assertNotIn("fairness", population.metadata)
        self.assertEqual(prompt.metadata["persona_id"], "0042")
        self.assertEqual(prompt.metadata["persona_attributes"]["risk_tolerance"], "medium")

    @override_settings(PROMPT_AGENT_PROVIDER="openai_compatible", PROMPT_AGENT_MODEL_NAME="agent-model")
    @patch("apps.prompts.services.generation.PromptGenerationService")
    @patch("apps.prompts.services.generation.PersonaService")
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
                "promptCount": 3,
                "persona": {
                    "enabled": True,
                    "selectedAttributes": ["demo_ethnicity_broad"],
                    "seed": 123,
                },
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        population = PromptPopulation.objects.get(pk=response.data["id"])
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
    @patch("apps.prompts.services.generation.PromptGenerationService")
    @patch("apps.prompts.services.generation.PersonaService")
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
                "promptCount": 1,
                "persona": {
                    "enabled": True,
                    "selectedAttributes": ["demo_ethnicity_broad"],
                    "seed": 123,
                },
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        prompt = Prompt.objects.get(population_id=response.data["id"])
        self.assertNotIn("expected_label", prompt.metadata)
        prompt_service_cls.return_value.generate_for_personas.assert_called_once()
        kwargs = prompt_service_cls.return_value.generate_for_personas.call_args.kwargs
        self.assertNotIn("desired_labels", kwargs)

    @override_settings(PROMPT_AGENT_PROVIDER="openai_compatible", PROMPT_AGENT_MODEL_NAME="agent-model")
    @patch("apps.prompts.services.generation.PromptGenerationService")
    @patch("apps.prompts.services.generation.PersonaService")
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
                "text": "As a female engineer, I want long-term investment advice.",
                "category": "persona",
                "persona_id": "0042",
                "persona_attributes": {"gender_identity": "Woman", "age_bracket": "25-34"},
                "missing_attributes": [],
            }
        ]

        response = self.client.post(
            f"/api/evaluations/{self.evaluation.id}/prompt-populations/generate",
            {
                "promptCount": 1,
                "persona": {
                    "enabled": True,
                    "selectedAttributes": ["gender_identity", "demo_ethnicity_broad"],
                    "seed": 123,
                },
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        population = PromptPopulation.objects.get(pk=response.data["id"])
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
        self.assertEqual(original.text, "As a female engineer, I want long-term investment advice.")
        self.assertEqual(counterfactual.text, "As a male engineer, I want long-term investment advice.")

    @override_settings(PROMPT_AGENT_PROVIDER="openai_compatible", PROMPT_AGENT_MODEL_NAME="agent-model")
    @patch("apps.prompts.services.generation.PromptGenerationService")
    @patch("apps.prompts.services.generation.PersonaService")
    def test_generate_population_with_picked_personas_skips_sampling(
        self,
        persona_service_cls,
        prompt_service_cls,
    ):
        persona_service_cls.return_value.get_personas_by_ids.return_value = [
            {"persona_id": "p1", "attributes": {"age_bracket": "25-34"}, "missing_attributes": []},
            {"persona_id": "p2", "attributes": {}, "missing_attributes": ["age_bracket"]},
            {"persona_id": "p3", "attributes": {"age_bracket": "35-44"}, "missing_attributes": []},
        ]
        prompt_service_cls.return_value.generate_for_personas.return_value = [
            {"text": f"Prompt {i}", "category": "persona", "persona_id": pid, "persona_attributes": {}}
            for i, pid in enumerate(["p1", "p3", "p1", "p3"])
        ]

        response = self.client.post(
            f"/api/evaluations/{self.evaluation.id}/prompt-populations/generate",
            {
                "promptCount": 4,
                "persona": {
                    "enabled": True,
                    "selectedAttributes": ["demo_ethnicity_broad"],
                    "personaIds": ["p1", "p2", "p3"],
                },
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        persona_service_cls.return_value.sample_projected_personas.assert_not_called()
        personas = prompt_service_cls.return_value.generate_for_personas.call_args.kwargs["personas"]
        # p2 lacks the diagnostic attribute; the others are cycled to 4 prompts.
        self.assertEqual([p["persona_id"] for p in personas], ["p1", "p3", "p1", "p3"])
        population = PromptPopulation.objects.get(pk=response.data["id"])
        persona_meta = population.metadata["persona"]
        self.assertEqual(persona_meta["persona_ids"], ["p1", "p2", "p3"])
        self.assertEqual(persona_meta["excluded_persona_ids"], ["p2"])
        self.assertEqual(persona_meta["unique_persona_count"], 2)
