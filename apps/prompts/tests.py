# pyright: reportMissingImports=false
from unittest.mock import Mock, patch

from django.test import TestCase

from apps.evaluations.models import Evaluation
from .models import PromptPopulation

from .services import (
    append_instruction,
    build_classification_output_instruction,
    build_gender_counterfactual_prompts,
    build_recommendation_output_instruction,
    swap_gender_words,
    with_classification_prompt_constraints,
    with_prompt_constraints,
    with_recommendation_prompt_constraints,
    PromptPopulationGenerationService,
)


class TaskObjectiveGenerationTests(TestCase):
    def test_generates_requested_count_for_each_objective(self):
        evaluation = Evaluation.objects.create(
            name="Support questions",
            use_case_description="Uploaded objectives",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
            task_objectives=[{"objective": "System operation"}, {"objective": "Login help"}],
        )
        population = PromptPopulation.objects.create(
            evaluation=evaluation,
            provider="test",
            model_name="agent",
            requested_prompt_count=4,
        )
        with patch("apps.prompts.services.generation.PromptGenerationService") as service_class:
            service_class.return_value.generate_population.return_value = [{"text": "prompt", "category": ""}]
            prompts = PromptPopulationGenerationService().generate(
                evaluation,
                {"provider": "test", "model_name": "agent", "prompt_count": 1, "temperature": 0.8},
                population,
            )

        self.assertEqual(service_class.return_value.generate_population.call_count, 2)
        self.assertEqual([call.kwargs["evaluation"].use_case_description for call in service_class.return_value.generate_population.call_args_list], ["System operation", "Login help"])
        self.assertEqual([prompt["objective"] for prompt in prompts], ["System operation", "Login help"])


class AppendInstructionTests(TestCase):
    def test_appends_instruction_once(self):
        text = append_instruction("Base prompt.", "Do the thing.")
        self.assertEqual(text, "Base prompt.\n\nDo the thing.")

    def test_replaces_stale_instruction_instead_of_stacking(self):
        text = append_instruction("Base prompt.", "分類輸出格式要求：\nOld labels.")
        text = append_instruction(text, "分類輸出格式要求：\nNew labels.")
        self.assertEqual(text.count("分類輸出格式要求："), 1)
        self.assertIn("New labels.", text)
        self.assertNotIn("Old labels.", text)


class ClassificationPromptConstraintTests(TestCase):
    def test_regenerating_instruction_after_label_edit_does_not_duplicate(self):
        evaluation = Evaluation.objects.create(
            name="Loan approval",
            use_case_description="Decide whether to approve a loan application.",
            task_type=Evaluation.TaskType.CLASSIFICATION,
            classification_labels=[
                {"value": 0, "name": "reject", "description": "不通過"},
                {"value": 1, "name": "approve", "description": "通過"},
            ],
        )
        prompts = with_classification_prompt_constraints(
            evaluation, [{"text": "Applicant has stable income.", "category": ""}]
        )
        text = prompts[0]["text"]

        # Labels changed after the prompt population was generated.
        evaluation.classification_labels = [
            {"value": 0, "name": "deny", "description": "拒絕"},
            {"value": 1, "name": "grant", "description": "核准"},
        ]
        evaluation.save(update_fields=["classification_labels"])

        refreshed = append_instruction(text, build_classification_output_instruction(evaluation))

        self.assertEqual(refreshed.count("分類輸出格式要求："), 1)
        self.assertIn("grant", refreshed)
        self.assertNotIn("approve", refreshed)


class RecommendationPromptConstraintTests(TestCase):
    def _evaluation(self, items):
        return Evaluation.objects.create(
            name="Product recs",
            use_case_description="Recommend the best products for a fitness enthusiast.",
            task_type=Evaluation.TaskType.RECOMMENDATION,
            recommendation_items=items,
        )

    def test_appends_candidate_list_and_selection_only_instruction(self):
        evaluation = self._evaluation(
            [
                {"name": "高蛋白燕麥飲", "description": "低糖、高蛋白"},
                {"name": "瑜伽墊", "description": "適合居家運動"},
                {"name": "藍牙耳機", "description": "適合通勤"},
            ]
        )

        prompts = with_recommendation_prompt_constraints(
            evaluation, [{"text": "我最近想選購一些健身相關商品。", "category": ""}]
        )
        text = prompts[0]["text"]

        self.assertIn("推薦候選清單：", text)
        self.assertIn("高蛋白燕麥飲", text)
        self.assertIn("瑜伽墊", text)
        self.assertIn("藍牙耳機", text)
        self.assertIn("我最近想選購一些健身相關商品。", text)

    def test_noop_for_non_recommendation_tasks(self):
        evaluation = Evaluation.objects.create(
            name="Career advice",
            use_case_description="Ask an AI for career advice",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
        )
        prompts = [{"text": "Some prompt.", "category": ""}]

        self.assertEqual(with_recommendation_prompt_constraints(evaluation, prompts), prompts)

    def test_noop_when_no_candidate_items_defined(self):
        evaluation = self._evaluation([])
        prompts = [{"text": "Some prompt.", "category": ""}]

        self.assertEqual(with_recommendation_prompt_constraints(evaluation, prompts), prompts)

    def test_regenerating_instruction_after_items_edit_does_not_duplicate(self):
        evaluation = self._evaluation(
            [
                {"name": "高蛋白燕麥飲", "description": ""},
                {"name": "瑜伽墊", "description": ""},
            ]
        )
        prompts = with_recommendation_prompt_constraints(
            evaluation, [{"text": "我最近想選購一些健身相關商品。", "category": ""}]
        )
        text = prompts[0]["text"]

        evaluation.recommendation_items = [
            {"name": "無糖豆漿", "description": ""},
            {"name": "運動水壺", "description": ""},
        ]
        evaluation.save(update_fields=["recommendation_items"])

        refreshed = append_instruction(text, build_recommendation_output_instruction(evaluation))

        self.assertEqual(refreshed.count("推薦候選清單："), 1)
        self.assertIn("無糖豆漿", refreshed)
        self.assertNotIn("高蛋白燕麥飲", refreshed)

    def test_with_prompt_constraints_does_not_cross_contaminate_task_types(self):
        evaluation = self._evaluation([{"name": "A", "description": ""}, {"name": "B", "description": ""}])
        prompts = with_prompt_constraints(evaluation, [{"text": "Prompt text.", "category": ""}])

        self.assertIn("推薦候選清單：", prompts[0]["text"])
        self.assertNotIn("分類輸出格式要求：", prompts[0]["text"])


class SwapGenderWordsTests(TestCase):
    def test_swaps_english_words_female_to_male(self):
        text = "As a woman, she believes her career and her mother inspired her."
        swapped = swap_gender_words(text, "female", "male")
        self.assertEqual(
            swapped,
            "As a man, he believes his career and his father inspired him.",
        )

    def test_swaps_english_words_male_to_female(self):
        text = "As a man, he believes his career and his father inspired him."
        swapped = swap_gender_words(text, "male", "female")
        self.assertEqual(
            swapped,
            "As a woman, she believes her career and her mother inspired her.",
        )

    def test_preserves_capitalization(self):
        text = "Woman: Her plan is solid."
        swapped = swap_gender_words(text, "female", "male")
        self.assertEqual(swapped, "Man: His plan is solid.")

    def test_noop_when_groups_are_equal_or_unsupported(self):
        text = "She is great."
        self.assertEqual(swap_gender_words(text, "male", "male"), text)
        self.assertEqual(swap_gender_words(text, "nonbinary", "male"), text)


class GenderCounterfactualWordSwapTests(TestCase):
    def _prompt_service(self, texts):
        service = Mock()
        service.generate_for_personas.return_value = [
            {
                "text": text,
                "category": "persona",
                "persona_id": f"p{index}",
                "persona_attributes": {},
                "missing_attributes": [],
            }
            for index, text in enumerate(texts)
        ]
        return service

    def test_calls_llm_once_per_persona_and_derives_counterfactual_by_swap(self):
        evaluation = Evaluation.objects.create(
            name="ETF advice",
            use_case_description="I want long-term ETF investment advice.",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
        )
        personas = [
            {
                "persona_id": "p001",
                "attributes": {"gender_identity": "Woman", "age": "35"},
                "missing_attributes": [],
            }
        ]
        prompt_service = self._prompt_service(["As a female engineer, I want long-term investment advice."])

        prompts = build_gender_counterfactual_prompts(
            evaluation=evaluation,
            personas=personas,
            prompt_service=prompt_service,
            provider="openai_compatible",
            model_name="agent-model",
            temperature=0.8,
            base_url=None,
            additional_instruction="",
        )

        # Exactly one LLM call for the one eligible persona (not one per pair side).
        prompt_service.generate_for_personas.assert_called_once()
        self.assertEqual(
            len(prompt_service.generate_for_personas.call_args.kwargs["personas"]), 1
        )

        self.assertEqual(len(prompts), 2)
        original, counterfactual = prompts
        self.assertEqual(original["text"], "As a female engineer, I want long-term investment advice.")
        self.assertEqual(counterfactual["text"], "As a male engineer, I want long-term investment advice.")
        self.assertEqual(original["fairness"]["role"], "original")
        self.assertEqual(original["fairness"]["group"], "female")
        self.assertEqual(counterfactual["fairness"]["role"], "counterfactual")
        self.assertEqual(counterfactual["fairness"]["group"], "male")
        self.assertEqual(original["fairness"]["pair_id"], counterfactual["fairness"]["pair_id"])

    def test_skips_persona_without_supported_gender_value(self):
        evaluation = Evaluation.objects.create(
            name="ETF advice",
            use_case_description="Some use case.",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
        )
        personas = [
            {"persona_id": "p001", "attributes": {"age": "35"}, "missing_attributes": ["gender_identity"]},
        ]
        prompt_service = Mock()

        prompts = build_gender_counterfactual_prompts(
            evaluation=evaluation,
            personas=personas,
            prompt_service=prompt_service,
            provider="openai_compatible",
            model_name="agent-model",
            temperature=0.8,
            base_url=None,
            additional_instruction="",
        )

        self.assertEqual(prompts, [])
        prompt_service.generate_for_personas.assert_not_called()

    def test_tags_recommendation_task_when_requested(self):
        evaluation = Evaluation.objects.create(
            name="Product recs",
            use_case_description="Recommend two products.",
            task_type=Evaluation.TaskType.RECOMMENDATION,
        )
        personas = [
            {"persona_id": "p001", "attributes": {"gender_identity": "Man"}, "missing_attributes": []},
        ]
        prompt_service = self._prompt_service(["Please recommend two products for him."])

        prompts = build_gender_counterfactual_prompts(
            evaluation=evaluation,
            personas=personas,
            prompt_service=prompt_service,
            provider="openai_compatible",
            model_name="agent-model",
            temperature=0.8,
            base_url=None,
            additional_instruction="",
            fairness_task="recommendation",
        )

        self.assertTrue(all(item["fairness"]["task"] == "recommendation" for item in prompts))


class SwapGenderWordsEdgeCaseTests(TestCase):
    def test_replaces_titles_with_trailing_period(self):
        self.assertEqual(swap_gender_words("Mrs. Lee met Ms. Wang.", "female", "male"), "Mr. Lee met Mr. Wang.")
        self.assertEqual(swap_gender_words("Mr. Lee is here.", "male", "female"), "Ms. Lee is here.")

    def test_her_as_object_pronoun_becomes_him(self):
        self.assertEqual(swap_gender_words("I gave her the book.", "female", "male"), "I gave him the book.")
        self.assertEqual(swap_gender_words("I told her about it.", "female", "male"), "I told him about it.")
        self.assertEqual(swap_gender_words("Help her", "female", "male"), "Help him")

    def test_her_as_possessive_becomes_his(self):
        self.assertEqual(swap_gender_words("She lost her own keys.", "female", "male"), "He lost his own keys.")

    def test_swap_is_single_pass(self):
        # "his" -> "her" must not be re-swapped by any later rule.
        self.assertEqual(swap_gender_words("I gave him his book.", "male", "female"), "I gave her her book.")


class CounterfactualPairIdTests(TestCase):
    def _prompt_service(self, texts):
        service = Mock()
        service.generate_for_personas.return_value = [{"text": text, "category": ""} for text in texts]
        return service

    def _build(self, texts, **kwargs):
        evaluation = Evaluation.objects.create(
            name="x", use_case_description="x", task_type=Evaluation.TaskType.TEXT_GENERATION
        )
        personas = [
            {"persona_id": f"p{index}", "attributes": {"gender_identity": "Woman"}, "missing_attributes": []}
            for index in range(len(texts))
        ]
        return build_gender_counterfactual_prompts(
            evaluation=evaluation,
            personas=personas,
            prompt_service=self._prompt_service(texts),
            provider="p",
            model_name="m",
            temperature=0.5,
            base_url=None,
            additional_instruction="",
            **kwargs,
        )

    def test_pair_ids_include_objective_index(self):
        first = self._build(["She codes.", "She sings."], objective_index=1)
        second = self._build(["She codes.", "She sings."], objective_index=2)
        ids = {item["fairness"]["pair_id"] for item in first + second}
        self.assertEqual(ids, {"cf_o01_001", "cf_o01_002", "cf_o02_001", "cf_o02_002"})
        self.assertEqual(first[0]["fairness"]["objective_index"], 1)

    def test_skips_pairs_whose_swap_is_identical_and_counts_them(self):
        stats: dict = {}
        prompts = self._build(["I want ETF advice.", "As a woman, I want ETF advice."], stats=stats)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[1]["text"], "As a man, I want ETF advice.")
        self.assertEqual(prompts[0]["fairness"]["pair_id"], "cf_001")
        self.assertEqual(stats["identical_swap_skipped"], 1)


class PromptViewTests(TestCase):
    def test_invalid_params_do_not_create_evaluation(self):
        from rest_framework.test import APIClient

        before = Evaluation.objects.count()
        response = APIClient().post(
            "/api/prompt-populations/generate/",
            {"task": "Some task", "modelName": "m", "promptCount": 501},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(Evaluation.objects.count(), before)

    def test_manual_prompt_sequence_uses_max_plus_one(self):
        from rest_framework.test import APIClient

        evaluation = Evaluation.objects.create(
            name="x", use_case_description="x", task_type=Evaluation.TaskType.TEXT_GENERATION
        )
        population = PromptPopulation.objects.create(
            evaluation=evaluation, provider="manual", model_name="manual", requested_prompt_count=0,
            status=PromptPopulation.Status.COMPLETED,
        )
        population.prompts.create(text="a", sequence=1)
        population.prompts.create(text="b", sequence=3)
        response = APIClient().post(
            f"/api/evaluations/{evaluation.id}/prompts/", {"text": "c"}, format="json"
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["index"], 4)


class PromptPopulationApiShapeTests(TestCase):
    def setUp(self):
        from rest_framework.test import APIClient

        self.client = APIClient()
        self.evaluation = Evaluation.objects.create(
            name="x", use_case_description="x", task_type=Evaluation.TaskType.TEXT_GENERATION
        )

    def make_population(self, **kwargs):
        defaults = {
            "evaluation": self.evaluation,
            "provider": "p",
            "model_name": "m",
            "requested_prompt_count": 1,
            "status": PromptPopulation.Status.COMPLETED,
        }
        return PromptPopulation.objects.create(**{**defaults, **kwargs})

    def test_summary_reports_population_id_status_and_newer_generation(self):
        completed = self.make_population(
            metadata={"persona": {"enabled": True, "selected_attributes": ["gender_identity"], "sampling_seed": 7}}
        )
        generating = self.make_population(status=PromptPopulation.Status.GENERATING)

        body = self.client.get(f"/api/evaluations/{self.evaluation.id}/prompt-population").json()

        self.assertEqual(body["populationId"], completed.id)
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["generatingPopulationId"], generating.id)
        self.assertEqual(body["persona"]["selectedAttributes"], ["gender_identity"])
        self.assertEqual(body["seed"], 7)

    def test_orphaned_generating_population_is_reaped(self):
        from datetime import timedelta

        from django.utils import timezone

        from .services.generation import INTERRUPTED_MESSAGE

        population = self.make_population(status=PromptPopulation.Status.GENERATING)
        PromptPopulation.objects.filter(pk=population.pk).update(
            created_at=timezone.now() - timedelta(minutes=5)
        )

        body = self.client.get(f"/api/prompt-populations/{population.id}/").json()

        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["errorMessage"], INTERRUPTED_MESSAGE)
        summary = self.client.get(f"/api/evaluations/{self.evaluation.id}/prompt-population").json()
        self.assertIsNone(summary["generatingPopulationId"])

    def test_prompt_list_is_camel_case_but_keeps_data_keys(self):
        population = self.make_population()
        population.prompts.create(
            text="hello",
            sequence=1,
            metadata={
                "persona_id": "0042",
                "persona_attributes": {"gender_identity": "Woman"},
                "missing_attributes": [],
                "fairness": {"pair_id": "cf_001", "role": "original", "source_persona_id": "0042"},
            },
        )

        [prompt] = self.client.get(f"/api/evaluations/{self.evaluation.id}/prompts").json()

        self.assertEqual(prompt["text"], "hello")
        self.assertNotIn("prompt", prompt)
        self.assertEqual(prompt["personaId"], "0042")
        self.assertEqual(prompt["personaAttributes"], {"gender_identity": "Woman"})
        self.assertEqual(prompt["fairness"]["pairId"], "cf_001")
        self.assertEqual(prompt["fairness"]["sourcePersonaId"], "0042")
