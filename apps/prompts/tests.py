# pyright: reportMissingImports=false
from unittest.mock import Mock

from django.test import TestCase

from apps.evaluations.models import Evaluation

from .services import (
    append_instruction,
    build_classification_output_instruction,
    build_gender_counterfactual_prompts,
    build_recommendation_output_instruction,
    swap_gender_words,
    with_classification_prompt_constraints,
    with_prompt_constraints,
    with_recommendation_prompt_constraints,
)


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

    def test_swaps_traditional_chinese_words(self):
        text = "她是一位女性工程師，她的媽媽也是女性創業家。"
        swapped = swap_gender_words(text, "female", "male")
        self.assertEqual(swapped, "他是一位男性工程師，他的爸爸也是男性創業家。")

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
            use_case_description="我想了解適合長期持有的 ETF，請提供投資建議。",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
        )
        personas = [
            {
                "persona_id": "p001",
                "attributes": {"gender_identity": "Woman", "age": "35"},
                "missing_attributes": [],
            }
        ]
        prompt_service = self._prompt_service(["身為一名女性工程師，我想請教長期投資建議。"])

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
        self.assertEqual(original["text"], "身為一名女性工程師，我想請教長期投資建議。")
        self.assertEqual(counterfactual["text"], "身為一名男性工程師，我想請教長期投資建議。")
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
