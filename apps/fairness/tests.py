# pyright: reportMissingImports=false
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase

from .chinese_stereotype import (
    _chinese_safe_transform_token,
    _merge_lexicon_tokens,
    looks_like_chinese,
)
from .services import (
    LangFairMetricService,
    build_counterfactual_text_pairs,
    build_recommendation_metric_inputs,
    extract_recommendation_items,
    infer_stereotype_categories,
    resolve_toxicity_classifiers,
)


def _response(response_id, response_text, fairness, persona_attributes=None):
    metadata = {"fairness": fairness}
    if persona_attributes is not None:
        metadata["persona_attributes"] = persona_attributes
    prompt = SimpleNamespace(metadata=metadata)
    return SimpleNamespace(id=response_id, prompt=prompt, response_text=response_text)


class ExtractRecommendationItemsTests(TestCase):
    def test_extracts_numbered_list(self):
        text = "推薦如下：\n1. 高蛋白燕麥飲：低糖高蛋白\n2. 無糖堅果穀物棒：健康補給"
        self.assertEqual(
            extract_recommendation_items(text),
            ["高蛋白燕麥飲", "無糖堅果穀物棒"],
        )

    def test_extracts_bulleted_list(self):
        text = "- Item A\n- Item B: some reason\n* Item C"
        self.assertEqual(extract_recommendation_items(text), ["Item A", "Item B", "Item C"])

    def test_returns_empty_for_free_form_text(self):
        text = "I think you would enjoy trying out some new snacks."
        self.assertEqual(extract_recommendation_items(text), [])

    def test_matches_candidate_items_by_first_occurrence_regardless_of_formatting(self):
        # No numbered/bulleted formatting at all -- the rule-based parser alone
        # would return nothing here, but the target LLM was constrained to
        # choose only from this fixed candidate list, so exact-name matching
        # should still find them in the order they appear.
        text = "我會推薦你先試試瑜伽墊，接著再考慮高蛋白燕麥飲，兩者都很適合你。"
        candidate_items = ["高蛋白燕麥飲", "瑜伽墊", "藍牙耳機"]

        self.assertEqual(
            extract_recommendation_items(text, candidate_items),
            ["瑜伽墊", "高蛋白燕麥飲"],
        )

    def test_falls_back_to_rule_based_parser_when_no_candidate_matches_found(self):
        text = "1. Item A\n2. Item B"
        candidate_items = ["高蛋白燕麥飲", "瑜伽墊"]

        self.assertEqual(extract_recommendation_items(text, candidate_items), ["Item A", "Item B"])


class BuildRecommendationMetricInputsTests(TestCase):
    def test_pairs_original_and_counterfactual_by_pair_id(self):
        rows = [
            _response(
                1,
                "1. Item A\n2. Item B\n3. Item C",
                {"task": "recommendation", "pair_id": "cf_001", "role": "original"},
            ),
            _response(
                2,
                "1. Item C\n2. Item A\n3. Item B",
                {"task": "recommendation", "pair_id": "cf_001", "role": "counterfactual"},
            ),
        ]

        result = build_recommendation_metric_inputs(rows)

        self.assertEqual(result["inputs"]["mode"], "pairwise")
        self.assertEqual(result["inputs"]["recLists1"], [["Item A", "Item B", "Item C"]])
        self.assertEqual(result["inputs"]["recLists2"], [["Item C", "Item A", "Item B"]])
        self.assertEqual(result["extractionFailures"], [])

    def test_truncates_mismatched_list_lengths_to_common_minimum(self):
        rows = [
            _response(
                1,
                "1. Item A\n2. Item B\n3. Item C",
                {"task": "recommendation", "pair_id": "cf_001", "role": "original"},
            ),
            _response(
                2,
                "1. Item C\n2. Item A",
                {"task": "recommendation", "pair_id": "cf_001", "role": "counterfactual"},
            ),
        ]

        result = build_recommendation_metric_inputs(rows)

        self.assertEqual(result["inputs"]["recLists1"], [["Item A", "Item B"]])
        self.assertEqual(result["inputs"]["recLists2"], [["Item C", "Item A"]])

    def test_truncates_all_pairs_to_shortest_list_across_the_whole_batch(self):
        # LangFair's evaluate_pairwise requires every list, across every pair,
        # to share one length -- not just matched within a single pair.
        rows = [
            _response(
                1,
                "1. Item A\n2. Item B\n3. Item C",
                {"task": "recommendation", "pair_id": "cf_001", "role": "original"},
            ),
            _response(
                2,
                "1. Item C\n2. Item A\n3. Item B",
                {"task": "recommendation", "pair_id": "cf_001", "role": "counterfactual"},
            ),
            _response(
                3,
                "1. Item D\n2. Item E",
                {"task": "recommendation", "pair_id": "cf_002", "role": "original"},
            ),
            _response(
                4,
                "1. Item E\n2. Item D",
                {"task": "recommendation", "pair_id": "cf_002", "role": "counterfactual"},
            ),
        ]

        result = build_recommendation_metric_inputs(rows)

        self.assertEqual(result["commonListLength"], 2)
        self.assertEqual(result["inputs"]["recLists1"], [["Item A", "Item B"], ["Item D", "Item E"]])
        self.assertEqual(result["inputs"]["recLists2"], [["Item C", "Item A"], ["Item E", "Item D"]])

    def test_records_extraction_failure_when_no_list_found(self):
        rows = [
            _response(
                1,
                "I recommend whatever fits your taste.",
                {"task": "recommendation", "pair_id": "cf_001", "role": "original"},
            ),
            _response(
                2,
                "1. Item A\n2. Item B",
                {"task": "recommendation", "pair_id": "cf_001", "role": "counterfactual"},
            ),
        ]

        result = build_recommendation_metric_inputs(rows)

        self.assertEqual(result["status"], "missing_input")

    def test_ignores_rows_not_tagged_as_recommendation_or_counterfactual_pairs(self):
        rows = [
            _response(1, "1. Item A\n2. Item B", {"task": "classification", "pair_id": "cf_001", "role": "original"}),
        ]

        result = build_recommendation_metric_inputs(rows)

        self.assertEqual(result["status"], "missing_input")

    def test_reuses_counterfactual_tagged_pairs_with_list_shaped_responses(self):
        rows = [
            _response(
                1,
                "1. 指數型基金（ETF）\n2. 債券基金",
                {"task": "counterfactual", "attribute": "gender", "pair_id": "cf_001", "role": "original"},
            ),
            _response(
                2,
                "1. 股票市場\n2. 債券投資",
                {"task": "counterfactual", "attribute": "gender", "pair_id": "cf_001", "role": "counterfactual"},
            ),
        ]

        result = build_recommendation_metric_inputs(rows)

        self.assertEqual(result["inputs"]["mode"], "pairwise")
        self.assertEqual(result["inputs"]["recLists1"], [["指數型基金（ETF）", "債券基金"]])
        self.assertEqual(result["inputs"]["recLists2"], [["股票市場", "債券投資"]])
        self.assertEqual(result["extractionFailures"], [])

    def test_uses_candidate_items_for_extraction_when_configured(self):
        # Free-form prose with no numbered/bulleted formatting -- only
        # possible to extract because a fixed candidate list constrains what
        # the target LLM was allowed to recommend.
        rows = [
            _response(
                1,
                "我會推薦瑜伽墊，也很推薦高蛋白燕麥飲。",
                {"task": "recommendation", "pair_id": "cf_001", "role": "original"},
            ),
            _response(
                2,
                "高蛋白燕麥飲很適合你，藍牙耳機也不錯。",
                {"task": "recommendation", "pair_id": "cf_001", "role": "counterfactual"},
            ),
        ]

        result = build_recommendation_metric_inputs(
            rows, candidate_items=["高蛋白燕麥飲", "瑜伽墊", "藍牙耳機"]
        )

        self.assertEqual(result["inputs"]["recLists1"], [["瑜伽墊", "高蛋白燕麥飲"]])
        self.assertEqual(result["inputs"]["recLists2"], [["高蛋白燕麥飲", "藍牙耳機"]])
        self.assertEqual(result["extractionFailures"], [])


class BuildCounterfactualTextPairsTests(TestCase):
    def test_groups_by_protected_group_not_role(self):
        # Pair cf_001's original persona was sampled female; pair cf_002's
        # original persona was sampled male. Grouping by role (the old
        # behavior) would put cf_001's female response and cf_002's male
        # response on the same side (texts1) purely because both happen to be
        # "original" -- washing out a real gender effect for distributional
        # metrics like Sentiment Bias. Grouping by resolved group keeps every
        # female response on one side and every male response on the other,
        # regardless of which one was generated first.
        rows = [
            _response(
                1,
                "female original response",
                {"task": "counterfactual", "attribute": "gender", "pair_id": "cf_001", "role": "original", "group": "female"},
            ),
            _response(
                2,
                "male counterfactual response",
                {"task": "counterfactual", "attribute": "gender", "pair_id": "cf_001", "role": "counterfactual", "group": "male"},
            ),
            _response(
                3,
                "male original response",
                {"task": "counterfactual", "attribute": "gender", "pair_id": "cf_002", "role": "original", "group": "male"},
            ),
            _response(
                4,
                "female counterfactual response",
                {"task": "counterfactual", "attribute": "gender", "pair_id": "cf_002", "role": "counterfactual", "group": "female"},
            ),
        ]

        result = build_counterfactual_text_pairs(rows)

        self.assertEqual(result["groups"], ["female", "male"])
        self.assertEqual(result["texts1"], ["female original response", "female counterfactual response"])
        self.assertEqual(result["texts2"], ["male counterfactual response", "male original response"])

    def test_returns_none_when_a_pair_is_missing_its_counterfactual_side(self):
        rows = [
            _response(
                1,
                "female original response",
                {"task": "counterfactual", "attribute": "gender", "pair_id": "cf_001", "role": "original", "group": "female"},
            ),
        ]

        self.assertIsNone(build_counterfactual_text_pairs(rows))

    def test_ignores_rows_not_tagged_as_counterfactual(self):
        rows = [
            _response(
                1,
                "resp",
                {"task": "recommendation", "pair_id": "cf_001", "role": "original", "group": "female"},
            ),
        ]

        self.assertIsNone(build_counterfactual_text_pairs(rows))


class InferStereotypeCategoriesTests(TestCase):
    def test_infers_category_from_fairness_attribute(self):
        rows = [
            _response(
                1,
                "resp",
                {"task": "counterfactual", "attribute": "gender", "pair_id": "cf_001", "role": "original", "group": "female"},
            ),
        ]

        self.assertEqual(infer_stereotype_categories(rows), ["gender"])

    def test_ignores_persona_attribute_keys_not_part_of_the_diagnosed_attribute(self):
        # persona_attributes now carries every dimension MatrAIx returns (age,
        # ethnicity, etc.), not just the attribute actually being diagnosed
        # (see PersonaService.get_projected_persona). A stray ethnicity-like
        # key riding along on the persona must not silently add "race" to the
        # categories scored -- no race counterfactual pair was ever built for
        # this run, only a gender one.
        rows = [
            _response(
                1,
                "resp",
                {"task": "counterfactual", "attribute": "gender", "pair_id": "cf_001", "role": "original", "group": "female"},
                persona_attributes={
                    "gender_identity": "Woman",
                    "age_bracket": "25-34",
                    "demo_ethnicity_broad": "Asian",
                },
            ),
        ]

        self.assertEqual(infer_stereotype_categories(rows), ["gender"])

    def test_defaults_to_gender_when_nothing_found(self):
        rows = [_response(1, "resp", {})]

        self.assertEqual(infer_stereotype_categories(rows), ["gender"])


class ResolveToxicityClassifiersTests(TestCase):
    def test_defaults_to_english_classifier_for_english_responses(self):
        self.assertEqual(
            resolve_toxicity_classifiers(["This is a plain English sentence about investing."], None),
            ["detoxify_unbiased"],
        )

    def test_defaults_to_multilingual_classifier_for_chinese_responses(self):
        self.assertEqual(
            resolve_toxicity_classifiers(["這是一段繁體中文的回應內容，包含許多中文字元。"], None),
            ["detoxify_multilingual"],
        )

    def test_respects_explicitly_requested_classifiers_even_for_chinese(self):
        self.assertEqual(
            resolve_toxicity_classifiers(["這是一段繁體中文的回應內容。"], ["toxigen"]),
            ["toxigen"],
        )


class LooksLikeChineseTests(TestCase):
    def test_detects_cjk_dominant_text(self):
        self.assertTrue(looks_like_chinese(["這是一個關於投資建議的中文句子，包含許多中文字元。"]))

    def test_rejects_english_text(self):
        self.assertFalse(looks_like_chinese(["This is a plain English sentence about investing."]))

    def test_rejects_empty_input(self):
        self.assertFalse(looks_like_chinese([]))
        self.assertFalse(looks_like_chinese([""]))


class EvaluateStereotypeChineseLexiconTests(TestCase):
    def test_tokenizer_segments_chinese_and_keeps_unicode_words(self):
        tokens = _merge_lexicon_tokens(
            ["女", "士", "適合", "保守", "的", "投資", "組合", "。"],
            ["女士", "保守"],
        )

        self.assertIn("女士", tokens)
        self.assertIn("保守", tokens)
        self.assertEqual(_chinese_safe_transform_token("保守。"), "保守")

    def test_auto_detects_chinese_and_computes_association_and_cooccurrence(self):
        # "保守"/"積極" (target adjectives) each co-occur with both "女士" and
        # "男士" (gender words) across the corpus, so both word-list metrics
        # have enough signal to return a real score instead of None.
        responses = [
            "女士適合保守的投資組合，重視穩健理財。",
            "男士適合保守的投資組合，也重視積極理財。",
            "女士通常偏好積極型基金以追求成長。",
            "男士通常偏好積極型基金以追求成長。",
        ]

        result = LangFairMetricService().evaluate_stereotype(
            responses=responses,
            options={"metrics": ["Stereotype Association", "Cooccurrence Bias"]},
        )

        self.assertEqual(result["lexicon"], "chinese")
        self.assertIsNotNone(result["metrics"]["Stereotype Association"])
        self.assertIsNotNone(result["metrics"]["Cooccurrence Bias"])
    def test_keeps_default_english_lexicon_for_english_responses(self):
        responses = [
            "The woman is a nurse who is very caring and gentle.",
            "The man is a soldier who is very strong and brave.",
        ]

        result = LangFairMetricService().evaluate_stereotype(
            responses=responses,
            options={"metrics": ["Stereotype Association", "Cooccurrence Bias"]},
        )

        self.assertEqual(result["lexicon"], "default")

    def test_explicit_use_chinese_lexicon_overrides_autodetection(self):
        responses = ["The woman is a nurse."]

        result = LangFairMetricService().evaluate_stereotype(
            responses=responses,
            options={
                "metrics": ["Stereotype Association", "Cooccurrence Bias"],
                "useChineseLexicon": True,
            },
        )

        self.assertEqual(result["lexicon"], "chinese")

    def test_accepts_custom_chinese_lexicon(self):
        result = LangFairMetricService().evaluate_stereotype(
            responses=[
                "女性適合創新工作。",
                "男性適合創新工作。",
            ],
            options={
                "metrics": ["Stereotype Association", "Cooccurrence Bias"],
                "chineseDemographicWordLists": {
                    "female": ["女性"],
                    "male": ["男性"],
                },
                "chineseStereotypeWordList": ["創新"],
            },
        )

        self.assertEqual(result["lexicon"], "chinese")
        self.assertIsNotNone(result["metrics"]["Stereotype Association"])
        self.assertIsNotNone(result["metrics"]["Cooccurrence Bias"])


class EvaluateCounterfactualChineseSentimentTests(TestCase):
    @patch("apps.fairness.chinese_sentiment.XlmRobertaSentimentClassifier")
    @patch("langfair.metrics.counterfactual.counterfactual.CounterfactualMetrics")
    def test_uses_xlm_roberta_for_chinese_sentiment(self, metrics_class, classifier_class):
        metrics_class.return_value.evaluate.return_value = {"metrics": {"Cosine": 0.9}}

        result = LangFairMetricService().evaluate_counterfactual(
            options={
                "texts1": ["女性適合保守投資。"],
                "texts2": ["男性適合積極投資。"],
            }
        )

        classifier_class.assert_called_once_with(device="cpu")
        self.assertEqual(len(metrics_class.call_args.kwargs["metrics"]), 4)
        self.assertEqual(result["metrics"]["Cosine"], 0.9)
