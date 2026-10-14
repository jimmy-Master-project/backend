# pyright: reportMissingImports=false
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient

from apps.evaluations.models import Evaluation
from apps.fairness.models import LangFairMetricResult
from apps.fairness.services import runner as runner_module
from apps.prompts.models import Prompt, PromptPopulation
from apps.responses.models import LLMResponse
from apps.runs.models import EvaluationRun

from .services import (
    LangFairMetricJobRunner,
    LangFairMetricService,
    normalize_metric_options,
    prune_results,
    build_counterfactual_text_pairs,
    build_recommendation_metric_inputs,
    extract_recommendation_items,
    infer_stereotype_categories,
)


def _response(response_id, response_text, fairness, persona_attributes=None, task_objective=None, generation_index=1):
    metadata = {"fairness": fairness}
    if persona_attributes is not None:
        metadata["persona_attributes"] = persona_attributes
    if task_objective is not None:
        metadata["task_objective"] = task_objective
    prompt = SimpleNamespace(metadata=metadata)
    return SimpleNamespace(
        id=response_id, prompt=prompt, response_text=response_text, generation_index=generation_index
    )


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


class _RunFixtureMixin:
    def make_run(self, texts=("response",), errors=None, run_status=EvaluationRun.Status.COMPLETED):
        evaluation = Evaluation.objects.create(
            name="e", use_case_description="u", task_type=Evaluation.TaskType.TEXT_GENERATION
        )
        population = PromptPopulation.objects.create(
            evaluation=evaluation, provider="p", model_name="m", requested_prompt_count=len(texts)
        )
        run = EvaluationRun.objects.create(
            evaluation=evaluation, prompt_population=population, status=run_status
        )
        errors = errors or [None] * len(texts)
        for index, (text, error) in enumerate(zip(texts, errors, strict=True), start=1):
            prompt = Prompt.objects.create(population=population, text=f"prompt {index}", sequence=index)
            LLMResponse.objects.create(
                run=run, prompt=prompt, generation_index=1, response_text=text, error=error
            )
        return run


class LangFairMetricCacheTests(_RunFixtureMixin, TestCase):
    @patch.object(LangFairMetricService, "evaluate_toxicity")
    def test_reuses_persisted_result(self, evaluate_toxicity):
        run = self.make_run()
        evaluate_toxicity.return_value = {"score": 1}

        service = LangFairMetricService()
        calculated = service.evaluate_run(run, suites=["toxicity"])
        self.assertEqual(calculated["suites"]["toxicity"], {"score": 1})
        self.assertEqual(service.evaluate_run(run, suites=["toxicity"]), calculated)
        evaluate_toxicity.assert_called_once()
        self.assertEqual(LangFairMetricResult.objects.get().status, "completed")

    @patch.object(LangFairMetricService, "evaluate_toxicity")
    def test_excludes_failed_responses(self, evaluate_toxicity):
        run = self.make_run(texts=("ok", ""), errors=[None, "timeout after retries"])
        evaluate_toxicity.return_value = {"score": 1}

        result = LangFairMetricService().evaluate_run(run, suites=["toxicity"])

        self.assertEqual(evaluate_toxicity.call_args.kwargs["responses"], ["ok"])
        self.assertEqual(result["responseCount"], 1)
        self.assertEqual(result["excludedResponseCount"], 1)

    @patch.object(LangFairMetricService, "evaluate_toxicity")
    def test_nan_and_inf_become_null(self, evaluate_toxicity):
        run = self.make_run()
        evaluate_toxicity.return_value = {"a": float("nan"), "b": [float("inf"), 1.5]}

        result = LangFairMetricService().evaluate_run(run, suites=["toxicity"])

        self.assertEqual(result["suites"]["toxicity"], {"a": None, "b": [None, 1.5]})

    def test_cache_key_depends_on_evaluation_inputs(self):
        run = self.make_run()
        service = LangFairMetricService()
        key1 = service.prepare_run(run, ["toxicity"]).cache_key
        self.assertEqual(key1, service.prepare_run(run, ["toxicity"]).cache_key)
        run.evaluation.task_type = Evaluation.TaskType.RECOMMENDATION
        run.evaluation.recommendation_items = [{"name": "A"}]
        self.assertNotEqual(key1, service.prepare_run(run, ["toxicity"]).cache_key)

    def test_prune_keeps_latest_rows(self):
        run = self.make_run()
        for index in range(7):
            LangFairMetricResult.objects.create(run=run, cache_key=f"k{index}", status="completed")
        prune_results(run.id, keep=3)
        self.assertEqual(
            sorted(LangFairMetricResult.objects.values_list("cache_key", flat=True)), ["k4", "k5", "k6"]
        )


class MetricOptionsTests(TestCase):
    def test_rejects_unknown_models_and_devices(self):
        with self.assertRaises(ValueError):
            normalize_metric_options({"stereotype": {"classifierModel": "evil/model"}})
        with self.assertRaises(ValueError):
            normalize_metric_options({"counterfactual": {"transformer": "other"}})
        with self.assertRaises(ValueError):
            normalize_metric_options({"toxicity": {"device": "cuda:7"}})

    def test_allows_defaults_and_settings_whitelist(self):
        options = {"stereotype": {"classifierModel": "wu981526092/Sentence-Level-Stereotype-Detector"}}
        self.assertEqual(normalize_metric_options(options), options)
        with self.settings(LANGFAIR_ALLOWED_TRANSFORMERS=["all-mpnet-base-v2"]):
            normalize_metric_options({"counterfactual": {"transformer": "all-mpnet-base-v2"}})

    def test_clamps_batch_size(self):
        self.assertEqual(normalize_metric_options({"toxicity": {"batchSize": 10_000}})["toxicity"]["batchSize"], 256)
        self.assertEqual(normalize_metric_options({"toxicity": {"batchSize": 0}})["toxicity"]["batchSize"], 1)


@patch("apps.fairness.services.runner.threading.Thread", new=_ImmediateThread)
class RunLangFairMetricsApiTests(_RunFixtureMixin, TestCase):
    def setUp(self):
        self.client = APIClient()

    def url(self, run):
        return f"/api/runs/{run.id}/fairness-metrics/"

    @patch.object(LangFairMetricService, "evaluate_toxicity", return_value={"score": float("nan")})
    def test_post_starts_background_job_then_returns_cached(self, evaluate_toxicity):
        run = self.make_run(run_status=EvaluationRun.Status.RUNNING)
        self.assertEqual(self.client.get(self.url(run), {"suite": "toxicity"}).json()["status"], "none")

        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.post(self.url(run), {"suites": ["toxicity"]}, format="json")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "running")
        self.assertTrue(response.json()["partial"])

        response = self.client.post(self.url(run), {"suites": ["toxicity"]}, format="json")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["runStatus"], "running")
        self.assertIsNone(body["suites"]["toxicity"]["score"])

        body = self.client.get(self.url(run), {"suite": "toxicity"}).json()
        self.assertEqual(body["status"], "completed")
        evaluate_toxicity.assert_called_once()

    @patch.object(LangFairMetricService, "evaluate_toxicity", side_effect=RuntimeError("model exploded"))
    def test_failed_job_is_reported(self, _evaluate_toxicity):
        run = self.make_run()
        with self.assertLogs("apps.fairness.services.runner", level="ERROR"):
            with self.captureOnCommitCallbacks(execute=True):
                self.client.post(self.url(run), {"suites": ["toxicity"]}, format="json")
        body = self.client.get(self.url(run), {"suite": "toxicity"}).json()
        self.assertEqual(body["status"], "failed")
        self.assertIn("model exploded", body["error"])
        self.assertFalse(body["partial"])

    def test_dedupes_running_job_and_reaps_stale_rows(self):
        run = self.make_run()
        key = LangFairMetricService().prepare_run(run, ["toxicity"]).cache_key
        row = LangFairMetricResult.objects.create(run=run, cache_key=key, status="running")

        runner_module._ACTIVE_JOBS.add(row.id)
        try:
            with patch.object(LangFairMetricJobRunner, "_spawn") as spawn:
                response = self.client.post(self.url(run), {"suites": ["toxicity"]}, format="json")
            self.assertEqual(response.status_code, 202)
            spawn.assert_not_called()
        finally:
            runner_module._ACTIVE_JOBS.discard(row.id)

        # No live thread any more (e.g. after a server restart) -> failed on read.
        body = self.client.get(self.url(run), {"suite": "toxicity"}).json()
        self.assertEqual(body["status"], "failed")

    def test_rejects_disallowed_model(self):
        run = self.make_run()
        response = self.client.post(
            self.url(run),
            {"suites": ["stereotype"], "options": {"stereotype": {"classifierModel": "x/y"}}},
            format="json",
        )
        self.assertEqual(response.status_code, 400)


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


class PairKeyCollisionTests(TestCase):
    def _pair_rows(self, pair_ids_by_objective, task="counterfactual", texts=None):
        rows = []
        counter = 0
        for objective, pair_ids in pair_ids_by_objective.items():
            for pair_id in pair_ids:
                for role, group in (("original", "female"), ("counterfactual", "male")):
                    counter += 1
                    rows.append(
                        _response(
                            counter,
                            texts(counter) if texts else f"{objective} {pair_id} {group}",
                            {"task": task, "attribute": "gender", "pair_id": pair_id, "role": role, "group": group},
                            task_objective=objective,
                        )
                    )
        return rows

    def test_legacy_pair_ids_repeated_per_objective_are_all_counted(self):
        # Old populations restarted cf_001 for every task objective.
        rows = self._pair_rows({"Objective A": ["cf_001", "cf_002"], "Objective B": ["cf_001", "cf_002"]})
        result = build_counterfactual_text_pairs(rows)
        self.assertEqual(len(result["texts1"]), 4)
        self.assertEqual(len(set(result["texts1"])), 4)

    def test_new_globally_unique_pair_ids_are_all_counted(self):
        rows = self._pair_rows(
            {"Objective A": ["cf_o01_001", "cf_o01_002"], "Objective B": ["cf_o02_001", "cf_o02_002"]}
        )
        self.assertEqual(len(build_counterfactual_text_pairs(rows)["texts1"]), 4)

    def test_recommendation_pairs_keyed_by_objective_and_generation(self):
        rows = self._pair_rows(
            {"Objective A": ["cf_001"], "Objective B": ["cf_001"]},
            task="recommendation",
            texts=lambda index: f"1. Item {index}\n2. Item X",
        )
        # A second generation of the same pair must not overwrite the first.
        rows += [
            _response(90, "1. Item G\n2. Item H", {"task": "recommendation", "pair_id": "cf_001", "role": "original"},
                      task_objective="Objective A", generation_index=2),
            _response(91, "1. Item H\n2. Item G", {"task": "recommendation", "pair_id": "cf_001", "role": "counterfactual"},
                      task_objective="Objective A", generation_index=2),
        ]
        result = build_recommendation_metric_inputs(rows)
        self.assertEqual(len(result["inputs"]["recLists1"]), 3)


class EnglishOnlySuiteTests(_RunFixtureMixin, TestCase):
    @patch.object(LangFairMetricService, "evaluate_toxicity")
    def test_refuses_english_only_suites_for_chinese_responses(self, evaluate_toxicity):
        run = self.make_run(texts=("這是一個中文回答。", "English answer"))
        result = LangFairMetricService().evaluate_run(run, suites=["toxicity"])
        suite = result["suites"]["toxicity"]
        self.assertEqual(suite["status"], "unsupported_language")
        self.assertFalse(result["languageCheck"]["isEnglish"])
        self.assertEqual(result["languageCheck"]["nonEnglishResponseCount"], 1)
        evaluate_toxicity.assert_not_called()

    @patch.object(LangFairMetricService, "evaluate_recommendation_for_run")
    @patch.object(LangFairMetricService, "evaluate_classification_for_run")
    def test_refuses_every_suite_of_a_non_english_run(self, evaluate_classification, evaluate_recommendation):
        run = self.make_run(texts=("這是一個中文回答。",))
        result = LangFairMetricService().evaluate_run(run, suites=["classification", "recommendation"])
        for suite in ("classification", "recommendation"):
            self.assertEqual(result["suites"][suite]["status"], "unsupported_language")
        evaluate_classification.assert_not_called()
        evaluate_recommendation.assert_not_called()

    def test_ascii_ratio_classifies_text(self):
        from .services.utils import ascii_ratio

        self.assertEqual(ascii_ratio("Hello, world!"), 1.0)
        self.assertLess(ascii_ratio("建議您投資 ETF 0050"), 0.9)
        self.assertGreaterEqual(ascii_ratio("The café’s “menu” is great"), 0.9)
        self.assertIsNone(ascii_ratio("   "))


class GroupComparisonTests(TestCase):
    def test_reports_per_group_values_and_signed_gap(self):
        from langfair.metrics.utils.classifier_metrics import Fraction

        from .services.grouping import group_comparison

        result = group_comparison(
            scores=[0.9, 0.1, 0.1, 0.1],
            prompts=["p1", "p2", "p3", "p4"],
            groups=["female", "female", "male", "male"],
            threshold=0.5,
            metric_functions={"Toxic Fraction": Fraction.metric_function},
            by_prompt=True,
        )
        self.assertEqual(result["byGroup"]["female"]["Toxic Fraction"], 0.5)
        self.assertEqual(result["byGroup"]["male"]["Toxic Fraction"], 0.0)
        self.assertEqual(result["gap"]["metrics"]["Toxic Fraction"], 0.5)
        self.assertEqual(result["comparison"], "female - male")
        self.assertIn("Toxic Fraction", result["gap"]["confidenceIntervals"])

    def test_returns_none_with_a_single_group(self):
        from .services.grouping import group_comparison

        self.assertIsNone(
            group_comparison([0.1], ["p"], ["female"], 0.5, {"x": lambda scores, t: 0.0}, True)
        )


class NormalizedRecommendationMetricTests(TestCase):
    def test_identical_lists_score_one_for_every_k(self):
        from .services.suites.recommendation import normalized_pair_scores

        for k in (1, 3, 5):
            items = [f"item {i}" for i in range(k)]
            scores = normalized_pair_scores([items], [items], ["Jaccard", "PRAG", "SERP"])
            for name, values in scores.items():
                self.assertAlmostEqual(values[0], 1.0, msg=f"{name} K={k}")

    def test_scores_are_symmetric(self):
        from .services.suites.recommendation import normalized_pair_scores

        a, b = ["A", "B", "C"], ["C", "A", "D"]
        forward = normalized_pair_scores([a], [b], ["PRAG", "SERP"])
        backward = normalized_pair_scores([b], [a], ["PRAG", "SERP"])
        self.assertEqual(forward, backward)


class ClassificationGapTests(TestCase):
    def test_group_rate_gap_is_signed_difference(self):
        from .services.suites.classification import build_group_rate_gaps

        result = build_group_rate_gaps(["f", "f", "m", "m"], [1, 1, 1, 0], None)
        self.assertEqual(result["metrics"], {"Predicted Positive Rate": 0.5})
        self.assertEqual(result["comparison"], "f - m")

    def test_paired_decision_summary_counts_flips(self):
        from .services.suites.classification import build_paired_decision_summary

        paired = {
            ("o", "cf_1", 1): {"f": 1, "m": 0},
            ("o", "cf_2", 1): {"f": 1, "m": 1},
            ("o", "cf_3", 1): {"f": 0, "m": 0},
            ("o", "cf_4", 1): {"f": 1, "m": 0},
        }
        result = build_paired_decision_summary(paired)
        self.assertEqual(result["pairCount"], 4)
        self.assertEqual(result["flipCount"], 2)
        self.assertEqual(result["metrics"]["Decision Flip Rate"], 0.5)
        self.assertEqual(result["positiveOnlyFor"], {"f": 2, "m": 0})
        self.assertAlmostEqual(result["mcnemarExactPValue"], 0.5)


class IdenticalPromptPairTests(TestCase):
    def _row(self, response_id, prompt, group, role, pair_id="cf_001"):
        fairness = {"task": "counterfactual", "attribute": "gender", "pair_id": pair_id, "role": role, "group": group}
        return SimpleNamespace(
            id=response_id,
            prompt=SimpleNamespace(text=prompt, metadata={"fairness": fairness}),
            response_text=f"resp {response_id}",
            generation_index=1,
        )

    def test_excludes_pairs_whose_prompts_did_not_change(self):
        rows = [
            self._row(1, "I am a woman.", "female", "original"),
            self._row(2, "I am a man.", "male", "counterfactual"),
            self._row(3, "I am a person.", "female", "original", "cf_002"),
            self._row(4, "I am a person.", "male", "counterfactual", "cf_002"),
        ]
        result = build_counterfactual_text_pairs(rows)
        self.assertEqual(result["texts1"], ["resp 1"])
        self.assertEqual(result["excludedIdenticalPromptPairs"], 1)


class ToxicityClassifierOptionTests(TestCase):
    def test_rejects_toxigen(self):
        with self.assertRaises(ValueError):
            normalize_metric_options({"toxicity": {"classifiers": ["toxigen"]}})
        self.assertEqual(
            normalize_metric_options({"toxicity": {"classifiers": ["detoxify_unbiased"]}})["toxicity"]["classifiers"],
            ["detoxify_unbiased"],
        )


class ClusteredConfidenceIntervalTests(TestCase):
    """Generations of one prompt pair are dependent and must be resampled together."""

    def test_pairs_and_noise_pairs_carry_prompt_pair_clusters(self):
        from .services.suites.counterfactual import build_counterfactual_noise_pairs

        rows = []
        for gen in (1, 2):
            for pair_id, response_id in (("cf_001", 10), ("cf_002", 20)):
                for role, group in (("original", "female"), ("counterfactual", "male")):
                    rows.append(
                        _response(
                            response_id + gen * 2 + (role == "counterfactual"),
                            f"{pair_id} {group} gen{gen}",
                            {"task": "counterfactual", "attribute": "gender", "pair_id": pair_id,
                             "role": role, "group": group, "objective_index": 1},
                            generation_index=gen,
                        )
                    )

        pairs = build_counterfactual_text_pairs(rows)
        noise = build_counterfactual_noise_pairs(rows)

        self.assertEqual(len(pairs["texts1"]), 4)
        self.assertEqual(sorted(set(pairs["clusters"])), [("1", "cf_001"), ("1", "cf_002")])
        self.assertEqual(pairs["clusters"].count(("1", "cf_001")), 2)
        # One same-group pair (gen1 vs gen2) per role per prompt pair.
        self.assertEqual(len(noise["texts1"]), 4)
        self.assertEqual(sorted(set(noise["clusters"])), [("1", "cf_001"), ("1", "cf_002")])

    def test_cluster_bootstrap_is_wider_than_naive_for_dependent_values(self):
        from .services.stats import mean_with_ci

        # 6 clusters of 5 identical values: really only 6 independent observations.
        values = [float(cluster) for cluster in range(6) for _ in range(5)]
        clusters = [cluster for cluster in range(6) for _ in range(5)]

        naive = mean_with_ci(values)
        clustered = mean_with_ci(values, clusters)

        self.assertEqual(naive["value"], clustered["value"])
        self.assertEqual(clustered["clusters"], 6)
        self.assertGreater(clustered["ciHigh"] - clustered["ciLow"], naive["ciHigh"] - naive["ciLow"])

    def test_difference_with_ci_detects_gap_only_beyond_noise(self):
        from .services.stats import difference_with_ci

        clusters = list(range(20))
        baseline = [0.9 + (index % 3) * 0.01 for index in clusters]
        same = difference_with_ci(baseline, clusters, baseline, clusters)
        lower = difference_with_ci([value - 0.1 for value in baseline], clusters, baseline, clusters)

        self.assertAlmostEqual(same["value"], 0.0)
        self.assertLessEqual(same["ciLow"], 0.0)
        self.assertGreaterEqual(same["ciHigh"], 0.0)
        self.assertAlmostEqual(lower["value"], -0.1)
        self.assertLess(lower["ciHigh"], 0.0)


class RaceGroupingTests(TestCase):
    """Race groups come from the persona's ethnicity, mapped onto LangFair's four groups."""

    def test_race_group_from_persona_and_selected_attributes_from_population(self):
        from .services.utils import extract_attribute_group, selected_protected_attributes

        black = _response(1, "x", {}, persona_attributes={"demo_ethnicity_broad": "Black / African"})
        other = _response(2, "y", {}, persona_attributes={"demo_ethnicity_broad": "Middle Eastern"})
        self.assertEqual(extract_attribute_group(black, "race"), "black")
        self.assertIsNone(extract_attribute_group(other, "race"))

        run = SimpleNamespace(prompt_population=SimpleNamespace(
            metadata={"persona": {"selected_attributes": ["gender_identity", "demo_ethnicity_broad"]}}
        ))
        self.assertEqual(selected_protected_attributes(run), ["gender", "race"])
        legacy = SimpleNamespace(prompt_population=SimpleNamespace(metadata={"persona": {"selected_attributes": ["age_bracket"]}}))
        self.assertEqual(selected_protected_attributes(legacy), ["gender"])

    def test_toxicity_group_comparison_is_keyed_by_attribute(self):
        from .services.suites.toxicity import evaluate_toxicity

        evaluator = SimpleNamespace(
            get_toxicity_scores=lambda sentences, *_: [0.9 if "bad" in s else 0.1 for s in sentences],
            evaluate=lambda **kwargs: {"metrics": {}, "data": {"score": kwargs["scores"]}},
        )
        with patch("apps.fairness.services.suites.toxicity.get_toxicity_evaluator", return_value=evaluator):
            result = evaluate_toxicity(
                responses=["Fine start. Very bad ending.", "b", "c", "d"],
                prompts=["p1", "p2", "p3", "p4"],
                options={"metrics": ["Toxic Fraction"]},
                groups={"race": ["white", "black", "asian", None]},
            )

        comparison = result["groupComparison"]["race"]
        self.assertEqual(comparison["groups"], ["asian", "black", "white"])
        self.assertEqual(comparison["comparison"], "max - min")


class AutoComputeTests(TestCase):
    """A completed run queues the same computation the results page requests by default."""

    def test_default_request_matches_results_page(self):
        from .services.auto import default_metric_request

        text = SimpleNamespace(task_type="text_generation", language="auto")
        self.assertEqual(default_metric_request(text), (None, {"language": "auto"}))

        labels = [{"value": 0, "name": "Reject"}, {"value": 1, "name": "Approve"}]
        classification = SimpleNamespace(task_type="classification", language="en", get_classification_labels=lambda: labels)
        suites, options = default_metric_request(classification)
        self.assertEqual(suites, ["classification"])
        self.assertEqual(
            options["classification"]["classificationMetricMapping"], {"positiveLabels": [1], "negativeLabels": [0]}
        )

    @patch("apps.fairness.services.runner.LangFairMetricJobRunner.request")
    def test_only_completed_runs_are_queued(self, request):
        from .services.auto import compute_metrics_for_completed_run

        evaluation = Evaluation.objects.create(name="a", use_case_description="b", task_type="text_generation")
        population = PromptPopulation.objects.create(
            evaluation=evaluation, provider="m", model_name="m", requested_prompt_count=1,
            actual_prompt_count=1, status=PromptPopulation.Status.COMPLETED,
        )
        running = EvaluationRun.objects.create(evaluation=evaluation, prompt_population=population, status="running")
        done = EvaluationRun.objects.create(evaluation=evaluation, prompt_population=population, status="completed")

        compute_metrics_for_completed_run(running.id)
        request.assert_not_called()
        compute_metrics_for_completed_run(done.id)
        request.assert_called_once()
        self.assertEqual(request.call_args.args[1:], (None, {"language": "en"}))

        with self.settings(FAIRNESS_AUTO_COMPUTE=False):
            compute_metrics_for_completed_run(done.id)
        request.assert_called_once()


class SentenceLevelScoringTests(TestCase):
    """Classifiers read only 512 tokens, so responses are scored per sentence (max)."""

    def test_split_and_max_per_response(self):
        from .services.utils import max_per_response, split_sentences

        self.assertEqual(split_sentences("One. Two?\n- three"), ["One.", "Two?", "- three"])
        self.assertEqual(split_sentences(""), [""])
        long_tail = "Polite opening. " * 300 + "Hateful last sentence."
        scores = max_per_response(["calm", long_tail], lambda sentences: [0.8 if "Hateful" in s else 0.01 for s in sentences])
        # The offending sentence sits far past 512 tokens and is still found.
        self.assertEqual(scores, [0.01, 0.8])

    def test_toxicity_uses_highest_sentence_score(self):
        from .services.suites.toxicity import evaluate_toxicity

        seen = {}
        evaluator = SimpleNamespace(
            get_toxicity_scores=lambda sentences, *_: [0.7 if "rude" in s else 0.0 for s in sentences],
            evaluate=lambda **kwargs: seen.update(kwargs) or {"metrics": {}, "data": {"score": kwargs["scores"]}},
        )
        with patch("apps.fairness.services.suites.toxicity.get_toxicity_evaluator", return_value=evaluator):
            result = evaluate_toxicity(responses=["Hello. You are rude.", "Hello."], options={"byPrompt": False})

        self.assertEqual(seen["scores"], [0.7, 0.0])
        self.assertIn("sentence", result["scoringUnit"])

    def test_split_chunks_packs_whole_sentences(self):
        from .services.utils import split_chunks

        words = lambda sentence: len(sentence.split())
        text = "One two three. Four five. Six seven eight nine. Ten."
        self.assertEqual(
            split_chunks(text, words, 5),
            ["One two three. Four five.", "Six seven eight nine. Ten."],
        )
        # A sentence longer than the budget becomes its own chunk.
        self.assertEqual(split_chunks("a b c d e f g. h.", words, 3), ["a b c d e f g.", "h."])

    def test_stereotype_chunks_follow_langfair_label_rule(self):
        from .services.evaluators import _chunked_stereotype_scores

        class Detector:
            # One "token" per word; chunks hold at most STEREOTYPE_CHUNK_TOKENS of them.
            tokenizer = staticmethod(lambda text, add_special_tokens=False: {"input_ids": text.split()})

            def __call__(self, chunks, truncation=True):
                self.seen = list(chunks)
                return [
                    {"label": "stereotype_gender", "score": 0.9} if "drivers" in chunk else {"label": "unrelated", "score": 0.99}
                    for chunk in chunks
                ]

        detector = Detector()
        metric = SimpleNamespace(categories=["gender", "race"], classifier_instance=detector)
        data = _chunked_stereotype_scores(metric, ["She could say: Women are bad drivers.", "She could say: Thanks."])

        self.assertEqual(data["stereotype_score_gender"], [0.9, 0.0])
        self.assertEqual(data["stereotype_score_race"], [0.0, 0.0])
        # Short responses stay one chunk: the fragment is judged with its context.
        self.assertIn("She could say: Women are bad drivers.", detector.seen)
