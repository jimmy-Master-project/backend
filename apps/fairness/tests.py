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
