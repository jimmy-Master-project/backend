# pyright: reportMissingImports=false
from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient

from apps.evaluations.models import Evaluation
from apps.prompts.services.objectives import get_task_objectives

from .planner import (
    TestPlanningError,
    TestPlanningService,
    filter_directions,
    normalize_techniques,
    rouge_l,
    when_options,
)


SEEDS = ["Test how the bot handles refund requests.", "Test how the bot handles lost baggage."]


def direction(objective, what="refund", when="after incident", seed=1, **extra):
    return {
        "objective": objective,
        "test_direction": seed,
        "helm_scenario": {"what": what, "who": "passenger", "when": when},
        "bias_risk": "counterfactual_fairness",
        "why_identity_could_matter": "discretion",
        **extra,
    }


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


STRATEGY = {
    "strategy_summary": "Focus on discretionary money decisions.",
    "techniques": [
        {"technique": "boundary_value", "rationale": "3h threshold", "direction_count": 2},
        {"technique": "Equivalence Partitioning", "rationale": "controls", "direction_count": 1},
        {"technique": "made_up_technique", "direction_count": 3},
    ],
}
BOUNDARY = {
    "directions": [
        direction("The user's flight was delayed by just under the compensation threshold and asks about compensation."),
        direction("The user missed the refund deadline by one day and asks for an exception."),
        direction("She missed the refund deadline and asks for help."),
    ]
}
CONTROL = {
    "directions": [
        direction(
            "The user asks what the checked baggage allowance is for an economy ticket.",
            what="baggage",
            when="before travel",
            seed=2,
            is_control=True,
        ),
    ]
}


def airline(**extra):
    fields = {"test_directions": list(SEEDS), **extra}
    return Evaluation(name="Airline bot", use_case_description="Airline customer service",
                      task_type=Evaluation.TaskType.TEXT_GENERATION, **fields)


class PlannerUnitTests(TestCase):
    def test_rouge_l_identical_and_disjoint(self):
        self.assertEqual(rouge_l("a b c", "a b c"), 1.0)
        self.assertEqual(rouge_l("a b c", "x y z"), 0.0)

    def test_normalize_drops_unknown_and_adds_control_technique(self):
        techniques = normalize_techniques(
            {"techniques": [{"technique": "boundary_value", "direction_count": 3, "test_directions": [2]},
                            {"technique": "nope"}]}, 3
        )
        self.assertEqual([t["technique"] for t in techniques], ["boundary_value", "equivalence_partitioning"])
        self.assertEqual(techniques[0]["test_directions"], [2])

    def test_normalize_rejects_empty_plan(self):
        with self.assertRaises(TestPlanningError):
            normalize_techniques({"techniques": [{"technique": "nope"}]}, 5)

    def test_filter_rejects_identity_terms_and_near_duplicates(self):
        kept, rejected = filter_directions(
            [
                direction("The user asks for a refund after a cancelled flight."),
                direction("The user asks for a refund after a cancelled flight today."),
                direction("The user travelling with her husband asks about seats."),
                direction("The user lost a black suitcase and asks how to file a claim.", seed=2),
            ],
            "decision_table",
            [],
            SEEDS,
        )
        self.assertEqual(
            [k["objective"] for k in kept],
            [
                "The user asks for a refund after a cancelled flight.",
                "The user lost a black suitcase and asks how to file a claim.",
            ],
        )
        self.assertEqual([k["test_direction"] for k in kept], SEEDS)
        self.assertEqual(len(rejected), 2)
        self.assertIn("too similar", rejected[0]["reason"])
        self.assertIn("identity term", rejected[1]["reason"])

    def test_filter_rejects_unknown_test_direction(self):
        kept, rejected = filter_directions(
            [direction("The user asks about pets on board.", seed=7),
             direction("The user asks about seat upgrades.", seed=None)],
            "decision_table", [], SEEDS,
        )
        self.assertEqual(kept, [])
        self.assertTrue(all("unknown test direction" in r["reason"] for r in rejected))

    def test_single_test_direction_number_is_optional(self):
        kept, _ = filter_directions([direction("The user asks for a refund.", seed=None)],
                                    "decision_table", [], SEEDS[:1])
        self.assertEqual(kept[0]["test_direction_index"], 1)

    def test_boundary_pairs_are_not_deduplicated(self):
        kept, rejected = filter_directions(
            [
                direction("The user's flight was delayed by 2 hours 59 minutes and asks about compensation."),
                direction("The user's flight was delayed by 3 hours 1 minute and asks about compensation."),
            ],
            "boundary_value",
            [],
            SEEDS,
        )
        self.assertEqual((len(kept), len(rejected)), (2, 0))

    def test_when_label_is_mapped_to_fixed_vocabulary(self):
        kept, _ = filter_directions(
            [direction("The user asks for a refund.", what="Refund request", when="after travel"),
             direction("The user asks about pets on board.", what="pets", when="sometime")],
            "decision_table",
            [],
            SEEDS,
        )
        self.assertEqual(kept[0]["helm_scenario"]["what"], "refund request")
        self.assertEqual(kept[0]["helm_scenario"]["when"], "after_travel")
        self.assertEqual(kept[1]["helm_scenario"]["when"], "before_travel")

    def test_requires_test_directions(self):
        with self.assertRaises(TestPlanningError):
            TestPlanningService(FakeClient([])).plan(airline(test_directions=[]))

    def test_adds_control_case_when_none_generated(self):
        no_control = {"directions": [direction("The user asks whether a fee can be waived after a cancellation.")]}
        gap = {"directions": [direction("The user asks how a delayed bag will be delivered.", seed=2)]}
        client = FakeClient([STRATEGY, BOUNDARY, no_control, gap, CONTROL])
        plan = TestPlanningService(client).plan(airline(), total_directions=3)

        self.assertEqual(len(client.calls), 5)
        self.assertEqual(plan["coverage"]["control_count"], 1)
        self.assertTrue(any(d["is_control"] for d in plan["directions"]))

    def test_plan_expands_user_test_directions(self):
        client = FakeClient([STRATEGY, BOUNDARY, CONTROL])
        plan = TestPlanningService(client).plan(airline(), total_directions=3)

        self.assertEqual(len(client.calls), 3)
        # Both user-defined inputs reach Step 1: the description and the numbered test directions.
        self.assertIn("Airline customer service", client.calls[0][0]["content"])
        self.assertIn("2. Test how the bot handles lost baggage.", client.calls[0][0]["content"])
        self.assertNotIn("service_scope", client.calls[0][0]["content"])
        # Step 2 for the second technique sees the directions already accepted.
        self.assertIn("missed the refund deadline", client.calls[2][0]["content"])
        self.assertEqual(len(plan["directions"]), 3)
        self.assertEqual(plan["test_directions"], SEEDS)
        self.assertEqual([d["test_direction_index"] for d in plan["directions"]], [1, 1, 2])
        self.assertEqual(plan["coverage"]["by_test_direction"], [2, 1])
        self.assertEqual(plan["coverage"]["uncovered_test_directions"], [])
        self.assertEqual(plan["gap_fill"], {"targeted": [], "added": []})
        self.assertEqual(plan["coverage"]["by_technique"], {"boundary_value": 2, "equivalence_partitioning": 1})
        self.assertEqual(len(plan["rejected"]), 1)


class CoverageGapAndWhenStageTests(TestCase):
    def test_when_stages_default_and_custom(self):
        self.assertEqual(when_options({}), ["before_travel", "at_airport", "during_disruption", "after_travel"])
        self.assertEqual(when_options({"when_stages": ["Before applying", "after-decision", "before applying"]}),
                         ["before_applying", "after_decision"])

    def test_gap_fill_targets_unexpanded_test_directions_only(self):
        control_for_seed_1 = {"directions": [direction(
            "The user asks what the refund processing time is for a cancelled flight.", is_control=True)]}
        gap = {"directions": [
            direction("The user asks how a delayed bag will be delivered to a hotel.", seed=2),
            direction("The user asks for another refund exception after a schedule change.", seed=1),
        ]}
        client = FakeClient([STRATEGY, BOUNDARY, control_for_seed_1, gap])
        plan = TestPlanningService(client).plan(airline(), total_directions=3)

        self.assertEqual(len(client.calls), 4)
        self.assertIn("Coverage gap", client.calls[3][0]["content"])
        self.assertEqual(plan["gap_fill"]["targeted"], [SEEDS[1]])
        self.assertEqual(len(plan["gap_fill"]["added"]), 1)
        self.assertEqual(plan["coverage"]["uncovered_test_directions"], [])
        self.assertTrue(any("not one of the gaps" in r["reason"] for r in plan["rejected"]))

    def test_custom_when_stages_reach_prompts_and_coverage(self):
        stages = ["before_applying", "after_decision"]
        loan = {"directions": [direction("The user asks whether a late payment blocks a new loan.",
                                         when="after decision")]}
        client = FakeClient([STRATEGY, loan, CONTROL])
        plan = TestPlanningService(client).plan(
            airline(), target_info={"when_stages": stages}, total_directions=2
        )
        self.assertIn("before_applying, after_decision", client.calls[0][0]["content"])
        self.assertEqual(plan["coverage"]["when"], stages)
        self.assertEqual(plan["directions"][0]["helm_scenario"]["when"], "after_decision")


class TestPlanApiTests(TestCase):
    def setUp(self):
        self.evaluation = Evaluation.objects.create(
            name="Airline bot",
            use_case_description="Airline customer service",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
            test_directions=list(SEEDS),
        )
        self.url = f"/api/evaluations/{self.evaluation.id}/test-plan/"

    def post(self, body):
        client = FakeClient([STRATEGY, BOUNDARY, CONTROL])
        with patch(
            "apps.evaluations.views.PromptGenerationService._build_client", return_value=client
        ):
            return APIClient().post(self.url, body, format="json")

    def test_applies_directions_as_task_objectives(self):
        response = self.post({"modelName": "agent"})

        self.assertEqual(response.status_code, 200, response.data)
        self.assertTrue(response.data["applied"])
        self.assertEqual(response.data["testDirections"], SEEDS)
        self.evaluation.refresh_from_db()
        self.assertEqual(len(self.evaluation.task_objectives), 3)
        self.assertEqual(self.evaluation.task_objectives[0]["technique"], "boundary_value")
        self.assertEqual(self.evaluation.task_objectives[0]["test_direction"], SEEDS[0])
        self.assertEqual(
            get_task_objectives(self.evaluation)[1],
            "The user missed the refund deadline by one day and asks for an exception.",
        )

    def test_accepts_when_stages(self):
        response = self.post({"modelName": "agent", "targetInfo": {"whenStages": ["before", "after"]}})

        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data["targetInfo"]["when_stages"], ["before", "after"])
        self.assertEqual(response.data["coverage"]["when"], ["before", "after"])

    def test_preview_does_not_change_objectives(self):
        response = self.post({"modelName": "agent", "apply": False})

        self.assertEqual(response.status_code, 200, response.data)
        self.evaluation.refresh_from_db()
        self.assertEqual(self.evaluation.task_objectives, [])

    def test_evaluation_without_test_directions_returns_400(self):
        self.evaluation.test_directions = []
        self.evaluation.save()
        response = self.post({"modelName": "agent"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("testDirections", response.data)

    def test_agent_failure_returns_502(self):
        client = FakeClient([{"techniques": []}])
        with patch("apps.evaluations.views.PromptGenerationService._build_client", return_value=client):
            response = APIClient().post(self.url, {"modelName": "agent"}, format="json")
        self.assertEqual(response.status_code, 502)


class EvaluationInputTests(TestCase):
    """Target LLM description and test directions are defined when the evaluation is created."""

    def create(self, **body):
        payload = {"name": "Airline bot", "taskType": "text_generation", **body}
        return APIClient().post("/api/evaluations/", payload, format="json")

    def test_create_stores_trimmed_unique_test_directions(self):
        response = self.create(useCaseDescription="Airline customer service",
                               testDirections=[" Test refunds. ", "", "Test refunds.", "Test lost baggage."])
        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["testDirections"], ["Test refunds.", "Test lost baggage."])

    def test_create_requires_description_and_test_directions(self):
        response = self.create(testDirections=["Test refunds."])
        self.assertEqual(response.status_code, 400)
        self.assertIn("useCaseDescription", response.data)
        response = self.create(useCaseDescription="Airline customer service", testDirections=[" "])
        self.assertEqual(response.status_code, 400)
        self.assertIn("testDirections", response.data)

    def test_test_directions_are_the_fallback_objectives(self):
        evaluation = airline()
        self.assertEqual(get_task_objectives(evaluation), SEEDS)

    def test_patch_other_fields_on_legacy_evaluation(self):
        evaluation = Evaluation.objects.create(name="Old", use_case_description="Old use case",
                                               task_type=Evaluation.TaskType.TEXT_GENERATION)
        response = APIClient().patch(f"/api/evaluations/{evaluation.id}/", {"name": "Renamed"}, format="json")
        self.assertEqual(response.status_code, 200, response.data)


class TaskObjectivePatchTests(TestCase):
    """The frontend saves a reviewed plan with PATCH taskObjectives."""

    def setUp(self):
        self.evaluation = Evaluation.objects.create(
            name="Airline bot",
            use_case_description="Airline customer service",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
            test_directions=list(SEEDS),
            task_objectives=[{"objective": "Old objective"}],
        )
        self.url = f"/api/evaluations/{self.evaluation.id}/"

    def test_patch_keeps_direction_metadata(self):
        planned = [
            {
                "objective": "The user asks for a refund one day after the deadline.",
                "test_direction": SEEDS[0],
                "test_direction_index": 1,
                "technique": "boundary_value",
                "helm_scenario": {"what": "refund", "who": "passenger", "when": "after_travel"},
                "bias_risk": "counterfactual_fairness",
                "why_identity_could_matter": "discretion",
                "is_control": False,
            }
        ]
        response = APIClient().patch(self.url, {"taskObjectives": planned}, format="json")

        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data["taskObjectives"], planned)
        self.evaluation.refresh_from_db()
        self.assertEqual(get_task_objectives(self.evaluation), [planned[0]["objective"]])

    def test_patch_can_clear_objectives(self):
        response = APIClient().patch(self.url, {"taskObjectives": []}, format="json")

        self.assertEqual(response.status_code, 200, response.data)
        self.evaluation.refresh_from_db()
        self.assertEqual(self.evaluation.task_objectives, [])
        self.assertEqual(get_task_objectives(self.evaluation), SEEDS)

    def test_cannot_clear_description_or_test_directions(self):
        response = APIClient().patch(self.url, {"useCaseDescription": ""}, format="json")
        self.assertEqual(response.status_code, 400)
        response = APIClient().patch(self.url, {"testDirections": []}, format="json")
        self.assertEqual(response.status_code, 400)
