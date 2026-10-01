# pyright: reportAttributeAccessIssue=false, reportMissingImports=false
from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.test import APIClient

from apps.evaluations.models import Evaluation
from apps.prompts.models import Prompt, PromptPopulation
from apps.runs.models import EvaluationRun
from apps.responses.models import LLMResponse


class RunResponseDeleteTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.evaluation = Evaluation.objects.create(
            name="Response deletion",
            use_case_description="Test response record deletion.",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
        )
        self.population = PromptPopulation.objects.create(
            evaluation=self.evaluation,
            provider="manual",
            model_name="manual",
            requested_prompt_count=2,
            actual_prompt_count=2,
            status=PromptPopulation.Status.COMPLETED,
        )
        self.run = EvaluationRun.objects.create(
            evaluation=self.evaluation,
            prompt_population=self.population,
            status=EvaluationRun.Status.COMPLETED,
            total_prompts=2,
            completed_prompts=2,
            total_generations=2,
            completed_generations=2,
        )
        self.responses = []
        for sequence in (1, 2):
            prompt = Prompt.objects.create(
                population=self.population,
                text=f"Prompt {sequence}",
                sequence=sequence,
            )
            self.responses.append(
                LLMResponse.objects.create(
                    run=self.run,
                    prompt=prompt,
                    generation_index=1,
                    response_text=f"Output {sequence}",
                )
            )

    def test_delete_removes_one_output_and_updates_run_counts(self):
        response = self.client.delete(
            f"/api/runs/{self.run.id}/responses/{self.responses[0].id}/"
        )

        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, EvaluationRun.Status.COMPLETED)
        self.assertEqual(self.run.completed_generations, 1)
        self.assertEqual(self.run.completed_prompts, 1)
        records = self.client.get(f"/api/runs/{self.run.id}/responses/")
        self.assertEqual(records.data["count"], 1)
        self.assertTrue(Evaluation.objects.filter(pk=self.evaluation.pk).exists())
        self.assertTrue(EvaluationRun.objects.filter(pk=self.run.pk).exists())
        self.assertTrue(Prompt.objects.filter(pk=self.responses[0].prompt_id).exists())

    def test_cannot_delete_response_while_run_is_active(self):
        self.run.status = EvaluationRun.Status.RUNNING
        self.run.save(update_fields=["status"])

        response = self.client.delete(
            f"/api/runs/{self.run.id}/responses/{self.responses[0].id}/"
        )

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        self.assertTrue(LLMResponse.objects.filter(pk=self.responses[0].id).exists())

    def test_response_must_belong_to_the_selected_run(self):
        other_run = EvaluationRun.objects.create(
            evaluation=self.evaluation,
            prompt_population=self.population,
            status=EvaluationRun.Status.COMPLETED,
        )

        response = self.client.delete(
            f"/api/runs/{other_run.id}/responses/{self.responses[0].id}/"
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertTrue(LLMResponse.objects.filter(pk=self.responses[0].id).exists())


@override_settings(EXTERNAL_RESPONSE_API_KEY="bot-key")
class ExternalResponsePostTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.evaluation = Evaluation.objects.create(
            name="External bot",
            use_case_description="Responses posted by an external bot.",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
        )
        self.population = PromptPopulation.objects.create(
            evaluation=self.evaluation,
            provider="manual",
            model_name="manual",
            requested_prompt_count=2,
            actual_prompt_count=2,
            status=PromptPopulation.Status.COMPLETED,
        )
        self.prompts = [
            Prompt.objects.create(
                population=self.population, text=f"Prompt {sequence}", sequence=sequence
            )
            for sequence in (1, 2)
        ]
        self.run = EvaluationRun.objects.create(
            evaluation=self.evaluation,
            prompt_population=self.population,
            status=EvaluationRun.Status.RUNNING,
            external=True,
            total_prompts=2,
            total_generations=2,
        )
        self.url = f"/api/runs/{self.run.id}/responses/"

    def post(self, items, key="bot-key"):
        return self.client.post(
            self.url, {"responses": items}, format="json", HTTP_X_API_KEY=key
        )

    def item(self, prompt, text="answer", generation_index=1):
        return {"promptId": prompt.id, "response": text, "generationIndex": generation_index}

    def test_rejects_invalid_api_key(self):
        response = self.post([self.item(self.prompts[0])], key="wrong")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_stores_responses_and_completes_run(self):
        response = self.post([self.item(self.prompts[0])])

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["stored"], 1)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, EvaluationRun.Status.RUNNING)
        self.assertEqual(self.run.completed_generations, 1)

        response = self.post([self.item(self.prompts[1])])

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, EvaluationRun.Status.COMPLETED)
        self.assertEqual(self.run.completed_prompts, 2)
        self.evaluation.refresh_from_db()
        self.assertEqual(self.evaluation.status, Evaluation.Status.COMPLETED)

    def test_duplicates_are_upserted_and_do_not_complete_run_early(self):
        response = self.post(
            [self.item(self.prompts[0], "first"), self.item(self.prompts[0], "second")]
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        response = self.post([self.item(self.prompts[0], "third")])

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["updated"], 1)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, EvaluationRun.Status.RUNNING)
        self.assertEqual(self.run.completed_generations, 1)
        self.assertEqual(LLMResponse.objects.get(run=self.run).response_text, "third")

    def test_rejects_out_of_range_generation_index(self):
        response = self.post([self.item(self.prompts[0], generation_index=2)])

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(LLMResponse.objects.filter(run=self.run).exists())

    def test_rejects_prompt_from_another_population(self):
        response = self.post([{"promptId": 999999, "response": "x"}])
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_rejects_posts_to_completed_run(self):
        self.run.status = EvaluationRun.Status.COMPLETED
        self.run.save(update_fields=["status"])

        response = self.post([self.item(self.prompts[0])])

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
