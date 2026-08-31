# pyright: reportAttributeAccessIssue=false, reportMissingImports=false
import threading

from django.db import connections
from django.utils import timezone as django_timezone

from apps.llms.clients.factory import build_llm_client
from apps.llms.models import LLMConfig
from apps.prompts.models import Prompt
from apps.responses.models import LLMResponse
from apps.prompts.services import (
    append_instruction,
    build_classification_output_instruction,
    build_recommendation_output_instruction,
)

from .models import EvaluationRun


class EvaluationRunService:
    def start(self, run, llm_config):
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(run.id, llm_config.id),
            daemon=True,
        )
        thread.start()

    def _run_in_thread(self, run_id, llm_config_id):
        try:
            self._execute(run_id, llm_config_id)
        finally:
            connections.close_all()

    def _execute(self, run_id, llm_config_id):
        run = EvaluationRun.objects.select_related("evaluation").get(pk=run_id)
        llm_config = LLMConfig.objects.get(pk=llm_config_id)

        run.status = EvaluationRun.Status.RUNNING
        run.started_at = django_timezone.now()
        run.save(update_fields=["status", "started_at"])

        try:
            client = build_llm_client(llm_config)
            prompts = Prompt.objects.filter(
                population=run.prompt_population
            ).order_by("sequence")

            for prompt in prompts:
                for generation_index in range(llm_config.generations_per_prompt):
                    prompt_text = str(prompt.text)
                    if run.evaluation.task_type == run.evaluation.TaskType.CLASSIFICATION:
                        prompt_text = append_instruction(
                            prompt_text,
                            build_classification_output_instruction(run.evaluation),
                        )
                    elif run.evaluation.task_type == run.evaluation.TaskType.RECOMMENDATION:
                        instruction = build_recommendation_output_instruction(run.evaluation)
                        if instruction:
                            prompt_text = append_instruction(prompt_text, instruction)
                    result = client.generate(prompt_text)

                    response = LLMResponse.objects.create(
                        run=run,
                        prompt=prompt,
                        generation_index=generation_index + 1,
                        response_text=result["text"],
                        input_tokens=result.get("input_tokens"),
                        output_tokens=result.get("output_tokens"),
                        latency_ms=result.get("latency_ms"),
                    )
                    run.completed_generations += 1
                    run.save(update_fields=["completed_generations"])

                run.completed_prompts += 1
                run.save(update_fields=["completed_prompts"])

            run.status = EvaluationRun.Status.COMPLETED
            run.finished_at = django_timezone.now()
            run.save(update_fields=["status", "finished_at"])

            run.evaluation.status = run.evaluation.Status.COMPLETED
            run.evaluation.save(update_fields=["status", "updated_at"])

        except Exception as exc:
            run.status = EvaluationRun.Status.FAILED
            run.error_message = str(exc)
            run.finished_at = django_timezone.now()
            run.save(update_fields=["status", "error_message", "finished_at"])

            run.evaluation.status = run.evaluation.Status.FAILED
            run.evaluation.save(update_fields=["status", "updated_at"])
