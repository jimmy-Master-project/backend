import json
import random
from pathlib import Path

from django.conf import settings
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.evaluations.models import Evaluation
from apps.llms.views import build_default_llm_config
from apps.prompts.models import Prompt, PromptPopulation
from apps.prompts.services import swap_gender_words
from apps.runs.models import EvaluationRun
from apps.runs.services import EvaluationRunService


class BoldValidationParams(serializers.Serializer):
    domain = serializers.ChoiceField(
        choices=["gender", "race", "profession", "religious_ideology", "political_ideology"],
        default="gender",
    )
    pair_count = serializers.IntegerField(min_value=1, max_value=100, default=100)
    seed = serializers.IntegerField(min_value=0, required=False, default=42)


class BoldValidationView(APIView):
    def post(self, request):
        params = BoldValidationParams(data=request.data)
        params.is_valid(raise_exception=True)
        domain = params.validated_data["domain"]
        pair_count = params.validated_data["pair_count"]
        seed = params.validated_data["seed"]

        if domain != "gender":
            return Response(
                {"error": "Only BOLD gender counterfactual validation is currently supported."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        pairs = load_gender_pairs(pair_count, seed)
        if not pairs:
            return Response({"error": "No usable BOLD gender prompts found."}, status=status.HTTP_400_BAD_REQUEST)

        evaluation = Evaluation.objects.create(
            name=f"BOLD gender validation ({len(pairs)} pairs)",
            use_case_description="BOLD benchmark counterfactual validation",
            task_type=Evaluation.TaskType.TEXT_GENERATION,
            language=Evaluation.Language.ENGLISH,
        )
        config = build_default_llm_config(evaluation, save=True)
        population = PromptPopulation.objects.create(
            evaluation=evaluation,
            provider=config.provider,
            model_name=config.model_name,
            temperature=config.temperature,
            requested_prompt_count=len(pairs) * 2,
            actual_prompt_count=len(pairs) * 2,
            status=PromptPopulation.Status.COMPLETED,
            metadata={
                "benchmark": "BOLD",
                "domain": domain,
                "pair_count": len(pairs),
                "seed": seed,
                "fairness": {"counterfactual_enabled": True, "attribute": "gender"},
            },
        )
        for sequence, prompt_data in enumerate(
            (prompt for pair in pairs for prompt in pair), start=1
        ):
            Prompt.objects.create(
                population=population,
                text=prompt_data["text"],
                category=domain,
                sequence=sequence,
                metadata=prompt_data["metadata"],
            )

        run = EvaluationRun.objects.create(
            evaluation=evaluation,
            prompt_population=population,
            status=EvaluationRun.Status.RUNNING,
            total_prompts=len(pairs) * 2,
            total_generations=len(pairs) * 2 * config.generations_per_prompt,
        )
        evaluation.status = Evaluation.Status.RUNNING
        evaluation.save(update_fields=["status", "updated_at"])
        EvaluationRunService().start(run, config)

        return Response(
            {
                "evaluation_id": evaluation.id,
                "run_id": run.id,
                "pair_count": len(pairs),
                "prompt_count": len(pairs) * 2,
                "status": run.status,
            },
            status=status.HTTP_202_ACCEPTED,
        )


def load_gender_pairs(pair_count: int, seed: int) -> list[tuple[dict, dict]]:
    path = Path(settings.BASE_DIR).parent / "bold" / "prompts" / "gender_prompt.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    grouped = {"female": [], "male": []}
    for subgroup, people in data.items():
        group = "female" if "actress" in subgroup.lower() else "male"
        for prompt_list in people.values():
            for prompt in prompt_list:
                text = str(prompt).strip()
                counterfactual = swap_gender_words(text, group, "male" if group == "female" else "female")
                if counterfactual != text:
                    grouped[group].append((text, counterfactual))

    rng = random.Random(seed)
    per_group = max(pair_count // 2, 1)
    selected = []
    for group in ("female", "male"):
        values = grouped[group][:]
        rng.shuffle(values)
        selected.extend((group, item) for item in values[:per_group])
    rng.shuffle(selected)
    pairs = []
    for index, (group, (original, counterfactual)) in enumerate(selected[:pair_count], start=1):
        opposite = "male" if group == "female" else "female"
        pairs.append(
            (
                {"text": original, "metadata": _fairness_metadata(index, group, "original")},
                {"text": counterfactual, "metadata": _fairness_metadata(index, opposite, "counterfactual")},
            )
        )
    return pairs


def _fairness_metadata(pair_id: int, group: str, role: str) -> dict:
    return {
        "source": "bold",
        "fairness": {
            "task": "counterfactual",
            "attribute": "gender",
            "group": group,
            "pair_id": f"bold_gender_{pair_id:03d}",
            "role": role,
        },
    }
