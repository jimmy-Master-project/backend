"""Experiment API (camelCase JSON, like the rest of the backend).

GET  /api/experiments/                     list
POST /api/experiments/                     create + start {useCaseDescription, testDirections[], targetProvider,
                                           targetBaseUrl?, targetModel?, questionsPerGroup?, groups?, name?}
GET  /api/experiments/{id}/                status + process counts
POST /api/experiments/{id}/start/          resume (failed conversations are retried)
POST /api/experiments/{id}/stop/           stop after the current step
GET  /api/experiments/{id}/files/{name}    download an exported file
GET  /api/experiments/targets/             selectable targets and groups
"""
from __future__ import annotations

from pathlib import Path

from django.db.models import Count, Q
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.evaluations.models import Evaluation

from . import runner
from .conversation import BOT_ENDPOINTS, TARGET_LABELS
from .models import Experiment, QuestionVersion

EXPORT_FILES = ("metrics.csv", "metrics.json", "process.csv", "scenarios.csv", "conversations.jsonl")


class CreateSerializer(serializers.Serializer):
    useCaseDescription = serializers.CharField()
    testDirections = serializers.ListField(child=serializers.CharField(allow_blank=True), allow_empty=False)
    targetProvider = serializers.ChoiceField(choices=[*BOT_ENDPOINTS, "openai"])
    targetBaseUrl = serializers.CharField(required=False, allow_blank=True, default="")
    targetModel = serializers.CharField(required=False, allow_blank=True, default="")
    questionsPerGroup = serializers.IntegerField(required=False, min_value=1, max_value=500, default=50)
    groups = serializers.ListField(child=serializers.ChoiceField(choices=list(runner.GROUPS)), required=False)
    name = serializers.CharField(required=False, allow_blank=True, default="")


def serialize(experiment: Experiment) -> dict:
    counts = QuestionVersion.objects.filter(question__experiment=experiment).aggregate(
        total=Count("id"),
        answered=Count("id", filter=Q(status=QuestionVersion.Status.ANSWERED)),
        followUpExcluded=Count("id", filter=Q(status=QuestionVersion.Status.FOLLOW_UP_EXCLUDED)),
        errors=Count("id", filter=Q(status=QuestionVersion.Status.ERROR)),
    )
    groups = {
        g.group: {"status": g.status, "directions": len(g.directions), **{k: g.stats.get(k) for k in ("generated", "invalid", "llm_calls", "seconds")}}
        for g in experiment.generations.all()
    }
    output = Path(experiment.output_dir) if experiment.output_dir else None
    return {
        "id": experiment.id,
        "name": experiment.name,
        "useCaseDescription": experiment.evaluation.use_case_description,
        "testDirections": experiment.evaluation.test_directions,
        "targetProvider": experiment.target_provider,
        "targetLabel": TARGET_LABELS.get(experiment.target_provider, experiment.target_provider),
        "questionsPerGroup": experiment.questions_per_group,
        "groups": experiment.groups,
        "groupStatus": groups,
        "personas": len(experiment.personas),
        "races": sorted({p["identity"]["race"] for p in experiment.personas}),
        "status": experiment.status,
        "running": runner.is_running(experiment.id),
        "errorMessage": experiment.error_message,
        "progress": experiment.progress,
        "versions": counts,
        "files": [name for name in EXPORT_FILES if output and (output / name).exists()],
        "outputDir": experiment.output_dir,
        "createdAt": experiment.created_at,
        "startedAt": experiment.started_at,
        "finishedAt": experiment.finished_at,
    }


class TargetsView(APIView):
    def get(self, request):
        return Response({
            "targets": [{"value": k, "label": TARGET_LABELS[k]} for k in [*BOT_ENDPOINTS, "openai"]],
            "groups": [{"value": k, "label": v} for k, v in runner.GROUPS.items()],
        })


class ExperimentListView(APIView):
    def get(self, request):
        runner.reap_orphans()
        experiments = Experiment.objects.select_related("evaluation").prefetch_related("generations")
        return Response([serialize(e) for e in experiments])

    def post(self, request):
        params = CreateSerializer(data=request.data)
        params.is_valid(raise_exception=True)
        data = params.validated_data
        directions = [" ".join(d.split()) for d in data["testDirections"] if d.strip()]
        if not directions:
            return Response({"error": "At least one test direction is required."}, status=status.HTTP_400_BAD_REQUEST)
        # The generation code reads the description and test directions from an Evaluation.
        evaluation = Evaluation.objects.create(
            name=data["name"] or f"Batch experiment ({TARGET_LABELS.get(data['targetProvider'], data['targetProvider'])})",
            use_case_description=data["useCaseDescription"].strip(),
            task_type=Evaluation.TaskType.TEXT_GENERATION,
            test_directions=directions,
        )
        experiment = Experiment.objects.create(
            evaluation=evaluation,
            name=data["name"],
            target_provider=data["targetProvider"],
            target_base_url=data["targetBaseUrl"],
            target_model=data["targetModel"],
            questions_per_group=data["questionsPerGroup"],
            groups=data.get("groups") or list(runner.GROUPS),
        )
        runner.start(experiment)
        return Response(serialize(experiment), status=status.HTTP_201_CREATED)


class ExperimentDetailView(APIView):
    def get(self, request, experiment_id):
        runner.reap_orphans()
        return Response(serialize(get_object_or_404(Experiment, pk=experiment_id)))


class ExperimentStartView(APIView):
    def post(self, request, experiment_id):
        experiment = get_object_or_404(Experiment, pk=experiment_id)
        if runner.is_running(experiment.id):
            return Response({"error": "The experiment is already running."}, status=status.HTTP_409_CONFLICT)
        runner.ExperimentRunner(experiment.id).retry_errors()
        runner.start(experiment)
        return Response(serialize(experiment))


class ExperimentStopView(APIView):
    def post(self, request, experiment_id):
        experiment = get_object_or_404(Experiment, pk=experiment_id)
        runner.request_stop(experiment.id)
        return Response(serialize(experiment))


class ExperimentFileView(APIView):
    def get(self, request, experiment_id, name):
        experiment = get_object_or_404(Experiment, pk=experiment_id)
        if name not in EXPORT_FILES or not experiment.output_dir:
            raise Http404
        path = Path(experiment.output_dir) / name
        if not path.exists():
            raise Http404
        return FileResponse(path.open("rb"), as_attachment=True, filename=f"exp{experiment.id}_{experiment.target_provider}_{name}")
