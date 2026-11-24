"""Metric and process files of a finished experiment (one target chatbot).

Written to EXPERIMENT_OUTPUT_DIR/exp<id>_<target>_<timestamp>/:
- metrics.csv     one row per group x comparison x metric (long format)
- metrics.json    the same, nested, plus the process data
- process.csv     per group: directions, questions, pairs, exclusions, cost (結果總覽表 1.1 / 2.1)
- scenarios.csv   per group: base questions per checklist scenario (1.2 / 2.2)
- conversations.jsonl   every version's text, turns and status (raw data)
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from django.conf import settings
from django.utils import timezone

from . import checklist
from .conversation import TARGET_LABELS
from .identity import comparisons
from .metrics import METRIC_KEYS, MetricModels, compute
from .models import BaseQuestion, Experiment, GroupGeneration, QuestionVersion

METRIC_LABELS = {
    "n_pairs": "有效配對數",
    "M1_CCS": "M1 反事實餘弦相似度（CCS）",
    "M2_CSB": "M2 反事實情緒偏差（CSB）",
    "M3_TIR": "M3 語氣類別不一致比例（TIR）",
    "M4_LSR": "M4 低相似度配對比例（LSR）",
    "M5_SODR": "M5 服務選項差異比例（SODR）",
    "M6_IIDR": "M6 重要資訊項目差異比例（IIDR）",
    "M7_TGap": "M7 群體平均毒性差距（TGap）",
    "M8_CROUGE_L": "M8 反事實 ROUGE-L（CROUGE-L）",
}


def output_root() -> Path:
    root = Path(getattr(settings, "EXPERIMENT_OUTPUT_DIR", Path(settings.BASE_DIR) / "experiment_outputs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _answers(experiment: Experiment, group: str) -> dict[int, dict[str, Any]]:
    """question id -> {"scenario", "answers": {version code: last answer}} (answered versions only)."""
    data: dict[int, dict[str, Any]] = {}
    versions = QuestionVersion.objects.filter(
        question__experiment=experiment, question__group=group, status=QuestionVersion.Status.ANSWERED
    ).select_related("question")
    for version in versions:
        entry = data.setdefault(version.question_id, {"scenario": version.question.scenario, "answers": {}})
        entry["answers"][version.code] = version.final_answer
    return data


def compute_results(experiment: Experiment) -> list[dict[str, Any]]:
    from .runner import GROUPS

    races = sorted({p["identity"]["race"] for p in experiment.personas})
    comps = comparisons(races)
    models = MetricModels()
    rows = []
    for group in experiment.groups:
        answers = _answers(experiment, group)
        for comp in comps:
            pairs = [
                {"a": e["answers"][comp["a"]], "b": e["answers"][comp["b"]], "scenario": e["scenario"]}
                for e in answers.values()
                if comp["a"] in e["answers"] and comp["b"] in e["answers"]
            ]
            rows.append({"group": group, "group_name": GROUPS.get(group, group), **comp, "metrics": compute(models, pairs)})
    return rows


def process_rows(experiment: Experiment, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .runner import GROUPS

    rows = []
    for generation in GroupGeneration.objects.filter(experiment=experiment).order_by("group"):
        questions = BaseQuestion.objects.filter(experiment=experiment, group=generation.group)
        versions = QuestionVersion.objects.filter(question__in=questions)
        invalid = Counter(q.invalid_reason for q in questions if not q.valid)
        rows.append({
            "group": generation.group,
            "group_name": GROUPS.get(generation.group, generation.group),
            "directions": len(generation.directions) if generation.group.startswith("A") else "",
            "scenarios_covered": len({q.scenario for q in questions if q.valid and q.scenario}),
            "questions_generated": questions.filter(valid=True).count(),
            "questions_invalid": questions.filter(valid=False).count(),
            "invalid_reasons": dict(invalid),
            "versions": versions.count(),
            "versions_answered": versions.filter(status=QuestionVersion.Status.ANSWERED).count(),
            "excluded_follow_up_after_5_turns": versions.filter(status=QuestionVersion.Status.FOLLOW_UP_EXCLUDED).count(),
            "excluded_error": versions.filter(status=QuestionVersion.Status.ERROR).count(),
            "valid_pairs_total": sum(r["metrics"]["n_pairs"] for r in results if r["group"] == generation.group),
            "llm_calls": generation.stats.get("llm_calls"),
            "generation_seconds": generation.stats.get("seconds"),
        })
    return rows


def scenario_rows(experiment: Experiment) -> list[dict[str, Any]]:
    rows = []
    for key, spec in [*checklist.SCENARIOS.items(), ("", {"label": "未分類"})]:
        row = {"scenario": key or "unclassified", "label": spec["label"]}
        for group in experiment.groups:
            row[group] = BaseQuestion.objects.filter(experiment=experiment, group=group, valid=True, scenario=key).count()
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def export_results(experiment: Experiment) -> Path:
    results = compute_results(experiment)
    process = process_rows(experiment, results)
    scenarios = scenario_rows(experiment)
    stamp = timezone.localtime().strftime("%Y%m%d-%H%M%S")
    directory = output_root() / f"exp{experiment.id}_{experiment.target_provider}_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)

    long_rows = []
    for row in results:
        for key in METRIC_KEYS:
            long_rows.append({
                "experiment_id": experiment.id,
                "target": experiment.target_provider,
                "target_label": TARGET_LABELS.get(experiment.target_provider, experiment.target_provider),
                "group": row["group"],
                "group_name": row["group_name"],
                "comparison_type": row["type"],
                "fixed_condition": row["fixed"],
                "comparison": row["comparison"],
                "version_a": row["a"],
                "version_b": row["b"],
                "metric": key,
                "metric_label": METRIC_LABELS[key],
                "value": row["metrics"].get(key),
                "n_pairs": row["metrics"]["n_pairs"],
            })
    _write_csv(directory / "metrics.csv", long_rows, list(long_rows[0]) if long_rows else ["experiment_id"])
    _write_csv(directory / "process.csv", process, list(process[0]) if process else ["group"])
    _write_csv(directory / "scenarios.csv", scenarios, ["scenario", "label", *experiment.groups])

    payload = {
        "experiment_id": experiment.id,
        "target": experiment.target_provider,
        "target_label": TARGET_LABELS.get(experiment.target_provider, experiment.target_provider),
        "evaluation_id": experiment.evaluation_id,
        "questions_per_group": experiment.questions_per_group,
        "groups": experiment.groups,
        "races": sorted({p["identity"]["race"] for p in experiment.personas}),
        "config": experiment.config,
        "exported_at": timezone.now().isoformat(),
        "process": process,
        "scenarios": scenarios,
        "results": results,
    }
    (directory / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (directory / "conversations.jsonl").open("w", encoding="utf-8") as handle:
        versions = QuestionVersion.objects.filter(question__experiment=experiment).select_related("question")
        for v in versions.order_by("question__group", "question__index", "id"):
            handle.write(json.dumps({
                "group": v.question.group, "index": v.question.index, "persona_id": v.question.persona_id,
                "scenario": v.question.scenario, "version": v.code, "text": v.text, "status": v.status,
                "turns": v.turns, "final_answer": v.final_answer, "error": v.error,
            }, ensure_ascii=False) + "\n")
    return directory
