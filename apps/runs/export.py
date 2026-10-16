# pyright: reportAttributeAccessIssue=false
"""CSV export of everything a run produced, split into one table per category.

Every table is a list of rows (dicts); ``build_table`` returns the rows of one
table and ``build_zip`` packs every table as its own CSV file. CSVs are written
with a UTF-8 BOM so Excel opens Chinese text correctly.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from collections.abc import Callable, Mapping
from typing import Any

from apps.fairness.models import LangFairMetricResult
from apps.fairness.services.utils import extract_attribute_group

Row = dict[str, Any]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return _json(value)
    return value


def _iso(value: Any) -> str:
    return value.isoformat() if value else ""


def latest_metric_result(run: Any) -> LangFairMetricResult | None:
    """The most recently computed completed LangFair result of the run."""
    return (
        LangFairMetricResult.objects.filter(
            run=run, status=LangFairMetricResult.Status.COMPLETED
        )
        .order_by("-updated_at", "-id")
        .first()
    )


def _metric_payload(run: Any) -> dict[str, Any]:
    row = latest_metric_result(run)
    result = getattr(row, "result", None)
    return result if isinstance(result, dict) else {}


def _fairness(prompt: Any) -> Mapping[str, Any]:
    fairness = (prompt.metadata or {}).get("fairness")
    return fairness if isinstance(fairness, Mapping) else {}


def _source_persona_id(prompt: Any) -> str:
    metadata = prompt.metadata or {}
    return str(_fairness(prompt).get("source_persona_id") or metadata.get("persona_id") or "")


# ---- tables --------------------------------------------------------------
def run_info_rows(run: Any) -> list[Row]:
    evaluation = run.evaluation
    population = run.prompt_population
    snapshot = run.llm_snapshot or {}
    metric_row = latest_metric_result(run)
    persona_config = (population.metadata or {}).get("persona") or {}
    rows: list[tuple[str, str, Any]] = [
        ("evaluation", "id", evaluation.id),
        ("evaluation", "name", evaluation.name),
        ("evaluation", "use_case_description", evaluation.use_case_description),
        ("evaluation", "task_type", evaluation.task_type),
        ("evaluation", "language", evaluation.language),
        ("evaluation", "task_objectives", evaluation.task_objectives),
        ("evaluation", "classification_labels", evaluation.get_classification_labels()),
        ("evaluation", "recommendation_items", evaluation.get_recommendation_items()),
        ("run", "id", run.id),
        ("run", "label", run.label),
        ("run", "status", run.status),
        ("run", "external", run.external),
        ("run", "total_prompts", run.total_prompts),
        ("run", "completed_prompts", run.completed_prompts),
        ("run", "total_generations", run.total_generations),
        ("run", "completed_generations", run.completed_generations),
        ("run", "failed_generations", run.responses.filter(error__isnull=False).count()),
        ("run", "started_at", _iso(run.started_at)),
        ("run", "finished_at", _iso(run.finished_at)),
        ("run", "error_message", run.error_message),
    ]
    rows += [("target_llm", key, value) for key, value in snapshot.items()]
    rows += [
        ("prompt_population", "id", population.id),
        ("prompt_population", "name", population.name),
        ("prompt_population", "generator_provider", population.provider),
        ("prompt_population", "generator_model", population.model_name),
        ("prompt_population", "generator_temperature", population.temperature),
        ("prompt_population", "requested_prompt_count", population.requested_prompt_count),
        ("prompt_population", "actual_prompt_count", population.actual_prompt_count),
        ("prompt_population", "additional_instruction", population.additional_instruction),
        ("prompt_population", "fairness_labels", persona_config.get("selected_attributes")),
        ("prompt_population", "sampling_seed", persona_config.get("sampling_seed")),
        ("prompt_population", "created_at", _iso(population.created_at)),
        ("fairness_metrics", "computed_at", _iso(getattr(metric_row, "updated_at", None))),
    ]
    return [{"section": section, "field": field, "value": value} for section, field, value in rows]


def prompt_rows(run: Any) -> list[Row]:
    rows = []
    for prompt in run.prompt_population.prompts.all():
        fairness = _fairness(prompt)
        rows.append({
            "prompt_id": prompt.id,
            "sequence": prompt.sequence,
            "category": prompt.category,
            "task_objective": (prompt.metadata or {}).get("task_objective", ""),
            "pair_id": fairness.get("pair_id", ""),
            "pair_role": fairness.get("role", ""),
            "pair_group": fairness.get("group", ""),
            "persona_id": (prompt.metadata or {}).get("persona_id", ""),
            "source_persona_id": _source_persona_id(prompt),
            "prompt_text": prompt.text,
        })
    return rows


def response_rows(run: Any) -> list[Row]:
    rows = []
    for response in run.responses.select_related("prompt"):
        prompt = response.prompt
        fairness = _fairness(prompt)
        rows.append({
            "response_id": response.id,
            "prompt_id": prompt.id,
            "prompt_sequence": prompt.sequence,
            "generation_index": response.generation_index,
            "category": prompt.category,
            "pair_id": fairness.get("pair_id", ""),
            "pair_role": fairness.get("role", ""),
            "gender_group": extract_attribute_group(response, "gender") or "",
            "race_group": extract_attribute_group(response, "race") or "",
            "source_persona_id": _source_persona_id(prompt),
            "prompt_text": prompt.text,
            "response_text": response.response_text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "error": response.error or "",
            "created_at": _iso(response.created_at),
        })
    return rows


def persona_rows(run: Any) -> list[Row]:
    """One row per source persona, with its attributes as columns."""
    personas: dict[str, Row] = {}
    for prompt in run.prompt_population.prompts.all():
        metadata = prompt.metadata or {}
        persona_id = _source_persona_id(prompt)
        if not persona_id:
            continue
        entry = personas.setdefault(persona_id, {"persona_id": persona_id, "prompt_count": 0})
        entry["prompt_count"] += 1
        attributes = metadata.get("persona_attributes")
        if not isinstance(attributes, Mapping):
            continue
        # Counterfactual twins carry the swapped gender; the original's attributes win.
        if _fairness(prompt).get("role") != "counterfactual":
            entry.update(attributes)
        else:
            entry.update({key: value for key, value in attributes.items() if key not in entry})
    return list(personas.values())


def _ci_fields(ci: Any) -> Row:
    ci = ci if isinstance(ci, Mapping) else {}
    return {
        "ci_low": ci.get("ciLow"),
        "ci_high": ci.get("ciHigh"),
        "n": ci.get("n"),
        "clusters": ci.get("clusters"),
    }


def _metric_block_rows(suite: str, section: str, block: Any) -> list[Row]:
    if not isinstance(block, Mapping):
        return []
    metrics = block.get("metrics")
    if not isinstance(metrics, Mapping):
        return []
    intervals = block.get("confidenceIntervals") or {}
    rows = []
    for name, value in metrics.items():
        ci = intervals.get(name) if isinstance(intervals, Mapping) else None
        if isinstance(value, Mapping):  # baselineGap stores value + CI together
            ci, value = value, value.get("value")
        rows.append({"suite": suite, "section": section, "metric": name, "value": value, **_ci_fields(ci)})
    return rows


def metric_summary_rows(run: Any) -> list[Row]:
    rows: list[Row] = []
    for suite, data in (_metric_payload(run).get("suites") or {}).items():
        if not isinstance(data, Mapping):
            continue
        if data.get("status") and not data.get("metrics"):
            rows.append({
                "suite": suite, "section": "status", "metric": data.get("status"),
                "value": data.get("message", ""),
            })
            continue
        rows += _metric_block_rows(suite, "main", data)
        rows += _metric_block_rows(suite, "noise_baseline", data.get("noiseBaseline"))
        rows += _metric_block_rows(suite, "baseline_gap", data.get("baselineGap"))
    return rows


def _comparison_blocks(node: Any, path: tuple[str, ...]):
    """Yield (path, block) for every nested dict that holds a byGroup comparison."""
    if not isinstance(node, Mapping):
        return
    if "byGroup" in node:
        yield path, node
        return
    for key, child in node.items():
        yield from _comparison_blocks(child, (*path, str(key)))


def group_comparison_rows(run: Any) -> list[Row]:
    rows: list[Row] = []
    for suite, data in (_metric_payload(run).get("suites") or {}).items():
        if not isinstance(data, Mapping):
            continue
        for path, block in _comparison_blocks(data.get("groupComparison"), ()):
            base = {"suite": suite, "attribute": path[-1] if path else "", "scope": ".".join(path)}
            counts = block.get("counts") or {}
            for group, metrics in (block.get("byGroup") or {}).items():
                count = counts.get(group) or {}
                for name, value in (metrics or {}).items():
                    rows.append({
                        **base, "type": "group", "group": group, "metric": name, "value": value,
                        "responses": count.get("responses"), "units": count.get("units"),
                    })
            gap = block.get("gap") or {}
            intervals = gap.get("confidenceIntervals") or {}
            for name, value in (gap.get("metrics") or {}).items():
                ci = intervals.get(name) or {}
                rows.append({
                    **base, "type": "gap", "group": block.get("comparison", ""), "metric": name,
                    "value": value, "ci_low": ci.get("ciLow"), "ci_high": ci.get("ciHigh"),
                })
    return rows


def _flatten(node: Any, path: str, out: list[Row], suite: str) -> None:
    if isinstance(node, Mapping):
        for key, child in node.items():
            _flatten(child, f"{path}.{key}" if path else str(key), out, suite)
    elif isinstance(node, list) and any(isinstance(item, (Mapping, list)) for item in node):
        for index, child in enumerate(node):
            _flatten(child, f"{path}[{index}]", out, suite)
    else:
        out.append({"suite": suite, "path": path, "value": node})


def metric_detail_rows(run: Any) -> list[Row]:
    """The full LangFair result, flattened to one row per leaf value."""
    payload = _metric_payload(run)
    rows: list[Row] = []
    for key, value in payload.items():
        if key == "suites":
            continue
        _flatten(value, key, rows, "")
    for suite, data in (payload.get("suites") or {}).items():
        _flatten(data, "", rows, suite)
    return rows


# key -> (file name, Chinese title, builder)
TABLES: dict[str, tuple[str, str, Callable[[Any], list[Row]]]] = {
    "run": ("01_run_info.csv", "執行資訊", run_info_rows),
    "responses": ("02_responses.csv", "回應", response_rows),
    "prompts": ("03_prompts.csv", "Prompt", prompt_rows),
    "personas": ("04_personas.csv", "虛擬人", persona_rows),
    "metrics": ("05_metrics_summary.csv", "公平性指標摘要", metric_summary_rows),
    "group_comparison": ("06_group_comparison.csv", "分組比較", group_comparison_rows),
    "metrics_detail": ("07_metrics_detail.csv", "指標完整明細", metric_detail_rows),
}


def to_csv_bytes(rows: list[Row]) -> bytes:
    columns: list[str] = []
    for row in rows:
        columns += [key for key in row if key not in columns]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns or ["(empty)"], extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(value) for key, value in row.items()})
    return buffer.getvalue().encode("utf-8-sig")


def build_table(run: Any, key: str) -> tuple[str, bytes]:
    file_name, _, builder = TABLES[key]
    return file_name, to_csv_bytes(builder(run))


def build_zip(run: Any) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for key in TABLES:
            file_name, content = build_table(run, key)
            archive.writestr(file_name, content)
    return buffer.getvalue()
