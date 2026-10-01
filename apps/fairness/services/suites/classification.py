# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..utils import _missing_input, _missing_keys, _safe_int, extract_protected_group


def evaluate_classification(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import numpy as np
    from langfair.metrics.classification.classification import ClassificationMetrics

    opts = dict(options or {})
    missing = _missing_keys(opts, ["groups", "yPred"])
    if missing:
        return _missing_input("classification", missing)
    groups = np.array(opts.pop("groups"))
    y_pred = np.array(opts.pop("yPred"))

    validation_error = validate_classification_metric_arrays(groups, y_pred)
    if validation_error is not None:
        return validation_error

    y_true = opts.pop("yTrue", None)
    if y_true is not None:
        y_true = np.array(y_true)
    metric_type = str(opts.pop("metricType", "all" if y_true is not None else "representation"))
    evaluator = ClassificationMetrics(metric_type=metric_type)
    evaluate_kwargs: dict[str, Any] = {
        "groups": groups,
        "y_pred": y_pred,
        "ratio": bool(opts.pop("ratio", False)),
    }
    if y_true is not None:
        evaluate_kwargs["y_true"] = y_true

    return evaluate_classification_metrics_safely(
        evaluator=evaluator,
        evaluate_kwargs=evaluate_kwargs,
    )


def evaluate_classification_for_run(
    run: Any,
    response_rows: Sequence[Any],
    options: Mapping[str, Any] | None = None,
    evaluate: Callable[..., dict[str, Any]] = evaluate_classification,
) -> dict[str, Any]:
    opts = dict(options or {})
    opts.pop("language", None)
    if "groups" in opts and "yPred" in opts:
        return evaluate(options=opts)

    extracted = build_classification_metric_inputs(run, response_rows, opts)
    if extracted.get("status") == "missing_input":
        return extracted

    metric_opts = {**opts, **extracted["inputs"]}
    metrics = evaluate(options=metric_opts)
    status_value = metrics.get("status") if isinstance(metrics, Mapping) else None
    return {
        "status": status_value or "ok",
        "inputs": extracted["inputs"],
        "classificationLabels": extracted["classificationLabels"],
        "mapping": extracted["mapping"],
        "extractionFailures": extracted["extractionFailures"],
        "diagnostics": extracted.get("diagnostics", {}),
        "metrics": metrics,
    }


def build_classification_metric_inputs(
    run: Any,
    response_rows: Sequence[Any],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    mapping = _normalize_classification_mapping(options)
    if mapping is None:
        return _missing_input(
            "classification",
            ["classificationMetricMapping.positiveLabels"],
        )

    labels = run.evaluation.get_classification_labels()
    if len(labels) != 2:
        return {
            "status": "missing_input",
            "suite": "classification",
            "missingOptions": ["binary classification labels"],
            "message": "LangFair classification metrics currently support binary classification only. Please configure exactly two classification labels.",
        }

    value_by_name = {str(label.get("name")): label.get("value") for label in labels}
    label_values = {label.get("value") for label in labels}
    positive_values = _resolve_label_refs(mapping["positive_labels"], value_by_name)
    negative_values = _resolve_label_refs(mapping.get("negative_labels", []), value_by_name)
    if len(positive_values) != 1:
        return _missing_input("classification", ["exactly one positiveLabel"])
    if len(negative_values) != 1:
        return _missing_input("classification", ["exactly one negativeLabel"])
    if unknown := sorted((positive_values | negative_values) - label_values):
        return {
            "status": "missing_input",
            "suite": "classification",
            "missingOptions": ["valid classification label mapping"],
            "message": f"Unknown classification labels in mapping: {unknown}",
        }
    if positive_values == negative_values:
        return {
            "status": "missing_input",
            "suite": "classification",
            "missingOptions": ["distinct positive and negative labels"],
            "message": "Positive and negative labels must be different for binary classification metrics.",
        }
    if positive_values | negative_values != label_values:
        return {
            "status": "missing_input",
            "suite": "classification",
            "missingOptions": ["complete binary classification label mapping"],
            "message": "Positive and negative labels must cover the two configured classification labels.",
        }

    groups: list[str] = []
    y_pred: list[int] = []
    y_true: list[int] = []
    has_expected_labels = True
    failures: list[dict[str, Any]] = []

    for response in response_rows:
        group = extract_protected_group(response)
        parsed = parse_classification_response(
            str(getattr(response, "response_text", "")),
            labels,
        )
        if group and parsed.get("ok"):
            decision = parsed["decision"]
            expected_label = extract_expected_classification_label(response)
            groups.append(group)
            y_pred.append(1 if decision in positive_values else 0)
            if expected_label in positive_values:
                y_true.append(1)
            elif expected_label in negative_values:
                y_true.append(0)
            else:
                has_expected_labels = False
        else:
            failures.append(
                {
                    "responseId": getattr(response, "id", None),
                    "reason": parsed.get("error") if not parsed.get("ok") else "missing_group",
                }
            )

    if not groups or not y_pred:
        return _missing_input("classification", ["valid classified responses with protected groups"])

    inputs = {"groups": groups, "yPred": y_pred}
    diagnostics: dict[str, Any] = {}
    if has_expected_labels and len(y_true) == len(y_pred):
        inputs["yTrue"] = y_true
        diagnostics["confusionMatrix"] = build_classification_confusion_summary(
            groups=groups,
            y_pred=y_pred,
            y_true=y_true,
        )

    return {
        "inputs": inputs,
        "classificationLabels": labels,
        "mapping": {
            "positiveLabels": sorted(positive_values),
            "negativeLabels": sorted(negative_values),
        },
        "extractionFailures": failures,
        "diagnostics": diagnostics,
    }


def _normalize_classification_mapping(options: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = options.get("classificationMetricMapping")
    if not isinstance(raw, Mapping):
        if "positiveLabels" in options:
            raw = options
        else:
            return None
    positives = raw.get("positiveLabels")
    negatives = raw.get("negativeLabels") or []
    if not isinstance(positives, Sequence) or isinstance(positives, str):
        return None
    if not isinstance(negatives, Sequence) or isinstance(negatives, str):
        negatives = []
    return {"positive_labels": list(positives), "negative_labels": list(negatives)}


def _resolve_label_refs(values: Sequence[Any], value_by_name: Mapping[str, Any]) -> set[Any]:
    resolved: set[Any] = set()
    for value in values:
        if isinstance(value, str) and value in value_by_name:
            resolved.add(value_by_name[value])
        else:
            resolved.add(value)
    return resolved


def parse_classification_response(text: str, labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized_text = text.strip()
    labels_by_value = {label.get("value"): label for label in labels}
    decision_from_plain_text = resolve_classification_label(normalized_text, labels)
    if decision_from_plain_text is not None:
        return {"ok": True, "decision": decision_from_plain_text}

    try:
        payload = json.loads(normalized_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", normalized_text, flags=re.DOTALL)
        if not match:
            return {"ok": False, "error": "invalid_classification_output"}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"ok": False, "error": "invalid_classification_output"}

    if not isinstance(payload, Mapping) or "decision" not in payload:
        return {"ok": False, "error": "missing_decision"}

    decision = payload.get("decision")
    if isinstance(decision, str) and decision.isdigit():
        try:
            decision = int(decision)
        except ValueError:
            return {"ok": False, "error": "unknown_decision"}
    elif isinstance(decision, str):
        resolved = resolve_classification_label(decision, labels)
        if resolved is not None:
            decision = resolved
    if decision not in labels_by_value:
        return {"ok": False, "error": "unknown_decision"}

    # Trust the numeric decision when it is valid. Some target models return a
    # free-form label such as "fall risk" even when the requested label name is
    # "yes"; treating that as a hard failure can drop an entire protected group.
    return {"ok": True, "decision": decision}


def resolve_classification_label(text: str, labels: Sequence[Mapping[str, Any]]) -> Any | None:
    candidate = text.strip().strip('"「」。，. ')
    if not candidate:
        return None
    for label in labels:
        label_refs = {
            str(label.get("value", "")),
            str(label.get("name", "")),
            str(label.get("description", "")),
        }
        if candidate in label_refs:
            return label.get("value")
    return None


def extract_expected_classification_label(response: Any) -> Any | None:
    prompt = getattr(response, "prompt", None)
    metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("expected_label")
    if isinstance(value, str) and value.isdigit():
        try:
            return int(value)
        except ValueError:
            return value
    return value


def build_classification_confusion_summary(
    groups: Sequence[str],
    y_pred: Sequence[int],
    y_true: Sequence[int],
) -> dict[str, Any]:
    by_group: dict[str, dict[str, Any]] = {}
    overall = _empty_confusion_counts()

    for group, pred, actual in zip(groups, y_pred, y_true, strict=False):
        group_counts = by_group.setdefault(str(group), _empty_confusion_counts())
        _add_confusion_observation(group_counts, pred, actual)
        _add_confusion_observation(overall, pred, actual)

    return {
        "positiveLabel": 1,
        "negativeLabel": 0,
        "overall": _with_confusion_rates(overall),
        "byGroup": {
            group: _with_confusion_rates(counts)
            for group, counts in sorted(by_group.items())
        },
    }


def _empty_confusion_counts() -> dict[str, Any]:
    return {"total": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}


def _add_confusion_observation(counts: dict[str, Any], pred: int, actual: int) -> None:
    counts["total"] += 1
    if actual == 1 and pred == 1:
        counts["tp"] += 1
    elif actual == 0 and pred == 0:
        counts["tn"] += 1
    elif actual == 0 and pred == 1:
        counts["fp"] += 1
    elif actual == 1 and pred == 0:
        counts["fn"] += 1


def _with_confusion_rates(counts: Mapping[str, Any]) -> dict[str, Any]:
    tp = _safe_int(counts.get("tp", 0), 0)
    tn = _safe_int(counts.get("tn", 0), 0)
    fp = _safe_int(counts.get("fp", 0), 0)
    fn = _safe_int(counts.get("fn", 0), 0)
    total = _safe_int(counts.get("total", 0), 0)
    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_rate(tp + tn, total),
        "falsePositiveRate": _safe_rate(fp, fp + tn),
        "falseNegativeRate": _safe_rate(fn, fn + tp),
        "falseDiscoveryRate": _safe_rate(fp, fp + tp),
        "falseOmissionRate": _safe_rate(fn, fn + tn),
        "predictedPositiveRate": _safe_rate(tp + fp, total),
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def evaluate_classification_metrics_safely(evaluator: Any, evaluate_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate LangFair classification metrics without failing the whole suite.

    Some LangFair classification metrics (for example FNRP/FPRP) require each
    protected group to contain specific confusion-matrix cells. Real runs can be
    valid for representation metrics while still missing positives/negatives for
    error-rate metrics, so return computable metrics plus per-metric failures.
    """
    scores: dict[str, Any] = {}
    failures: list[dict[str, str]] = []

    for metric in getattr(evaluator, "metrics", []):
        metric_name = str(getattr(metric, "name", metric.__class__.__name__))
        try:
            scores[metric_name] = metric.evaluate(**evaluate_kwargs)
        except (AssertionError, TypeError, ValueError) as exc:
            failures.append({"metric": metric_name, "message": str(exc).strip()})

    if not scores:
        return {
            "status": "insufficient_input",
            "message": "No LangFair classification metrics could be computed for the current class/group distribution.",
            "metricFailures": failures,
        }
    if failures:
        scores["metricFailures"] = failures
    return scores


def validate_classification_metric_arrays(groups: Any, y_pred: Any) -> dict[str, Any] | None:
    group_values = sorted({str(value) for value in groups.tolist()})
    try:
        pred_values = sorted({int(value) for value in y_pred.tolist()})
    except (TypeError, ValueError):
        pred_values = sorted({str(value) for value in y_pred.tolist()})
    if len(group_values) != 2:
        return {
            "status": "insufficient_input",
            "suite": "classification",
            "message": "Classification fairness metrics require exactly two protected groups in the evaluated responses.",
            "groupValues": group_values,
            "predictionValues": pred_values,
        }
    if pred_values != [0, 1]:
        return {
            "status": "insufficient_input",
            "suite": "classification",
            "message": "Classification fairness metrics require both negative (0) and positive (1) predictions. The current run predicted only one class, so LangFair parity metrics cannot be computed.",
            "groupValues": group_values,
            "predictionValues": pred_values,
        }
    return None

