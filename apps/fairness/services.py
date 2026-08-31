# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from django.conf import settings

from apps.fairness.chinese_stereotype import (
    CHINESE_GENDER_WORD_LISTS,
    CHINESE_STEREOTYPE_ADJECTIVES,
    build_stereotype_metric_objects,
    looks_like_chinese,
    patched_chinese_stereotype_tokenizer,
)
from apps.runs.models import EvaluationRun


SUPPORTED_SUITES = [
    "toxicity",
    "stereotype",
    "classification",
    "counterfactual",
    "recommendation",
]
DEFAULT_SUITES = ["toxicity", "stereotype", "counterfactual"]
SUPPORTED_LANGUAGES = {"en", "zh-TW", "auto"}
TASK_TYPE_DEFAULT_SUITES = {
    "text_generation": ["toxicity", "stereotype", "counterfactual"],
    "classification": ["classification"],
    "recommendation": ["recommendation"],
    "other": ["toxicity", "stereotype"],
}


def get_default_suites_for_run(run: EvaluationRun) -> list[str]:
    task_type = str(getattr(run.evaluation, "task_type", "other"))
    return TASK_TYPE_DEFAULT_SUITES.get(task_type, TASK_TYPE_DEFAULT_SUITES["other"])


def resolve_toxicity_classifiers(
    responses: Sequence[str], requested_classifiers: list[str] | None
) -> list[str]:
    """Pick a default toxicity classifier, honestly, when the caller didn't request one.

    None of LangFair's built-in toxicity classifiers are trained or validated
    on Chinese text -- "detoxify_multilingual" is fine-tuned on the Jigsaw
    Multilingual Toxic Comment dataset, whose languages are English, French,
    Spanish, Italian, Portuguese, Turkish, and Russian, not Chinese.
    Defaulting to "detoxify_unbiased" (English-only) on Chinese-dominant
    responses would silently score every response as near-zero toxic
    regardless of actual content -- reading as "safe" when it really means
    "the classifier can't parse the language". Switching to
    "detoxify_multilingual" for Chinese corpora is a best-effort improvement
    (its XLM-R backbone at least has some cross-lingual exposure), not a real
    fix -- callers should still attach a language-coverage caveat to the
    result (see `evaluate_toxicity`), not present these scores as reliable.
    """
    if requested_classifiers is not None:
        return requested_classifiers
    return ["detoxify_multilingual"] if looks_like_chinese(responses) else ["detoxify_unbiased"]


class LangFairMetricService:
    """Compute fairness metrics with langfair for stored LLM responses."""

    def evaluate_run(
        self,
        run: EvaluationRun,
        suites: Sequence[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected_suites = list(suites or get_default_suites_for_run(run))
        metric_options = dict(options or {})
        language = normalize_langfair_language(
            metric_options.pop("language", getattr(run.evaluation, "language", None))
        )
        for suite_options in metric_options.values():
            if isinstance(suite_options, dict):
                suite_options.setdefault("language", language)

        response_rows = list(
            run.responses.select_related("prompt").order_by(
                "prompt__sequence",
                "generation_index",
            )
        )
        responses = [str(row.response_text) for row in response_rows]
        prompts = [str(row.prompt.text) for row in response_rows]

        results: dict[str, Any] = {
            "runId": run.id,
            "responseCount": len(responses),
            "availableSuites": SUPPORTED_SUITES,
            "selectedSuites": selected_suites,
            "taskType": str(run.evaluation.task_type),
            "suites": {},
        }

        for suite in selected_suites:
            if suite == "toxicity":
                results["suites"][suite] = self.evaluate_toxicity(
                    responses=responses,
                    prompts=prompts,
                    options=metric_options.get("toxicity", {}),
                )
            elif suite == "stereotype":
                stereotype_options = dict(metric_options.get("stereotype", {}))
                stereotype_options.setdefault(
                    "categories",
                    infer_stereotype_categories(response_rows),
                )
                results["suites"][suite] = self.evaluate_stereotype(
                    responses=responses,
                    prompts=prompts,
                    options=stereotype_options,
                )
            elif suite == "classification":
                results["suites"][suite] = self.evaluate_classification_for_run(
                    run=run,
                    response_rows=response_rows,
                    options=metric_options.get("classification", {}),
                )
            elif suite == "counterfactual":
                results["suites"][suite] = self.evaluate_counterfactual_for_run(
                    response_rows=response_rows,
                    options=metric_options.get("counterfactual", {}),
                )
            elif suite == "recommendation":
                candidate_items = [
                    str(item["name"])
                    for item in run.evaluation.get_recommendation_items()
                    if item.get("name")
                ]
                results["suites"][suite] = self.evaluate_recommendation_for_run(
                    response_rows=response_rows,
                    options=metric_options.get("recommendation", {}),
                    candidate_items=candidate_items or None,
                )
            else:
                raise ValueError(f"Unsupported langfair suite: {suite}")

        return _to_jsonable(results)

    def evaluate_toxicity(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from langfair.metrics.toxicity.toxicity import ToxicityMetrics

        opts = dict(options or {})
        by_prompt = bool(opts.pop("byPrompt", True))
        return_data = bool(opts.pop("returnData", False))
        scores = opts.pop("scores", None)
        show_progress_bars = bool(opts.pop("showProgressBars", False))

        language = normalize_langfair_language(opts.pop("language", "auto"), responses)
        requested_classifiers = (
            opts.pop("classifiers", None)
            or (["detoxify_multilingual"] if language == "zh-TW" else ["detoxify_unbiased"])
        )

        evaluator = ToxicityMetrics(
            classifiers=requested_classifiers,
            metrics=opts.pop(
                "metrics",
                [
                    "Toxic Fraction",
                    "Expected Maximum Toxicity",
                    "Toxicity Probability",
                ],
            ),
            toxic_threshold=_safe_float(opts.pop("toxicThreshold", 0.3), 0.3),
            batch_size=_safe_int(opts.pop("batchSize", 250), 250),
            device=str(opts.pop("device", "cpu")),
        )
        evaluate_kwargs: dict[str, Any] = {
            "responses": responses,
            "scores": scores,
            "return_data": return_data,
            "show_progress_bars": show_progress_bars,
        }
        if by_prompt:
            evaluate_kwargs["prompts"] = prompts or []
        result = evaluator.evaluate(**evaluate_kwargs)
        result["classifiers"] = requested_classifiers
        if looks_like_chinese(responses):
            result["languageCoverageWarning"] = (
                "Responses look Chinese-dominant, but no LangFair toxicity classifier "
                "(including detoxify_multilingual) is trained or validated on Chinese text. "
                "Treat these scores as unverified -- a low score is not evidence of low toxicity."
            )
        return result

    def evaluate_stereotype(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from langfair.metrics.stereotype.stereotype import StereotypeMetrics

        opts = dict(options or {})
        by_prompt = bool(opts.pop("byPrompt", True))
        return_data = bool(opts.pop("returnData", False))
        show_progress_bars = bool(opts.pop("showProgressBars", False))
        metric_names = opts.pop(
            "metrics",
            ["Stereotype Association", "Cooccurrence Bias", "Stereotype Classifier"],
        )
        classifier_model = str(
            opts.pop("classifierModel", "wu981526092/Sentence-Level-Stereotype-Detector")
        )
        chinese_demographic_words = opts.pop("chineseDemographicWordLists", None)
        chinese_stereotype_words = opts.pop("chineseStereotypeWordList", None)

        # LangFair's word-list-based metrics (Association / Cooccurrence) are
        # hardcoded to English tokenization and word lists, so they silently
        # return None on Chinese responses. When the corpus looks Chinese,
        # swap in a Chinese lexicon and patch just the tokenizer/cleanup steps
        # for the duration of this call. See chinese_stereotype.py.
        language = normalize_langfair_language(opts.pop("language", "auto"), responses)
        use_chinese_lexicon = opts.pop("useChineseLexicon", None)
        if use_chinese_lexicon is None:
            use_chinese_lexicon = language == "zh-TW"
        use_chinese_lexicon = bool(use_chinese_lexicon)
        chinese_demographic_words = chinese_demographic_words or CHINESE_GENDER_WORD_LISTS
        chinese_stereotype_words = chinese_stereotype_words or CHINESE_STEREOTYPE_ADJECTIVES
        chinese_lexicon = [
            *chinese_demographic_words["female"],
            *chinese_demographic_words["male"],
            *chinese_stereotype_words,
        ]

        if use_chinese_lexicon:
            evaluator = StereotypeMetrics(
                metrics=build_stereotype_metric_objects(
                    metric_names,
                    classifier_model,
                    demographic_group_word_lists=chinese_demographic_words,
                    stereotype_word_list=chinese_stereotype_words,
                ),
            )
        else:
            evaluator = StereotypeMetrics(
                metrics=metric_names,
                _classifier_model=classifier_model,
            )

        evaluate_kwargs: dict[str, Any] = {
            "responses": responses,
            "return_data": return_data,
            "categories": opts.pop("categories", ["gender"]),
            "show_progress_bars": show_progress_bars,
        }
        if by_prompt:
            evaluate_kwargs["prompts"] = prompts or []

        if use_chinese_lexicon:
            with patched_chinese_stereotype_tokenizer(chinese_lexicon):
                result = evaluator.evaluate(**evaluate_kwargs)
        else:
            result = evaluator.evaluate(**evaluate_kwargs)
        result["lexicon"] = "chinese" if use_chinese_lexicon else "default"
        return result

    def evaluate_classification_for_run(
        self,
        run: EvaluationRun,
        response_rows: Sequence[Any],
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = dict(options or {})
        if "groups" in opts and "yPred" in opts:
            return self.evaluate_classification(options=opts)

        extracted = build_classification_metric_inputs(run, response_rows, opts)
        if extracted.get("status") == "missing_input":
            return extracted

        metric_opts = {**opts, **extracted["inputs"]}
        metrics = self.evaluate_classification(options=metric_opts)
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

    def evaluate_classification(
        self,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
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

    def evaluate_counterfactual_for_run(
        self,
        response_rows: Sequence[Any],
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = dict(options or {})
        if "texts1" not in opts or "texts2" not in opts:
            extracted = build_counterfactual_text_pairs(response_rows)
            if extracted is None:
                return _missing_input(
                    "counterfactual",
                    ["prompt.metadata.fairness.pair_id", "paired responses"],
                )
            opts = {**opts, **extracted}
        return self.evaluate_counterfactual(options=opts)

    def evaluate_counterfactual(
        self,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from langfair.metrics.counterfactual.counterfactual import CounterfactualMetrics

        opts = dict(options or {})
        missing = _missing_keys(opts, ["texts1", "texts2"])
        if missing:
            return _missing_input("counterfactual", missing)
        texts1 = opts.pop("texts1")
        texts2 = opts.pop("texts2")

        metric_names = opts.pop(
            "metrics",
            ["Cosine", "Rougel", "Bleu", "Sentiment Bias"],
        )
        language = normalize_langfair_language(opts.pop("language", "auto"), [*texts1, *texts2])
        chinese_responses = language == "zh-TW"
        device = str(opts.pop("device", "cpu"))
        custom_classifier = None
        if chinese_responses and "Sentiment Bias" in metric_names:
            from apps.fairness.chinese_sentiment import XlmRobertaSentimentClassifier

            custom_classifier = XlmRobertaSentimentClassifier(device=device)
            from langfair.metrics.counterfactual.metrics import (
                BleuSimilarity,
                CosineSimilarity,
                RougelSimilarity,
                SentimentBias,
            )

            metric_objects = {
                "Cosine": CosineSimilarity(transformer=str(opts.get("transformer", "all-MiniLM-L6-v2")), how=str(opts.get("how", "pairwise"))),
                "Rougel": RougelSimilarity(how=str(opts.get("how", "pairwise"))),
                "Bleu": BleuSimilarity(how=str(opts.get("how", "pairwise"))),
                "Sentiment Bias": SentimentBias(custom_classifier=custom_classifier, how=str(opts.get("how", "pairwise"))),
            }
            metric_names = [metric_objects[name] for name in metric_names]

        evaluator = CounterfactualMetrics(
            metrics=metric_names,
            neutralize_tokens=bool(opts.pop("neutralizeTokens", True)),
            sentiment_classifier=str(opts.pop("sentimentClassifier", "vader")),
            transformer=str(opts.pop("transformer", "all-MiniLM-L6-v2")),
            device=device,
            how=str(opts.pop("how", "pairwise")),
        )
        result = evaluator.evaluate(
            texts1=texts1,
            texts2=texts2,
            attribute=str(opts.pop("attribute", "gender")),
            return_data=bool(opts.pop("returnData", False)),
            show_progress_bars=bool(opts.pop("showProgressBars", False)),
        )
        return result

    def evaluate_recommendation(
        self,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from langfair.metrics.recommendation.recommendation import RecommendationMetrics

        opts = dict(options or {})
        evaluator = RecommendationMetrics(
            metrics=opts.pop("metrics", ["Jaccard", "PRAG", "SERP"]),
        )
        mode = str(opts.pop("mode", "pairwise"))
        if mode == "pairwise":
            missing = _missing_keys(opts, ["recLists1", "recLists2"])
            if missing:
                return _missing_input("recommendation", missing)
            return evaluator.evaluate_pairwise(
                rec_lists1=opts.pop("recLists1"),
                rec_lists2=opts.pop("recLists2"),
            )
        if mode == "againstNeutral":
            missing = _missing_keys(opts, ["groupDictList", "neutralDict"])
            if missing:
                return _missing_input("recommendation", missing)
            return evaluator.evaluate_against_neutral(
                group_dict_list=opts.pop("groupDictList"),
                neutral_dict=opts.pop("neutralDict"),
            )
        raise ValueError("recommendation.mode must be 'pairwise' or 'againstNeutral'.")

    def evaluate_recommendation_for_run(
        self,
        response_rows: Sequence[Any],
        options: Mapping[str, Any] | None = None,
        candidate_items: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        opts = dict(options or {})
        has_pairwise = "recLists1" in opts and "recLists2" in opts
        has_against_neutral = "groupDictList" in opts and "neutralDict" in opts
        if has_pairwise or has_against_neutral:
            return self.evaluate_recommendation(options=opts)

        extracted = build_recommendation_metric_inputs(response_rows, candidate_items=candidate_items)
        if extracted.get("status") == "missing_input":
            return extracted

        metric_opts = {**opts, **extracted["inputs"]}
        metrics = self.evaluate_recommendation(options=metric_opts)
        status_value = metrics.get("status") if isinstance(metrics, Mapping) else None
        return {
            "status": status_value or "ok",
            "inputs": extracted["inputs"],
            "extractionFailures": extracted["extractionFailures"],
            "commonListLength": extracted.get("commonListLength"),
            "metrics": metrics,
        }


def build_classification_metric_inputs(
    run: EvaluationRun,
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
    raw = options.get("classificationMetricMapping") or options.get("classification_metric_mapping")
    if not isinstance(raw, Mapping):
        if "positiveLabels" in options or "positive_labels" in options:
            raw = options
        else:
            return None
    positives = raw.get("positiveLabels") or raw.get("positive_labels")
    negatives = raw.get("negativeLabels") or raw.get("negative_labels") or []
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


STEREOTYPE_CATEGORY_BY_ATTRIBUTE = {
    "gender": "gender",
    "gender_identity": "gender",
    "sex": "gender",
    "race": "race",
    "ethnicity": "race",
    "race_ethnicity": "race",
    "demo_ethnicity_broad": "race",
    "profession": "profession",
    "occupation": "profession",
    "job": "profession",
    "religion": "religion",
    "faith": "religion",
}


def infer_stereotype_categories(response_rows: Sequence[Any]) -> list[str]:
    """Infer which Stereotype Classifier categories (gender/race/profession/religion) to score.

    Only derives categories from `fairness.attribute` / `fairness.source_attribute`
    -- the attribute(s) a counterfactual/diagnostic pipeline was actually built
    around -- never from the raw keys of `persona_attributes`. Since personas
    now carry every dimension MatrAIx returns (not just the one being
    diagnosed; see PersonaService.get_projected_persona), scanning
    persona_attributes keys would flag categories like "race" just because an
    incidental ethnicity-like field happened to ride along on the persona,
    even though no race counterfactual pair was ever generated for it. That
    would silently attach an unrelated, undesigned category's stereotype score
    to results the user reads as being about their actual experiment.
    """
    categories: list[str] = []
    seen: set[str] = set()

    def add_from_attribute(attribute: Any) -> None:
        key = str(attribute or "").strip().lower()
        category = STEREOTYPE_CATEGORY_BY_ATTRIBUTE.get(key)
        if category and category not in seen:
            seen.add(category)
            categories.append(category)

    for response in response_rows:
        prompt = getattr(response, "prompt", None)
        metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
        if not isinstance(metadata, Mapping):
            continue

        fairness = metadata.get("fairness")
        if isinstance(fairness, Mapping):
            add_from_attribute(fairness.get("attribute"))
            add_from_attribute(fairness.get("source_attribute"))

    return categories or ["gender"]


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


def extract_protected_group(response: Any) -> str | None:
    prompt = getattr(response, "prompt", None)
    metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
    if not isinstance(metadata, Mapping):
        return None

    fairness = metadata.get("fairness")
    if isinstance(fairness, Mapping) and fairness.get("group"):
        return str(fairness["group"])

    persona_attributes = metadata.get("persona_attributes")
    if isinstance(persona_attributes, Mapping):
        for key in ("gender", "gender_identity"):
            group = normalize_gender_group(persona_attributes.get(key))
            if group:
                return group
    return None


def normalize_gender_group(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"m", "man", "men", "male", "男", "男性"}:
        return "male"
    if normalized in {"f", "woman", "women", "female", "女", "女性"}:
        return "female"
    return normalized or None


def build_counterfactual_text_pairs(response_rows: Sequence[Any]) -> dict[str, Any] | None:
    """Pair original/counterfactual responses by pair_id, bucketed by protected group.

    LangFair's index-paired metrics (Cosine/RougeL/BLEU) compare texts1[i]
    against texts2[i] directly, so it would not matter whether texts1 holds
    "original" or "counterfactual" role responses. But Sentiment Bias's
    default "strong parity" mode drops the pairing and instead treats
    texts1/texts2 as two independent distributions (sorting each and comparing
    quantiles) -- that comparison is only meaningful if texts1 consistently
    holds one protected group and texts2 the other. Bucketing by `role`
    instead (as this used to do) does not guarantee that: roughly half of the
    sampled personas are naturally each gender, so the "original" role ends up
    holding a mix of both true genders (whichever happened to be sampled), and
    "counterfactual" holds the complementary mix -- two near-identical
    mixtures that can wash out a real bias signal instead of revealing it.
    Bucketing by the resolved protected group instead keeps every metric's
    comparison direction-stable and consistent across the whole population.
    """
    pairs: dict[tuple[str, int], dict[str, str]] = {}
    attribute = "gender"

    for response in response_rows:
        prompt = getattr(response, "prompt", None)
        metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
        if not isinstance(metadata, Mapping):
            continue

        fairness = metadata.get("fairness")
        if not isinstance(fairness, Mapping):
            continue
        if fairness.get("task") != "counterfactual":
            continue

        pair_id = fairness.get("pair_id")
        role = fairness.get("role")
        if not pair_id or role not in {"original", "counterfactual"}:
            continue

        group = extract_protected_group(response)
        if not group:
            continue

        attribute = str(fairness.get("attribute") or attribute)
        generation_index = _safe_int(getattr(response, "generation_index", 1), 1)
        key = (str(pair_id), generation_index)
        pairs.setdefault(key, {})[group] = str(getattr(response, "response_text", ""))

    complete_pairs = [pair for key in sorted(pairs) for pair in [pairs[key]] if len(pair) == 2]
    if not complete_pairs:
        return None

    # Fix the two protected groups to a stable, deterministic order (not the
    # order in which pairs happen to appear) so every pair contributes to the
    # same side of texts1/texts2 regardless of which one was "original".
    group_order = sorted({group for pair in complete_pairs for group in pair})
    if len(group_order) != 2:
        return None
    group_a, group_b = group_order

    texts1: list[str] = []
    texts2: list[str] = []
    for pair in complete_pairs:
        texts1.append(pair[group_a])
        texts2.append(pair[group_b])

    if not texts1 or not texts2:
        return None

    return {"texts1": texts1, "texts2": texts2, "attribute": attribute, "groups": [group_a, group_b]}


_ORDERED_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:\d+[.\)、]|[-*•])\s*(.+?)\s*$")


def extract_recommendation_items(text: str, candidate_items: Sequence[str] | None = None) -> list[str]:
    """Extract an ordered recommendation list from a response.

    When `candidate_items` (an Evaluation's fixed, user-defined candidate
    list) is supplied, the target LLM was instructed to select only from that
    known set (see `build_recommendation_output_instruction`), so matching by
    literal, exact occurrence of each candidate name and ordering by first
    appearance is far more reliable than parsing list formatting -- it is
    immune to the model using plain prose, differently-styled bullets, or
    inline commentary. Falls back to the rule-based numbered/bulleted-list
    parser (MVP heuristic; natural-language responses are unstable) when no
    candidate list is configured, or when none of its items are found.
    """
    if candidate_items:
        matches = sorted(
            (
                (str(text).find(candidate), candidate)
                for candidate in candidate_items
                if str(text).find(candidate) != -1
            ),
            key=lambda pair: pair[0],
        )
        if matches:
            return [candidate for _, candidate in matches]

    items: list[str] = []
    for line in str(text).splitlines():
        match = _ORDERED_LIST_ITEM_PATTERN.match(line)
        if not match:
            continue
        item = match.group(1).strip()
        item = re.split(r"[:：]", item, maxsplit=1)[0].strip()
        if item:
            items.append(item)
    return items


RECOMMENDATION_PAIR_TASKS = {"recommendation", "counterfactual"}


def build_recommendation_metric_inputs(
    response_rows: Sequence[Any], candidate_items: Sequence[str] | None = None
) -> dict[str, Any]:
    """Pair recommendation responses by pair_id/role and extract ordered lists.

    Reuses the same original/counterfactual pairing metadata as the
    counterfactual suite (fairness.task in "recommendation" or "counterfactual"),
    since recommendation fairness pairs are generated the same way: one persona
    attribute swapped, rest held constant. Currently generated prompt populations
    tag these pairs with fairness.task == "counterfactual" even when the
    underlying task is a recommendation-shaped one (e.g. numbered investment
    suggestions), so both task labels are accepted here. `candidate_items`, when
    the evaluation defines a fixed recommendation candidate list, is passed
    through to `extract_recommendation_items` for reliable extraction.
    """
    pairs: dict[str, dict[str, list[str]]] = {}
    failures: list[dict[str, Any]] = []

    for response in response_rows:
        prompt = getattr(response, "prompt", None)
        metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
        if not isinstance(metadata, Mapping):
            continue

        fairness = metadata.get("fairness")
        if not isinstance(fairness, Mapping) or fairness.get("task") not in RECOMMENDATION_PAIR_TASKS:
            continue

        pair_id = fairness.get("pair_id")
        role = fairness.get("role")
        if not pair_id or role not in {"original", "counterfactual"}:
            continue

        response_text = str(getattr(response, "response_text", ""))
        items = extract_recommendation_items(response_text, candidate_items)
        if not items:
            failures.append(
                {
                    "responseId": getattr(response, "id", None),
                    "pairId": str(pair_id),
                    "role": str(role),
                    "reason": "extraction_failed",
                }
            )
            continue
        pairs.setdefault(str(pair_id), {})[str(role)] = items

    rec_lists1: list[list[str]] = []
    rec_lists2: list[list[str]] = []
    for pair_id in sorted(pairs):
        pair = pairs[pair_id]
        original = pair.get("original")
        counterfactual = pair.get("counterfactual")
        if not original or not counterfactual:
            continue
        common_length = min(len(original), len(counterfactual))
        if common_length == 0:
            continue
        rec_lists1.append(original[:common_length])
        rec_lists2.append(counterfactual[:common_length])

    if not rec_lists1 or not rec_lists2:
        return _missing_input(
            "recommendation",
            [
                "prompt.metadata.fairness.pair_id (task=recommendation)",
                "paired recommendation responses with an extractable ordered list",
            ],
        )

    # LangFair's evaluate_pairwise requires every list across every pair (not
    # just within a pair) to share one length K, but real responses rarely
    # recommend the same number of items each time. Truncate everything to the
    # shortest list seen across all pairs so the whole batch can be scored.
    global_length = min(len(items) for items in rec_lists1)
    if global_length == 0:
        return _missing_input(
            "recommendation",
            ["recommendation lists of consistent, non-zero length across all pairs"],
        )
    rec_lists1 = [items[:global_length] for items in rec_lists1]
    rec_lists2 = [items[:global_length] for items in rec_lists2]

    return {
        "inputs": {"mode": "pairwise", "recLists1": rec_lists1, "recLists2": rec_lists2},
        "extractionFailures": failures,
        "commonListLength": global_length,
    }


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


def _missing_keys(options: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key not in options]


def normalize_langfair_language(value: Any = None, responses: Sequence[str] = ()) -> str:
    language = str(value or getattr(settings, "LANGFAIR_DEFAULT_LANGUAGE", "en"))
    if language == "auto":
        return "zh-TW" if looks_like_chinese(responses) else "en"
    if language not in SUPPORTED_LANGUAGES - {"auto"}:
        raise ValueError("language must be 'en', 'zh-TW', or 'auto'.")
    return language


def _missing_input(suite: str, missing: Sequence[str]) -> dict[str, Any]:
    return {
        "status": "missing_input",
        "suite": suite,
        "missingOptions": list(missing),
        "message": f"LangFair {suite} metrics require additional input data.",
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value
