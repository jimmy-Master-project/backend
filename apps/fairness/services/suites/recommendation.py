from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Any

from ..evaluators import as_key, get_recommendation_evaluator
from ..stats import mean_with_ci
from ..utils import PAIR_ROLES, _missing_input, _missing_keys, fairness_metadata, pair_key, prompt_text

DEFAULT_RECOMMENDATION_METRICS = ["Jaccard", "PRAG", "SERP"]


def _metric_objects(metric_names: Sequence[str]) -> dict[str, Any]:
    from langfair.metrics.recommendation.metrics import PRAG, SERP, JaccardSimilarity

    available = {"Jaccard": JaccardSimilarity(), "PRAG": PRAG(), "SERP": SERP()}
    return {name: available[name] for name in metric_names if name in available}


def normalized_pair_scores(
    lists1: Sequence[Sequence[str]], lists2: Sequence[Sequence[str]], metric_names: Sequence[str]
) -> dict[str, list[float]]:
    """Per-pair, symmetric, 0-1 normalized Jaccard / PRAG / SERP.

    LangFair's raw SERP and PRAG never reach 1: for two identical length-K
    lists SERP = (K+3)/(4(K+1)) and PRAG = (K-1)/(2(K+1)), so values depend on
    K and are not comparable across runs. Each score is divided by the score
    of a list against itself (same K), so identical lists score 1. SERP and
    PRAG are also asymmetric and LangFair's ``_pairwise_calculations`` takes
    ``min(m(a, b), m(a, b))`` (same order twice), so the intended symmetric
    ``min(m(a, b), m(b, a))`` is used here.
    """
    metrics = _metric_objects(metric_names)
    scores: dict[str, list[float]] = {name: [] for name in metrics}
    for list1, list2 in zip(lists1, lists2, strict=True):
        list1, list2 = list(list1), list(list2)
        reference = [f"__item_{index}__" for index in range(len(list1))]
        for name, metric in metrics.items():
            ceiling = metric.evaluate(reference, reference)
            value = min(metric.evaluate(list1, list2), metric.evaluate(list2, list1))
            scores[name].append(value / ceiling if ceiling else 0.0)
    return scores


def normalized_metrics_with_ci(
    lists1: Sequence[Sequence[str]], lists2: Sequence[Sequence[str]], metric_names: Sequence[str]
) -> dict[str, Any]:
    pair_scores = normalized_pair_scores(lists1, lists2, metric_names)
    intervals = {name: mean_with_ci(values) for name, values in pair_scores.items()}
    return {
        "metrics": {name: interval["value"] for name, interval in intervals.items()},
        "confidenceIntervals": intervals,
        "pairCount": len(lists1),
    }


def evaluate_recommendation(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    opts = dict(options or {})
    evaluator = get_recommendation_evaluator(
        as_key(opts.pop("metrics", DEFAULT_RECOMMENDATION_METRICS))
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
    response_rows: Sequence[Any],
    options: Mapping[str, Any] | None = None,
    candidate_items: Sequence[str] | None = None,
    evaluate: Callable[..., dict[str, Any]] = evaluate_recommendation,
) -> dict[str, Any]:
    opts = dict(options or {})
    opts.pop("language", None)
    has_pairwise = "recLists1" in opts and "recLists2" in opts
    has_against_neutral = "groupDictList" in opts and "neutralDict" in opts
    if has_pairwise or has_against_neutral:
        return evaluate(options=opts)

    extracted = build_recommendation_metric_inputs(response_rows, candidate_items=candidate_items)
    if extracted.get("status") == "missing_input":
        return extracted

    metric_opts = {**opts, **extracted["inputs"]}
    metric_names = list(as_key(opts.get("metrics", DEFAULT_RECOMMENDATION_METRICS)) or ())
    raw_metrics = evaluate(options=metric_opts)
    status_value = raw_metrics.get("status") if isinstance(raw_metrics, Mapping) else None
    result: dict[str, Any] = {
        "status": status_value or "ok",
        "inputs": extracted["inputs"],
        "extractionFailures": extracted["extractionFailures"],
        "commonListLength": extracted.get("commonListLength"),
        "excludedIdenticalPromptPairs": extracted.get("excludedIdenticalPromptPairs", 0),
        "comparison": "original vs counterfactual (symmetric)",
        "langfairRawMetrics": raw_metrics,
    }
    if status_value:
        return result
    inputs = extracted["inputs"]
    result.update(normalized_metrics_with_ci(inputs["recLists1"], inputs["recLists2"], metric_names))
    baseline = extracted.get("noiseBaselineLists")
    if baseline and baseline["lists1"]:
        result["noiseBaseline"] = {
            "description": (
                "Same metrics between different generations of the same prompt. "
                "Cross-group similarity only indicates bias when clearly below this baseline."
            ),
            **normalized_metrics_with_ci(baseline["lists1"], baseline["lists2"], metric_names),
        }
    return result


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
    """Pair recommendation responses by pair/role and extract ordered lists.

    Reuses the same original/counterfactual pairing metadata as the
    counterfactual suite (fairness.task in "recommendation" or "counterfactual"),
    since recommendation fairness pairs are generated the same way: one persona
    attribute swapped, rest held constant. Currently generated prompt populations
    tag these pairs with fairness.task == "counterfactual" even when the
    underlying task is a recommendation-shaped one (e.g. numbered investment
    suggestions), so both task labels are accepted here. `candidate_items`, when
    the evaluation defines a fixed recommendation candidate list, is passed
    through to `extract_recommendation_items` for reliable extraction.

    Pairs are keyed by (objective, pair_id, generation_index) so repeated
    generations and legacy per-objective pair ids do not overwrite each other.
    """
    pairs: dict[tuple[str, str, int], dict[str, list[str]]] = {}
    pair_prompts: dict[tuple[str, str, int], set[str | None]] = {}
    generations: dict[tuple[str, str, str], dict[int, list[str]]] = {}
    failures: list[dict[str, Any]] = []

    for response in response_rows:
        fairness = fairness_metadata(response)
        if fairness is None or fairness.get("task") not in RECOMMENDATION_PAIR_TASKS:
            continue

        pair_id = fairness.get("pair_id")
        role = fairness.get("role")
        if not pair_id or role not in PAIR_ROLES:
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
        key = pair_key(response, fairness, pair_id)
        pairs.setdefault(key, {})[str(role)] = items
        pair_prompts.setdefault(key, set()).add(prompt_text(response))
        generations.setdefault((key[0], key[1], str(role)), {})[key[2]] = items

    # Pairs whose two prompts are textually identical (the attribute swap
    # failed) measure only sampling noise, so they are left out.
    identical = {
        key for key, texts in pair_prompts.items()
        if len(pairs[key]) == 2 and len(texts) == 1 and None not in texts
    }
    rec_lists1: list[list[str]] = []
    rec_lists2: list[list[str]] = []
    for key, pair in pairs.items():
        if key in identical:
            continue
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
    rec_lists1 = [items[:global_length] for items in rec_lists1]
    rec_lists2 = [items[:global_length] for items in rec_lists2]

    baseline1: list[list[str]] = []
    baseline2: list[list[str]] = []
    for by_index in generations.values():
        for first, second in combinations(sorted(by_index), 2):
            if min(len(by_index[first]), len(by_index[second])) >= global_length:
                baseline1.append(by_index[first][:global_length])
                baseline2.append(by_index[second][:global_length])

    return {
        "inputs": {"mode": "pairwise", "recLists1": rec_lists1, "recLists2": rec_lists2},
        "extractionFailures": failures,
        "commonListLength": global_length,
        "excludedIdenticalPromptPairs": len(identical),
        "noiseBaselineLists": {"lists1": baseline1, "lists2": baseline2},
    }
