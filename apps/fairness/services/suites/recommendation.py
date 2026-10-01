from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..evaluators import as_key, get_recommendation_evaluator
from ..utils import PAIR_ROLES, _missing_input, _missing_keys, fairness_metadata, pair_key

DEFAULT_RECOMMENDATION_METRICS = ["Jaccard", "PRAG", "SERP"]


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
    metrics = evaluate(options=metric_opts)
    status_value = metrics.get("status") if isinstance(metrics, Mapping) else None
    return {
        "status": status_value or "ok",
        "inputs": extracted["inputs"],
        "extractionFailures": extracted["extractionFailures"],
        "commonListLength": extracted.get("commonListLength"),
        "metrics": metrics,
    }


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
        pairs.setdefault(pair_key(response, fairness, pair_id), {})[str(role)] = items

    rec_lists1: list[list[str]] = []
    rec_lists2: list[list[str]] = []
    for pair in pairs.values():
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

    return {
        "inputs": {"mode": "pairwise", "recLists1": rec_lists1, "recLists2": rec_lists2},
        "extractionFailures": failures,
        "commonListLength": global_length,
    }
