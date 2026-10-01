from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..evaluators import as_key, get_counterfactual_evaluator
from ..options import DEFAULT_DEVICE, DEFAULT_TRANSFORMER
from ..utils import (
    PAIR_ROLES,
    _missing_input,
    _missing_keys,
    extract_protected_group,
    fairness_metadata,
    pair_key,
)

DEFAULT_COUNTERFACTUAL_METRICS = ["Cosine", "Rougel", "Bleu", "Sentiment Bias"]


def evaluate_counterfactual(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    opts = dict(options or {})
    missing = _missing_keys(opts, ["texts1", "texts2"])
    if missing:
        return _missing_input("counterfactual", missing)
    texts1 = opts.pop("texts1")
    texts2 = opts.pop("texts2")

    evaluator = get_counterfactual_evaluator(
        as_key(opts.pop("metrics", DEFAULT_COUNTERFACTUAL_METRICS)),
        bool(opts.pop("neutralizeTokens", True)),
        str(opts.pop("sentimentClassifier", "vader")),
        str(opts.pop("transformer", DEFAULT_TRANSFORMER)),
        str(opts.pop("device", DEFAULT_DEVICE)),
        str(opts.pop("how", "pairwise")),
    )
    return evaluator.evaluate(
        texts1=texts1,
        texts2=texts2,
        attribute=str(opts.pop("attribute", "gender")),
        return_data=bool(opts.pop("returnData", False)),
        show_progress_bars=bool(opts.pop("showProgressBars", False)),
    )


def evaluate_counterfactual_for_run(
    response_rows: Sequence[Any],
    options: Mapping[str, Any] | None = None,
    evaluate: Callable[..., dict[str, Any]] = evaluate_counterfactual,
) -> dict[str, Any]:
    opts = dict(options or {})
    opts.pop("language", None)
    if "texts1" not in opts or "texts2" not in opts:
        extracted = build_counterfactual_text_pairs(response_rows)
        if extracted is None:
            return _missing_input(
                "counterfactual",
                ["prompt.metadata.fairness.pair_id", "paired responses"],
            )
        opts = {**opts, **extracted}
    opts.pop("groups", None)
    return evaluate(options=opts)


def build_counterfactual_text_pairs(response_rows: Sequence[Any]) -> dict[str, Any] | None:
    """Pair original/counterfactual responses, bucketed by protected group.

    Pairs are keyed by (objective, pair_id, generation_index) -- see
    ``utils.pair_key`` -- so legacy populations whose pair ids restart per task
    objective are still paired correctly instead of overwriting each other.

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
    pairs: dict[tuple[str, str, int], dict[str, str]] = {}
    attribute = "gender"

    for response in response_rows:
        fairness = fairness_metadata(response)
        if fairness is None or fairness.get("task") != "counterfactual":
            continue

        pair_id = fairness.get("pair_id")
        if not pair_id or fairness.get("role") not in PAIR_ROLES:
            continue

        group = extract_protected_group(response)
        if not group:
            continue

        attribute = str(fairness.get("attribute") or attribute)
        key = pair_key(response, fairness, pair_id)
        pairs.setdefault(key, {})[group] = str(getattr(response, "response_text", ""))

    complete_pairs = [pair for pair in pairs.values() if len(pair) == 2]
    if not complete_pairs:
        return None

    # Fix the two protected groups to a stable, deterministic order (not the
    # order in which pairs happen to appear) so every pair contributes to the
    # same side of texts1/texts2 regardless of which one was "original".
    group_order = sorted({group for pair in complete_pairs for group in pair})
    if len(group_order) != 2:
        return None
    group_a, group_b = group_order

    texts1 = [pair[group_a] for pair in complete_pairs]
    texts2 = [pair[group_b] for pair in complete_pairs]
    return {"texts1": texts1, "texts2": texts2, "attribute": attribute, "groups": [group_a, group_b]}
