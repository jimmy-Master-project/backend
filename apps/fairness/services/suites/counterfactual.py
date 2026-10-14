from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Any

from ..evaluators import as_key, get_counterfactual_evaluator
from ..options import DEFAULT_DEVICE, DEFAULT_TRANSFORMER
from ..stats import difference_with_ci, mean_with_ci, paired_wasserstein_with_ci
from ..utils import (
    PAIR_ROLES,
    _missing_input,
    _missing_keys,
    extract_protected_group,
    fairness_metadata,
    pair_key,
    prompt_text,
)

DEFAULT_COUNTERFACTUAL_METRICS = ["Cosine", "Rougel", "Bleu", "Sentiment Bias"]


def evaluate_counterfactual(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    opts = dict(options or {})
    missing = _missing_keys(opts, ["texts1", "texts2"])
    if missing:
        return _missing_input("counterfactual", missing)
    texts1 = opts.pop("texts1")
    texts2 = opts.pop("texts2")
    # Cluster label per pair (e.g. the prompt pair all its generations share).
    clusters = opts.pop("clusters", None)

    evaluator = get_counterfactual_evaluator(
        as_key(opts.pop("metrics", DEFAULT_COUNTERFACTUAL_METRICS)),
        bool(opts.pop("neutralizeTokens", True)),
        str(opts.pop("sentimentClassifier", "vader")),
        str(opts.pop("transformer", DEFAULT_TRANSFORMER)),
        str(opts.pop("device", DEFAULT_DEVICE)),
        str(opts.pop("how", "pairwise")),
    )
    return_data = bool(opts.pop("returnData", False))
    result = evaluator.evaluate(
        texts1=texts1,
        texts2=texts2,
        attribute=str(opts.pop("attribute", "gender")),
        # Pair-level scores are always needed for the confidence intervals.
        return_data=True,
        show_progress_bars=bool(opts.pop("showProgressBars", False)),
    )
    data = result.pop("data", None) or {}
    result["pairCount"] = len(texts1)
    result["confidenceIntervals"] = _confidence_intervals(evaluator, data, texts1, texts2, clusters)
    result["ciUnit"] = "prompt pair (all generations together)" if clusters is not None else "response pair"
    if return_data:
        result["data"] = data
    return result


def _confidence_intervals(
    evaluator: Any,
    data: Mapping[str, Any],
    texts1: Sequence[str],
    texts2: Sequence[str],
    clusters: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Bootstrap CIs over pairs (or clusters of pairs) for every metric with pair-level data."""
    intervals: dict[str, Any] = {}
    for name, values in data.items():
        if name in {"texts1", "texts2"}:
            continue
        if name == "Sentiment Bias":
            interval = _sentiment_bias_interval(evaluator, texts1, texts2, clusters)
            if interval is not None:
                intervals[name] = interval
        elif isinstance(values, list | tuple):
            intervals[name] = mean_with_ci(values, clusters)
    return intervals


def _sentiment_bias_interval(
    evaluator: Any, texts1: Sequence[str], texts2: Sequence[str], clusters: Sequence[Any] | None = None
) -> dict[str, Any] | None:
    """CI of strong-parity Sentiment Bias (Wasserstein-1), resampling pairs.

    LangFair only returns per-pair absolute differences, so the two groups'
    sentiment scores are recomputed with the same classifier object.
    """
    metric = next((m for m in getattr(evaluator, "metrics", []) if getattr(m, "name", "") == "Sentiment Bias"), None)
    if metric is None or getattr(metric, "parity", "strong") != "strong" or not hasattr(metric, "_get_sentiment_scores"):
        return None
    scores1, scores2 = metric._get_sentiment_scores(list(texts1), list(texts2))
    return paired_wasserstein_with_ci(scores1, scores2, clusters)


def evaluate_counterfactual_for_run(
    response_rows: Sequence[Any],
    options: Mapping[str, Any] | None = None,
    evaluate: Callable[..., dict[str, Any]] = evaluate_counterfactual,
) -> dict[str, Any]:
    opts = dict(options or {})
    opts.pop("language", None)
    from_run = "texts1" not in opts or "texts2" not in opts
    if from_run:
        extracted = build_counterfactual_text_pairs(response_rows)
        if extracted is None:
            return _missing_input(
                "counterfactual",
                ["prompt.metadata.fairness.pair_id", "paired responses"],
            )
        opts = {**opts, **extracted}
    groups = opts.pop("groups", None)
    excluded_identical = opts.pop("excludedIdenticalPromptPairs", 0)
    return_data = bool(opts.pop("returnData", False))
    # Pair-level data is needed for the baseline comparison; dropped below
    # unless the caller asked for it.
    result = evaluate(options={**opts, "returnData": True})
    if not isinstance(result, dict) or result.get("status") == "missing_input":
        return result
    if groups:
        result["groups"] = list(groups)
        result["comparison"] = f"texts1 = {groups[0]}, texts2 = {groups[1]}"
    result["excludedIdenticalPromptPairs"] = excluded_identical

    baseline = build_counterfactual_noise_pairs(response_rows) if from_run else None
    if baseline is not None:
        baseline_opts = {key: value for key, value in opts.items() if key not in {"texts1", "texts2", "clusters"}}
        baseline_result = evaluate(options={**baseline_opts, **baseline, "returnData": True})
        if isinstance(baseline_result, dict):
            baseline_data = baseline_result.pop("data", None) or {}
            result["noiseBaseline"] = {
                "description": (
                    "Same metrics between different generations of the same prompt (same group). "
                    "Cross-group values only indicate bias when clearly worse than this baseline."
                ),
                **baseline_result,
            }
            gap = _baseline_gap(result.get("data") or {}, opts.get("clusters"), baseline_data, baseline["clusters"])
            if gap:
                result["baselineGap"] = {
                    "description": (
                        "Cross-group similarity minus the same-prompt noise baseline, on the prompt pairs that "
                        "have both. A negative value whose CI excludes 0 means responses differ across groups "
                        "more than sampling noise explains."
                    ),
                    "metrics": gap,
                }
    if not return_data:
        result.pop("data", None)
    return result


# Pair-level similarity metrics (higher = more alike) that can be compared to the baseline.
SIMILARITY_METRICS = ("Cosine Similarity", "RougeL Similarity", "Bleu Similarity")


def _baseline_gap(
    cross_data: Mapping[str, Any],
    cross_clusters: Sequence[Any] | None,
    baseline_data: Mapping[str, Any],
    baseline_clusters: Sequence[Any],
) -> dict[str, Any]:
    if cross_clusters is None:
        return {}
    gap: dict[str, Any] = {}
    for name in SIMILARITY_METRICS:
        cross, base = cross_data.get(name), baseline_data.get(name)
        if isinstance(cross, list | tuple) and isinstance(base, list | tuple):
            gap[name] = difference_with_ci(cross, cross_clusters, base, baseline_clusters)
    return gap


def build_counterfactual_noise_pairs(response_rows: Sequence[Any]) -> dict[str, Any] | None:
    """Pair different generations of the same counterfactual prompt (noise floor).

    Two generations of one prompt differ only by sampling randomness, so their
    similarity is the best cross-group similarity one could expect from an
    unbiased model.
    """
    generations: dict[tuple[str, str, str], dict[int, str]] = {}
    for response in response_rows:
        fairness = fairness_metadata(response)
        if fairness is None or fairness.get("task") != "counterfactual":
            continue
        pair_id = fairness.get("pair_id")
        role = fairness.get("role")
        if not pair_id or role not in PAIR_ROLES:
            continue
        objective, pair, generation_index = pair_key(response, fairness, pair_id)
        generations.setdefault((objective, pair, str(role)), {})[generation_index] = str(
            getattr(response, "response_text", "")
        )

    texts1: list[str] = []
    texts2: list[str] = []
    clusters: list[tuple[str, str]] = []
    for (objective, pair, _role), by_index in generations.items():
        for first, second in combinations(sorted(by_index), 2):
            texts1.append(by_index[first])
            texts2.append(by_index[second])
            # Same cluster as the cross-group pairs of this prompt pair.
            clusters.append((objective, pair))
    if not texts1:
        return None
    return {"texts1": texts1, "texts2": texts2, "clusters": clusters}


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
    pair_prompts: dict[tuple[str, str, int], set[str | None]] = {}
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
        pair_prompts.setdefault(key, set()).add(prompt_text(response))

    complete = {key: pair for key, pair in pairs.items() if len(pair) == 2}
    # A pair whose two prompts are textually identical means the attribute
    # swap failed; any response difference is pure sampling noise.
    identical = {key for key in complete if len(pair_prompts[key]) == 1 and None not in pair_prompts[key]}
    complete_keys = [key for key in complete if key not in identical]
    complete_pairs = [complete[key] for key in complete_keys]
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
    # Every generation of one prompt pair shares that pair's prompts, so the
    # generations are not independent: they form one bootstrap cluster.
    clusters = [(objective, pair_id) for (objective, pair_id, _generation) in complete_keys]
    return {
        "texts1": texts1,
        "texts2": texts2,
        "clusters": clusters,
        "attribute": attribute,
        "groups": [group_a, group_b],
        "excludedIdenticalPromptPairs": len(identical),
    }
