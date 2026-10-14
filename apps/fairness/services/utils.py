"""Shared constants and small helpers for LangFair metric computation."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

SUPPORTED_SUITES = [
    "toxicity",
    "stereotype",
    "classification",
    "counterfactual",
    "recommendation",
]
DEFAULT_SUITES = ["toxicity", "stereotype", "counterfactual"]
SUPPORTED_LANGUAGES = {"en", "auto"}
TASK_TYPE_DEFAULT_SUITES = {
    "text_generation": ["toxicity", "stereotype", "counterfactual"],
    "classification": ["classification"],
    "recommendation": ["recommendation"],
    "other": ["toxicity", "stereotype"],
}
PAIR_ROLES = {"original", "counterfactual"}

# LangFair's models and word lists only understand English (detoxify_unbiased,
# the stereotype classifier and word lists, VADER, rouge_score's tokenizer,
# all-MiniLM-L6-v2), so a run containing any non-English response is refused
# as a whole -- every selected suite, including classification and
# recommendation, so results never mix languages.
# A response counts as English when at least this share of its letters
# (str.isalpha, which includes CJK characters) are ASCII. Punctuation, digits
# and symbols are ignored so curly quotes / em dashes in English text and
# ASCII markdown or numbers in Chinese text do not skew the ratio.
ENGLISH_ASCII_RATIO = 0.9
MAX_REPORTED_NON_ENGLISH_IDS = 20


# Toxicity and stereotype classifiers read at most 512 tokens and silently
# drop the rest, while LLM answers often run to thousands of characters, so a
# response is scored in parts and takes its highest part score:
# - toxicity per sentence (a toxic sentence is toxic on its own);
# - stereotype per chunk of whole sentences that fits the model: scored per
#   sentence, the detector fires on short fragments such as "She could say:".
TOXICITY_SCORING_UNIT = "sentence (response score = highest sentence score)"
STEREOTYPE_SCORING_UNIT = "chunk of whole sentences up to 480 tokens (response score = highest chunk score)"
STEREOTYPE_CHUNK_TOKENS = 480
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: Any) -> list[str]:
    """Sentences of a response (line breaks also end a sentence); never empty."""
    parts = [part.strip() for part in _SENTENCE_BREAK.split(str(text or ""))]
    sentences = [part for part in parts if part]
    return sentences or [str(text or "")]


def split_chunks(text: Any, count_tokens: Any, budget: int) -> list[str]:
    """Consecutive whole sentences packed into chunks of at most ``budget`` tokens.

    A single sentence longer than the budget becomes its own chunk (the
    classifier then truncates just that sentence).
    """
    chunks: list[str] = []
    current: list[str] = []
    used = 0
    for sentence in split_sentences(text):
        size = count_tokens(sentence)
        if current and used + size > budget:
            chunks.append(" ".join(current))
            current, used = [], 0
        current.append(sentence)
        used += size
    if current:
        chunks.append(" ".join(current))
    return chunks or [str(text or "")]


def max_per_response(responses: Sequence[Any], score_sentences: Any, split: Any = None) -> list[float]:
    """Score every part (sentence by default) in one batch, then take each response's maximum."""
    split = split or split_sentences
    owners: list[int] = []
    sentences: list[str] = []
    for index, response in enumerate(responses):
        for sentence in split(response):
            owners.append(index)
            sentences.append(sentence)
    scores = list(score_sentences(sentences)) if sentences else []
    best = [0.0] * len(responses)
    for owner, score in zip(owners, scores, strict=True):
        best[owner] = max(best[owner], float(score))
    return best


def get_default_suites_for_run(run: Any) -> list[str]:
    task_type = str(getattr(run.evaluation, "task_type", "other"))
    return TASK_TYPE_DEFAULT_SUITES.get(task_type, TASK_TYPE_DEFAULT_SUITES["other"])


def normalize_langfair_language(value: Any = None, responses: Sequence[str] = ()) -> str:
    language = str(value or "en")
    if language == "auto":
        return "en"
    if language not in SUPPORTED_LANGUAGES - {"auto"}:
        raise ValueError("language must be 'en' or 'auto'.")
    return language


def ascii_ratio(text: str) -> float | None:
    """Share of letters that are ASCII (None when the text has no letters)."""
    chars = [char for char in str(text) if char.isalpha()]
    if not chars:
        return None
    return sum(1 for char in chars if ord(char) < 128) / len(chars)


def check_english_responses(response_rows: Sequence[Any]) -> dict[str, Any]:
    """Classify every response as English or not by its ASCII character ratio."""
    non_english_ids: list[Any] = []
    checked = 0
    for row in response_rows:
        ratio = ascii_ratio(getattr(row, "response_text", ""))
        if ratio is None:
            continue
        checked += 1
        if ratio < ENGLISH_ASCII_RATIO:
            non_english_ids.append(getattr(row, "id", None))
    return {
        "isEnglish": not non_english_ids,
        "method": "ascii_ratio",
        "asciiRatioThreshold": ENGLISH_ASCII_RATIO,
        "checkedResponseCount": checked,
        "nonEnglishResponseCount": len(non_english_ids),
        "nonEnglishResponseIds": non_english_ids[:MAX_REPORTED_NON_ENGLISH_IDS],
    }


def unsupported_language(suite: str, language_check: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "unsupported_language",
        "suite": suite,
        "message": (
            f"LangFair {suite} metrics were not computed: "
            f"{language_check['nonEnglishResponseCount']} of {language_check['checkedResponseCount']} "
            "responses are not English (ASCII letter ratio below "
            f"{language_check['asciiRatioThreshold']}), so the whole run is refused. "
            "LangFair only supports English; scoring other languages (e.g. Chinese) yields misleading values."
        ),
        "languageCheck": dict(language_check),
    }


def prompt_metadata(response: Any) -> Mapping[str, Any]:
    prompt = getattr(response, "prompt", None)
    metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
    return metadata if isinstance(metadata, Mapping) else {}


def fairness_metadata(response: Any) -> Mapping[str, Any] | None:
    fairness = prompt_metadata(response).get("fairness")
    return fairness if isinstance(fairness, Mapping) else None


def prompt_text(response: Any) -> str | None:
    prompt = getattr(response, "prompt", None)
    text = getattr(prompt, "text", None) if prompt is not None else None
    return None if text is None else str(text)


def pair_key(response: Any, fairness: Mapping[str, Any], pair_id: Any) -> tuple[str, str, int]:
    """Key a counterfactual pair member as (objective, pair_id, generation_index).

    Older prompt populations restarted pair ids (``cf_001``...) for every task
    objective, so pair_id alone collides across objectives. The objective is
    taken from ``fairness.objective_index`` (new populations) or the prompt's
    ``task_objective`` metadata (all populations generated per objective).
    """
    objective = fairness.get("objective_index")
    if objective is None:
        objective = prompt_metadata(response).get("task_objective") or ""
    generation_index = _safe_int(getattr(response, "generation_index", 1), 1)
    return (str(objective), str(pair_id), generation_index)


def extract_protected_group(response: Any) -> str | None:
    metadata = prompt_metadata(response)
    if not metadata:
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


def selected_protected_attributes(run: Any) -> list[str]:
    """LangFair attributes ('gender', 'race') the run's population was diagnosed on.

    Comes from the population's persona config (the fairness labels the user
    picked). Populations without one -- or with only non-LangFair attributes
    from before labels were restricted -- fall back to gender, the attribute
    every counterfactual pair carries.
    """
    from apps.personas.protected_attributes import langfair_attribute

    population = getattr(run, "prompt_population", None)
    metadata = getattr(population, "metadata", None)
    persona = metadata.get("persona") if isinstance(metadata, Mapping) else None
    keys = persona.get("selected_attributes") if isinstance(persona, Mapping) else None
    attributes: list[str] = []
    for key in keys or []:
        attribute = langfair_attribute(key)
        if attribute and attribute not in attributes:
            attributes.append(attribute)
    return attributes or ["gender"]


def extract_attribute_group(response: Any, attribute: str) -> str | None:
    """The response's LangFair group for one attribute ('gender' or 'race').

    Gender keeps the counterfactual pair's group (it is what was swapped into
    the prompt); every other attribute comes from the source persona.
    """
    if attribute == "gender":
        return extract_protected_group(response)
    from apps.personas.protected_attributes import PROTECTED_ATTRIBUTES, langfair_group

    persona_attributes = prompt_metadata(response).get("persona_attributes")
    if not isinstance(persona_attributes, Mapping):
        return None
    for key, spec in PROTECTED_ATTRIBUTES.items():
        if spec["attribute"] == attribute:
            return langfair_group(key, persona_attributes.get(key))
    return None


def normalize_gender_group(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"m", "man", "men", "male", "男", "男性"}:
        return "male"
    if normalized in {"f", "woman", "women", "female", "女", "女性"}:
        return "female"
    return normalized or None


def _missing_keys(options: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key not in options]


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


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas values to plain JSON types; NaN/inf become None.

    DRF renders JSON strictly (allow_nan=False), so a single NaN metric (e.g. a
    group with no positives) would otherwise turn the whole response into a 500.
    """
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, str | bool | int) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item"):
        return _to_jsonable(value.item())
    return value
