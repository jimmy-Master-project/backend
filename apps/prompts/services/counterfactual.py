# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingImports=false
import re
from typing import Any

from apps.evaluations.models import Evaluation
from apps.prompt_agents.services import PromptGenerationService


SUPPORTED_GENDER_COUNTERFACTUALS = {
    "male": "female",
    "female": "male",
}


def build_counterfactual_pair_id(pair_index: int, objective_index: int | None = None) -> str:
    """Pair ids must be unique within a whole population, not just per objective.

    Fairness pairing keys responses by pair_id, so restarting the counter for
    every task objective (the old ``cf_001`` scheme) made later objectives
    overwrite earlier ones.
    """
    if objective_index is None:
        return f"cf_{pair_index:03d}"
    return f"cf_o{objective_index:02d}_{pair_index:03d}"


def build_gender_counterfactual_prompts(
    evaluation: Evaluation,
    personas: list[dict[str, Any]],
    prompt_service: PromptGenerationService,
    provider: str,
    model_name: str,
    temperature: float,
    base_url: str | None,
    additional_instruction: str,
    fairness_task: str = "counterfactual",
    objective_index: int | None = None,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build original/counterfactual gender prompt pairs.

    The Prompt Generation Agent is called exactly once per persona (same as the
    plain persona-conditioned path), so the "original" prompt keeps the LLM's
    normal diverse, realistic wording. The "counterfactual" side is derived from
    that same generated text by swapping gendered words, not by asking the LLM
    to write it a second time independently -- two independent calls cannot
    guarantee the pair differs only in the protected attribute.

    Pairs whose swap produced identical text (no gendered word found) are
    dropped: they cannot reveal any gender effect and only dilute the metrics.
    The number dropped is accumulated into ``stats["identical_swap_skipped"]``.
    """
    eligible: list[tuple[dict[str, Any], str, str, str]] = []
    for persona in personas:
        attributes = persona.get("attributes") or {}
        gender_attribute = find_gender_attribute(attributes)
        gender = normalize_gender(attributes.get(gender_attribute) if gender_attribute else None)
        counterfactual_gender = SUPPORTED_GENDER_COUNTERFACTUALS.get(gender)
        if not counterfactual_gender:
            continue
        eligible.append((persona, gender_attribute or "gender_identity", gender, counterfactual_gender))

    if not eligible:
        return []

    generated = prompt_service.generate_for_personas(
        evaluation=evaluation,
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        personas=[persona for persona, *_ in eligible],
        base_url=base_url,
        additional_instruction=additional_instruction,
    )

    prompts: list[dict[str, Any]] = []
    skipped_identical = 0
    pair_index = 0
    for persona_index, ((persona, source_attribute, gender, counterfactual_gender), item) in enumerate(
        zip(eligible, generated, strict=False), start=1
    ):
        original_text = str(item["text"])
        counterfactual_text = swap_gender_words(original_text, gender, counterfactual_gender)
        if counterfactual_text == original_text:
            skipped_identical += 1
            continue

        pair_index += 1
        pair_id = build_counterfactual_pair_id(pair_index, objective_index)
        source_persona_id = str(persona.get("persona_id") or f"persona_{persona_index:03d}")
        attributes = persona.get("attributes") or {}
        missing_attributes = persona.get("missing_attributes", [])
        base_fairness: dict[str, Any] = {
            "task": fairness_task,
            "attribute": "gender",
            "source_attribute": source_attribute,
            "pair_id": pair_id,
        }
        if objective_index is not None:
            base_fairness["objective_index"] = objective_index

        prompts.append(
            {
                "text": original_text,
                "category": evaluation.task_type,
                "persona_id": source_persona_id,
                "persona_attributes": attributes,
                "missing_attributes": missing_attributes,
                "fairness": {**base_fairness, "group": gender, "role": "original"},
            }
        )
        prompts.append(
            {
                "text": counterfactual_text,
                "category": evaluation.task_type,
                "persona_id": f"{source_persona_id}_cf_gender_{counterfactual_gender}",
                "persona_attributes": {
                    **attributes,
                    source_attribute: format_counterfactual_gender(counterfactual_gender),
                },
                "missing_attributes": missing_attributes,
                "fairness": {
                    **base_fairness,
                    "group": counterfactual_gender,
                    "role": "counterfactual",
                    "source_persona_id": source_persona_id,
                },
            }
        )

    if stats is not None:
        stats["identical_swap_skipped"] = stats.get("identical_swap_skipped", 0) + skipped_identical
    return prompts


def _case_preserving(match: Any, replacement: str) -> str:
    original = match.group(0)
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


# Whole-word pairs. Matching uses (?<!\w)...(?!\w) lookarounds instead of \b so
# entries ending in punctuation ("mrs.") still match before a space ("Mrs. Lee").
# All pairs are applied in a single regex pass, so a replacement can never be
# re-swapped by a later pair.
_ENGLISH_GENDER_WORD_PAIRS = {
    "female_to_male": (
        ("grandmother", "grandfather"),
        ("granddaughter", "grandson"),
        ("stepmother", "stepfather"),
        ("businesswoman", "businessman"),
        ("saleswoman", "salesman"),
        ("chairwoman", "chairman"),
        ("policewoman", "policeman"),
        ("mother", "father"),
        ("daughter", "son"),
        ("sister", "brother"),
        ("actress", "actor"),
        ("waitress", "waiter"),
        ("herself", "himself"),
        ("hers", "his"),
        ("she", "he"),
        ("woman", "man"),
        ("women", "men"),
        ("girls", "boys"),
        ("girl", "boy"),
        ("wife", "husband"),
        ("ladies", "gentlemen"),
        ("lady", "gentleman"),
        ("aunt", "uncle"),
        ("niece", "nephew"),
        ("her", "his"),  # resolved to his/him contextually in _swap_her
        ("mrs.", "mr."),
        ("ms.", "mr."),
        ("mrs", "mr"),
        ("madam", "sir"),
        ("female", "male"),
    ),
    "male_to_female": (
        ("grandfather", "grandmother"),
        ("grandson", "granddaughter"),
        ("stepfather", "stepmother"),
        ("businessman", "businesswoman"),
        ("salesman", "saleswoman"),
        ("chairman", "chairwoman"),
        ("policeman", "policewoman"),
        ("father", "mother"),
        ("son", "daughter"),
        ("brother", "sister"),
        ("actor", "actress"),
        ("waiter", "waitress"),
        ("himself", "herself"),
        ("his", "her"),
        ("him", "her"),
        ("he", "she"),
        ("man", "woman"),
        ("men", "women"),
        ("boys", "girls"),
        ("boy", "girl"),
        ("husband", "wife"),
        ("gentlemen", "ladies"),
        ("gentleman", "lady"),
        ("uncle", "aunt"),
        ("nephew", "niece"),
        ("mr.", "ms."),
        ("mr", "ms"),
        ("sir", "madam"),
        ("male", "female"),
    ),
}



# Words that, when they directly follow "her", indicate object use ("gave her
# the book", "told her about", "made her feel") rather than possessive ("her
# career"). Anything else that looks like a word is treated as a noun/adjective
# (possessive). No POS tagger (spaCy) is a dependency, so this is a heuristic.
_HER_OBJECT_FOLLOWERS = frozenset(
    """
    a an the this that these those some any every each no all both either neither
    my your his its our their me you him us them it one someone something anyone anything
    to for with from about in on at by of into onto upon over under through across
    around after before behind beside between toward towards without within against
    during since until till via per like near off out up down away back again
    and or but nor so yet because if when while as than though although unless
    whether where why how what which who whom whose
    too very really just also even still already now then today tonight tomorrow
    yesterday soon later here there well much more most less least enough
    is was are were be been am will would can could should may might must shall do does did
    not never always often sometimes
    feel go do be know think make take come see get stay leave try understand decide
    choose become find keep let give tell ask help say
    """.split()
)


def _swap_her(match: re.Match[str]) -> str:
    following = match.string[match.end():]
    next_word = re.match(r"\s+([A-Za-z][\w'-]*)", following)
    if next_word is None:
        # End of text, punctuation, or a non-word token -> object pronoun.
        return _case_preserving(match, "him")
    word = next_word.group(1).lower()
    if word in _HER_OBJECT_FOLLOWERS or (word.endswith("ly") and word not in {"family", "only"}):
        return _case_preserving(match, "him")
    return _case_preserving(match, "his")


def _swap_his(match: re.Match[str]) -> str:
    # Predicative "his" ("the book is his.") -> "hers"; otherwise possessive "her".
    following = match.string[match.end():]
    if re.match(r"\s+[A-Za-z]", following) is None:
        return _case_preserving(match, "hers")
    return _case_preserving(match, "her")


def _build_swap_pattern(pairs: tuple[tuple[str, str], ...]) -> re.Pattern[str]:
    alternatives = sorted((source for source, _ in pairs), key=len, reverse=True)
    body = "|".join(re.escape(source) for source in alternatives)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


_SWAP_TABLES = {
    direction: (dict(pairs), _build_swap_pattern(pairs))
    for direction, pairs in _ENGLISH_GENDER_WORD_PAIRS.items()
}


def swap_gender_words(text: str, from_group: str, to_group: str) -> str:
    """Best-effort swap of gendered words in free-form LLM-generated text.

    This is a deliberately simple rule-based pass (natural language is
    unstable), covering common English gendered nouns, titles and pronouns.
    English "her" is ambiguous between possessive ("her career") and object
    ("gave her the book") use; see ``_swap_her`` for the heuristic.
    """
    if from_group == to_group or from_group not in {"male", "female"} or to_group not in {"male", "female"}:
        return text

    direction = "female_to_male" if from_group == "female" else "male_to_female"
    mapping, pattern = _SWAP_TABLES[direction]

    def replace(match: re.Match[str]) -> str:
        word = match.group(0).lower()
        if direction == "female_to_male" and word == "her":
            return _swap_her(match)
        if direction == "male_to_female" and word == "his":
            return _swap_his(match)
        return _case_preserving(match, mapping[word])

    return pattern.sub(replace, text)



def has_gender_attribute(selected_attributes: list[str]) -> bool:
    return get_selected_gender_attribute(selected_attributes) is not None


def get_selected_gender_attribute(selected_attributes: list[str]) -> str | None:
    for key in selected_attributes:
        if key in {"gender_identity", "gender"}:
            return key
    return None


def find_gender_attribute(attributes: dict[str, Any]) -> str | None:
    for key in ("gender_identity", "gender"):
        if key in attributes:
            return key
    return None


def format_counterfactual_gender(counterfactual_group: str) -> str:
    if counterfactual_group == "male":
        return "Man"
    if counterfactual_group == "female":
        return "Woman"
    return counterfactual_group


def normalize_gender(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"m", "man", "men", "male"}:
        return "male"
    if normalized in {"f", "woman", "women", "female"}:
        return "female"
    return normalized
