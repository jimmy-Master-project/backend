"""Identity words, persona eligibility and the group versions of a question.

Decisions (docs/論文構想/研究問題構想.md):
- gender: man / woman; non-binary personas are not part of the study;
- age: 35-54 = middle-aged, 55+ = elderly, younger personas are filtered out;
- race: the persona's own MatrAIx label, written as-is.

Every base question states its asker's identity once as an exact phrase,
e.g. "an elderly East Asian woman". Versions are built by replacing that
phrase: a dimension is either swapped to another value or removed (GenFair
MR1/MR2: removal of a sensitive attribute). With no gender the noun becomes
"person" ("a Caucasian person", "a person").
"""
from __future__ import annotations

import itertools
import re
from typing import Any

GENDER_KEY = "gender_identity"
AGE_KEY = "age_bracket"
RACE_KEY = "demo_ethnicity_broad"
IDENTITY_KEYS = (GENDER_KEY, AGE_KEY, RACE_KEY)

GENDER_WORDS = {"man": "man", "male": "man", "men": "man", "m": "man",
                "woman": "woman", "female": "woman", "women": "woman", "f": "woman"}
GENDERS = ("man", "woman")
AGES = ("middle-aged", "elderly")
MIDDLE_AGED_MIN, ELDERLY_MIN = 35, 55


def gender_word(value: Any) -> str | None:
    return GENDER_WORDS.get(str(value or "").strip().lower())


def age_word(value: Any) -> str | None:
    """'35-44' / '45-54' -> middle-aged; '55-64' / '65+' -> elderly; else None."""
    match = re.match(r"\s*(\d+)", str(value or ""))
    if not match:
        return None
    lower = int(match.group(1))
    if lower >= ELDERLY_MIN:
        return "elderly"
    if lower >= MIDDLE_AGED_MIN:
        return "middle-aged"
    return None


def race_word(value: Any) -> str | None:
    text = " ".join(str(value or "").split())
    return text or None


def persona_identity(attributes: dict[str, Any]) -> dict[str, str] | None:
    """The identity words of an eligible persona, or None if it is out of the study."""
    identity = {
        "gender": gender_word(attributes.get(GENDER_KEY)),
        "age": age_word(attributes.get(AGE_KEY)),
        "race": race_word(attributes.get(RACE_KEY)),
    }
    return identity if all(identity.values()) else None


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def core(gender: str | None = None, age: str | None = None, race: str | None = None) -> str:
    """The phrase without article: 'elderly East Asian woman', 'Caucasian person', 'person'."""
    return " ".join([w for w in (age, race) if w] + [gender or "person"])


def compose(gender: str | None = None, age: str | None = None, race: str | None = None) -> str:
    """'an elderly East Asian woman', 'a Caucasian person', 'a person'."""
    phrase = core(gender, age, race)
    return f"{article(phrase)} {phrase}"


def identity_core(identity: dict[str, str]) -> str:
    return core(identity["gender"], identity["age"], identity["race"])


def identity_phrase(identity: dict[str, str]) -> str:
    return compose(identity["gender"], identity["age"], identity["race"])


def _pattern(phrase_core: str) -> re.Pattern:
    # An optional a/an before the phrase is re-derived for the replacement.
    return re.compile(r"\b(?:(a|an)\s+)?" + re.escape(phrase_core) + r"\b", re.IGNORECASE)


def contains_phrase(text: str, phrase_core: str) -> bool:
    return _pattern(phrase_core).search(text) is not None


def replace_phrase(text: str, phrase_core: str, replacement_core: str) -> str:
    """Replace every occurrence, fixing a/an and keeping a capital first letter."""

    def repl(match: re.Match) -> str:
        found = match.group(0)
        new = f"{article(replacement_core)} {replacement_core}" if match.group(1) else replacement_core
        return new[:1].upper() + new[1:] if found[:1].isupper() else new

    return _pattern(phrase_core).sub(repl, text)


def version_code(gender: str | None, age: str | None, race: str | None) -> str:
    return f"G={gender or '-'}|A={age or '-'}|R={race or '-'}"


def version_specs(races: list[str]) -> list[dict[str, str | None]]:
    """Unmarked, single-dimension and two-dimension versions (9 + 5R of them)."""
    dims = {"gender": [None, *GENDERS], "age": [None, *AGES], "race": [None, *races]}
    specs = []
    for gender, age, race in itertools.product(dims["gender"], dims["age"], dims["race"]):
        if sum(v is not None for v in (gender, age, race)) <= 2:
            specs.append({"gender": gender, "age": age, "race": race})
    return specs


def build_versions(text: str, identity: dict[str, str], races: list[str]) -> list[dict[str, Any]]:
    phrase = identity_core(identity)
    versions = []
    for spec in version_specs(races):
        versions.append(
            {
                **spec,
                "code": version_code(spec["gender"], spec["age"], spec["race"]),
                "text": replace_phrase(text, phrase, core(spec["gender"], spec["age"], spec["race"])),
            }
        )
    return versions


def comparisons(races: list[str]) -> list[dict[str, Any]]:
    """Every comparison of 結果總覽表.md, as (version a, version b) codes.

    type: 性別 / 年齡 / 種族 (single dimension) or 固定X，比較Y (two dimensions).
    """
    values = {"性別": ("gender", list(GENDERS)), "年齡": ("age", list(AGES)), "種族": ("race", list(races))}
    labels = {"man": "男性", "woman": "女性", "middle-aged": "中年", "elderly": "老年"}

    def label(value: str | None) -> str:
        return "未明示" if value is None else labels.get(value, value)

    def code(spec: dict[str, str | None]) -> str:
        return version_code(spec.get("gender"), spec.get("age"), spec.get("race"))

    def pairs(options: list[str]) -> list[tuple[str | None, str]]:
        return [*itertools.combinations(options, 2), *((None, v) for v in options)]

    result = []
    for dim_name, (dim, options) in values.items():
        for a, b in pairs(options):
            result.append({
                "type": dim_name, "fixed": "", "comparison": f"{label(a)} vs. {label(b)}",
                "a": code({dim: a}), "b": code({dim: b}),
            })
    for fixed_name, compare_name in itertools.permutations(values, 2):
        fixed_dim, fixed_options = values[fixed_name]
        dim, options = values[compare_name]
        for fixed_value in fixed_options:
            for a, b in pairs(options):
                result.append({
                    "type": f"固定{fixed_name}，比較{compare_name}",
                    "fixed": f"{fixed_name}：{label(fixed_value)}",
                    "comparison": f"{label(fixed_value)}：{label(a)} vs. {label(b)}",
                    "a": code({fixed_dim: fixed_value, dim: a}),
                    "b": code({fixed_dim: fixed_value, dim: b}),
                })
    return result
