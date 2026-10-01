"""camelCase helpers for API payloads.

The API speaks camelCase; models and internal JSON (``metadata`` blobs) stay
snake_case. Serializers declare camelCase fields explicitly; this helper is
for the few hand-built responses that expose a fixed-schema internal dict.
Never apply it to data-keyed dicts (persona attributes, filters, LangFair
output): their keys are data, not field names.
"""
import re
from typing import Any

_SNAKE_SEGMENT = re.compile(r"_([a-z0-9])")


def to_camel(key: str) -> str:
    return _SNAKE_SEGMENT.sub(lambda match: match.group(1).upper(), key)


def camelize_keys(value: Any, keep: frozenset[str] | set[str] = frozenset()) -> Any:
    """Recursively camelCase dict keys; values under keys in ``keep`` pass through untouched."""
    if isinstance(value, dict):
        return {
            to_camel(str(key)): item if key in keep else camelize_keys(item, keep)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [camelize_keys(item, keep) for item in value]
    return value
