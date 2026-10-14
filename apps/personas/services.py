# pyright: reportMissingImports=false
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from django.conf import settings

from services.matraix_client import MatrAIxClient, MatrAIxUnavailableError

# Dimensions shown on a persona card built from a single-persona lookup.
CARD_SUMMARY_KEYS = ("age_bracket", "region", "domain", "life_stage")


class PersonaServiceError(Exception):
    pass


class PersonaService:
    def __init__(self, client: MatrAIxClient | None = None):
        self.client = client or MatrAIxClient()

    def get_attributes(self, pool: str | None = None) -> dict[str, Any]:
        selected_pool = pool or settings.MATRAIX_DEFAULT_PERSONA_POOL
        catalog = self.client.get_dimension_catalog(selected_pool)
        return {
            "pool": selected_pool,
            "attributes": self._extract_attributes(catalog),
            "catalog": catalog,
        }

    def sample_projected_personas(
        self,
        sample_size: int,
        selected_attributes: list[str],
        pool: str | None = None,
        filters: dict[str, Any] | None = None,
        stratify: Any | None = None,
        seed: int | None = None,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        selected_pool = pool or settings.MATRAIX_DEFAULT_PERSONA_POOL
        active_filters = filters or {}
        sample_payloads: list[dict[str, Any]] = []
        persona_ids: list[str] = []
        personas: list[dict[str, Any]] = []
        seen: set[str] = set()

        # The sampler can return personas that do not contain every selected
        # (diagnostic) attribute. Oversample and filter out personas missing
        # any of them so a request for N gender-conditioned prompts does not
        # become fewer prompts just because some candidates are missing
        # gender_identity -- selected_attributes only gates inclusion here, it
        # no longer limits which fields end up in `attributes`.
        for attempt in range(8):
            remaining = sample_size - len(personas)
            if remaining <= 0:
                break
            attempt_seed = None if seed is None else seed + attempt
            request_size = max(remaining * 3, remaining)
            sample_payload = self.client.sample_personas(
                sample_size=request_size,
                pool=selected_pool,
                filters=active_filters,
                stratify=stratify,
                seed=attempt_seed,
            )
            sample_payloads.append(sample_payload)
            new_ids = self._extract_persona_ids(sample_payload)
            for persona_id in new_ids:
                if persona_id in seen:
                    continue
                seen.add(persona_id)
                persona_ids.append(persona_id)
                persona = self.get_projected_persona(persona_id, selected_attributes, selected_pool)
                if not persona.get("missing_attributes"):
                    personas.append(persona)
                    if len(personas) >= sample_size:
                        break
            if not new_ids:
                break

        if not persona_ids:
            raise PersonaServiceError("MatrAIx sample response did not contain persona IDs.")
        if len(personas) < sample_size and not allow_partial:
            raise PersonaServiceError(
                f"Only {len(personas)} personas contain all selected attributes for the requested sample size {sample_size}. "
                "Try a larger persona pool, loosen filters, or select attributes with broader coverage."
            )
        return {
            "pool": selected_pool,
            "sample_size": sample_size,
            "selected_attributes": selected_attributes,
            "filters": active_filters,
            "stratify": stratify,
            "seed": seed,
            "matraix_sample": sample_payloads[0] if len(sample_payloads) == 1 else {"attempts": sample_payloads},
            "personas": personas,
            "partial": len(personas) < sample_size,
        }

    def list_personas(
        self,
        limit: int,
        offset: int = 0,
        pool: str | None = None,
        persona_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """A page of personas for manual selection (summary card per persona).

        With `persona_ids`, returns the cards of exactly those personas (e.g.
        the ones picked for a previous population) instead of a page.
        """
        selected_pool = pool or settings.MATRAIX_DEFAULT_PERSONA_POOL
        if persona_ids:
            # MatrAIx refuses id previews on large pools, so cards are built
            # from single-persona lookups instead.
            personas = self._cards_for_ids(persona_ids, selected_pool)
            return {
                "pool": selected_pool,
                "offset": 0,
                "limit": len(persona_ids),
                "total": None,
                "personas": personas,
            }
        payload = self.client.list_personas(limit=limit, offset=offset, pool=selected_pool)
        cards = payload.get("personas") or []
        personas = [
            {
                "persona_id": str(card.get("personaId") or card.get("persona_id") or card.get("id")),
                "name": card.get("name") or "",
                "source": card.get("source") or "",
                "summary": card.get("dimensions") if isinstance(card.get("dimensions"), dict) else {},
                # Flat "key value ..." text of every attribute; used for search.
                "search_text": card.get("searchText") or "",
            }
            for card in cards
            if isinstance(card, dict)
        ]
        return {
            "pool": selected_pool,
            "offset": offset,
            "limit": limit,
            # MatrAIx browses a fixed preview subset of the pool.
            "total": payload.get("previewCount"),
            "personas": personas,
        }

    def _cards_for_ids(self, persona_ids: list[str], pool: str) -> list[dict[str, Any]]:
        def card(persona_id: str) -> dict[str, Any] | None:
            try:
                payload = self.client.get_persona(persona_id, pool)
            except MatrAIxUnavailableError:
                return None  # Unknown / removed persona: just leave it out.
            dimensions = self._extract_dimensions(payload)
            name = payload.get("name") or ""
            source = payload.get("source") or ""
            pairs = " ".join(f"{key} {value}" for key, value in dimensions.items())
            return {
                "persona_id": self._extract_persona_id(payload) or persona_id,
                "name": name,
                "source": source,
                "summary": {key: dimensions[key] for key in CARD_SUMMARY_KEYS if key in dimensions},
                "search_text": f"{persona_id} {name} {source} {pairs}",
            }

        with ThreadPoolExecutor(max_workers=8) as pool_executor:
            cards = list(pool_executor.map(card, persona_ids))
        return [item for item in cards if item is not None]

    def get_personas_by_ids(
        self,
        persona_ids: list[str],
        selected_attributes: list[str],
        pool: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hand-picked personas (in the given order, duplicates dropped)."""
        selected_pool = pool or settings.MATRAIX_DEFAULT_PERSONA_POOL
        unique_ids = list(dict.fromkeys(str(persona_id) for persona_id in persona_ids))
        return [
            self.get_projected_persona(persona_id, selected_attributes, selected_pool)
            for persona_id in unique_ids
        ]

    def get_projected_persona(
        self,
        persona_id: str,
        selected_attributes: list[str],
        pool: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a persona's full attribute set from MatrAIx.

        `selected_attributes` no longer projects (discards) fields from the
        result -- every dimension MatrAIx returns for this persona is kept and
        passed on to prompt generation. Handing the generator only the
        diagnostic attribute(s) starves it of any other context, making that
        attribute the sole thing it has to write about and amplifying
        generation-stage bias instead of isolating it. `selected_attributes`
        now serves one purpose: naming which attribute(s) this population is
        being diagnosed on (e.g. gender_identity for the gender counterfactual
        pipeline) -- personas missing any of them are still excluded so the
        diagnostic attribute is guaranteed to be present.
        """
        selected_pool = pool or settings.MATRAIX_DEFAULT_PERSONA_POOL
        payload = self.client.get_persona(persona_id, selected_pool)
        dimensions = self._extract_dimensions(payload)
        missing_attributes = [
            key for key in selected_attributes if key not in dimensions
        ]
        return {
            "persona_id": self._extract_persona_id(payload) or persona_id,
            "attributes": dict(dimensions),
            "missing_attributes": missing_attributes,
        }

    def _extract_persona_ids(self, payload: dict[str, Any]) -> list[str]:
        for key in ("persona_ids", "personaIds", "ids"):
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item) for item in value if item is not None]

        personas = payload.get("personas") or payload.get("results") or payload.get("data")
        if isinstance(personas, list):
            ids: list[str] = []
            for item in personas:
                if isinstance(item, dict):
                    persona_id = self._extract_persona_id(item)
                    if persona_id:
                        ids.append(persona_id)
                elif item is not None:
                    ids.append(str(item))
            return ids

        return []

    def _extract_persona_id(self, payload: dict[str, Any]) -> str | None:
        for key in ("persona_id", "personaId", "id"):
            value = payload.get(key)
            if value is not None:
                return str(value)
        return None

    def _extract_dimensions(self, payload: dict[str, Any]) -> dict[str, Any]:
        dimensions = payload.get("dimensions") or payload.get("attributes")
        if isinstance(dimensions, dict):
            return dimensions

        persona = payload.get("persona") or payload.get("data")
        if isinstance(persona, dict):
            nested = persona.get("dimensions") or persona.get("attributes")
            if isinstance(nested, dict):
                return nested

        return {}

    def _extract_attributes(self, catalog: dict[str, Any]) -> list[dict[str, Any]]:
        lookup = self._build_dimension_lookup(catalog)
        attributes: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(attribute: dict[str, Any] | None):
            if not attribute:
                return
            key = attribute.get("key")
            if not key or key in seen:
                return
            seen.add(str(key))
            attributes.append(attribute)

        dimension_categories = catalog.get("dimensionCategories")
        if isinstance(dimension_categories, dict):
            for profile_key in ("devProfile", "profile"):
                profile = dimension_categories.get(profile_key)
                if isinstance(profile, dict):
                    for attribute in self._extract_attributes_from_groups(
                        profile.get("groups"), lookup
                    ):
                        add(attribute)

        for key in ("dimensions", "attributes"):
            value = catalog.get(key)
            if isinstance(value, dict):
                for dimension_key, item in value.items():
                    add(self._normalize_attribute(item, fallback_key=str(dimension_key)))
            elif isinstance(value, list):
                for item in value:
                    add(self._normalize_attribute(item))

        return attributes

    def _build_dimension_lookup(self, catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for key in ("dimensions", "attributes"):
            value = catalog.get(key)
            if isinstance(value, dict):
                for dimension_key, item in value.items():
                    attribute = self._normalize_attribute(item, fallback_key=str(dimension_key))
                    if attribute:
                        lookup[str(attribute["key"])] = attribute
            elif isinstance(value, list):
                for item in value:
                    attribute = self._normalize_attribute(item)
                    if attribute:
                        lookup[str(attribute["key"])] = attribute
        return lookup

    def _extract_attributes_from_groups(
        self,
        groups: Any,
        lookup: dict[str, dict[str, Any]],
        parent_category: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(groups, list):
            return []
        attributes: list[dict[str, Any]] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            category = str(group.get("label") or parent_category or "Other")
            dimensions = group.get("dimensions") or []
            if isinstance(dimensions, list):
                for dimension in dimensions:
                    attributes.append(self._normalize_attribute(dimension, category=category) or {})
            dimension_ids = group.get("dimensionIds") or group.get("dimension_ids") or []
            if isinstance(dimension_ids, list):
                for dimension_id in dimension_ids:
                    key = str(dimension_id)
                    attribute = lookup.get(key) or self._normalize_attribute(key, category=category)
                    if attribute and not attribute.get("category"):
                        attribute = {**attribute, "category": category}
                    attributes.append(attribute or {})
            attributes.extend(
                self._extract_attributes_from_groups(
                    group.get("subgroups"), lookup, parent_category=category
                )
            )
        return [attribute for attribute in attributes if attribute]

    def _normalize_attribute(
        self,
        item: Any,
        fallback_key: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any] | None:
        if isinstance(item, str):
            return {"key": item, "label": item, "category": category or "Other"}
        if not isinstance(item, dict):
            if fallback_key:
                return {"key": fallback_key, "label": fallback_key, "category": category or "Other"}
            return None

        key = item.get("key") or item.get("id") or item.get("name") or fallback_key
        if not key:
            return None
        attribute: dict[str, Any] = {
            "key": str(key),
            "label": item.get("label") or item.get("name") or str(key),
            "category": item.get("category") or category or "Other",
        }
        description = item.get("description")
        if description:
            attribute["description"] = description
        values = item.get("values") or item.get("allowedValues") or item.get("allowed_values")
        if isinstance(values, list):
            attribute["values"] = [str(value) for value in values if value is not None]
        return attribute
