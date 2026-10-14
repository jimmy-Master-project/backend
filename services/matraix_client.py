import threading
from typing import Any
from urllib.parse import quote

import httpx
from django.conf import settings

# One pooled client per process: reuses TCP connections to MatrAIx instead of
# opening a new connection per request. httpx.Client is thread-safe.
_http_client: httpx.Client | None = None
_http_client_lock = threading.Lock()


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        with _http_client_lock:
            if _http_client is None:
                _http_client = httpx.Client()
    return _http_client


class MatrAIxUnavailableError(Exception):
    pass


class MatrAIxClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: float | None = None):
        self.base_url = (base_url or settings.MATRAIX_BASE_URL).rstrip("/")
        self.timeout = httpx.Timeout(
            timeout_seconds or settings.MATRAIX_TIMEOUT_SECONDS,
            connect=settings.MATRAIX_CONNECT_TIMEOUT_SECONDS,
        )

    def check_health(self) -> dict[str, Any]:
        return self._request("GET", "/api/health")

    def get_dimension_catalog(self, pool: str | None = None) -> dict[str, Any]:
        params = {"pool": pool or settings.MATRAIX_DEFAULT_PERSONA_POOL}
        return self._request("GET", "/api/persona-pool/catalog", params=params)

    def sample_personas(
        self,
        sample_size: int,
        pool: str | None = None,
        filters: dict[str, Any] | None = None,
        stratify: Any | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        fields = stratify
        if isinstance(fields, str):
            fields = [fields]
        payload: dict[str, Any] = {
            "sampleSize": sample_size,
            "pool": pool or settings.MATRAIX_DEFAULT_PERSONA_POOL,
            "dimensionFilters": filters or {},
            "fields": fields,
        }
        if seed is not None:
            payload["seed"] = seed
        return self._request("POST", "/api/persona-pool/sample", json=payload)

    def list_personas(self, limit: int, offset: int = 0, pool: str | None = None) -> dict[str, Any]:
        """One page of persona cards in the pool's fixed browse order."""
        params = {
            "pool": pool or settings.MATRAIX_DEFAULT_PERSONA_POOL,
            "limit": limit,
            "offset": offset,
        }
        return self._request("GET", "/api/persona-pool/personas", params=params)

    def get_persona(self, persona_id: str, pool: str | None = None) -> dict[str, Any]:
        params = {"pool": pool or settings.MATRAIX_DEFAULT_PERSONA_POOL}
        return self._request(
            "GET",
            f"/api/persona-pool/personas/{quote(str(persona_id), safe='')}",
            params=params,
        )

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = _get_http_client().request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise MatrAIxUnavailableError("Persona service is unavailable.") from exc
        except ValueError as exc:
            raise MatrAIxUnavailableError("Persona service returned invalid JSON.") from exc

        if not isinstance(data, dict):
            raise MatrAIxUnavailableError("Persona service returned an unexpected response.")
        return data
