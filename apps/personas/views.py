# pyright: reportMissingImports=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false
from typing import Any, cast

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from services.matraix_client import MatrAIxUnavailableError

from .protected_attributes import serialize_protected_attributes
from .serializers import PersonaListQuerySerializer, PersonaSampleRequestSerializer
from .services import PersonaService, PersonaServiceError


class PersonaAttributesView(APIView):
    def get(self, request):
        pool = request.query_params.get("pool") or settings.MATRAIX_DEFAULT_PERSONA_POOL
        try:
            catalog = PersonaService().get_attributes(pool)
            # The raw MatrAIx catalog is not forwarded: `attributes` is already
            # the normalized list extracted from it.
            return Response(
                {
                    "pool": catalog["pool"],
                    "attributes": catalog["attributes"],
                    # The only attributes a fairness label may use (LangFair-supported).
                    "protectedAttributes": serialize_protected_attributes(),
                }
            )
        except MatrAIxUnavailableError:
            return Response(
                {
                    "error": "matraix_unavailable",
                    "message": "Persona service is unavailable.",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )


def _unavailable():
    return Response(
        {"error": "matraix_unavailable", "message": "Persona service is unavailable."},
        status=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


class PersonaListView(APIView):
    """GET persona/personas?offset=&limit= : browse personas page by page."""

    def get(self, request):
        query = PersonaListQuerySerializer(data=request.query_params)
        query.is_valid(raise_exception=True)
        data = cast(dict[str, Any], query.validated_data)
        try:
            result = PersonaService().list_personas(
                limit=data["limit"],
                offset=data["offset"],
                pool=data.get("pool") or settings.MATRAIX_DEFAULT_PERSONA_POOL,
                persona_ids=data.get("ids") or None,
            )
        except MatrAIxUnavailableError:
            return _unavailable()
        return Response(
            {
                "pool": result["pool"],
                "offset": result["offset"],
                "limit": result["limit"],
                "total": result["total"],
                "personas": [
                    {
                        "personaId": persona["persona_id"],
                        "name": persona["name"],
                        "source": persona["source"],
                        # Keys are MatrAIx dimension names (data), passed through as-is.
                        "summary": persona["summary"],
                        "searchText": persona["search_text"],
                    }
                    for persona in result["personas"]
                ],
            }
        )


class PersonaDetailView(APIView):
    """GET persona/personas/<id> : every attribute of one persona."""

    def get(self, request, persona_id):
        pool = request.query_params.get("pool") or settings.MATRAIX_DEFAULT_PERSONA_POOL
        try:
            persona = PersonaService().get_projected_persona(persona_id, [], pool)
        except MatrAIxUnavailableError:
            return _unavailable()
        return Response(_serialize_persona(persona))


class PersonaSampleView(APIView):
    def post(self, request):
        serializer = PersonaSampleRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = cast(dict[str, Any], serializer.validated_data)

        try:
            result = PersonaService().sample_projected_personas(
                sample_size=data["sample_size"],
                selected_attributes=data["selected_attributes"],
                pool=data.get("pool") or settings.MATRAIX_DEFAULT_PERSONA_POOL,
                filters=data.get("filters") or {},
                stratify=data.get("stratify"),
                seed=data.get("seed"),
                allow_partial=True,
            )
        except MatrAIxUnavailableError:
            return Response(
                {
                    "error": "matraix_unavailable",
                    "message": "Persona service is unavailable.",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except PersonaServiceError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_502_BAD_GATEWAY)

        return Response(
            {
                "count": len(result["personas"]),
                "pool": result["pool"],
                "selectedAttributes": result["selected_attributes"],
                "personas": [_serialize_persona(persona) for persona in result["personas"]],
                "partial": result.get("partial", False),
            }
        )


def _serialize_persona(persona):
    return {
        "personaId": persona["persona_id"],
        # Keys are MatrAIx dimension names (data) and are passed through as-is.
        "attributes": persona["attributes"],
        "missingAttributes": persona["missing_attributes"],
    }
