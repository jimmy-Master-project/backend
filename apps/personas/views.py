# pyright: reportMissingImports=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false
from typing import Any, cast

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from services.matraix_client import MatrAIxUnavailableError

from .serializers import PersonaSampleRequestSerializer
from .services import PersonaService, PersonaServiceError


class PersonaAttributesView(APIView):
    def get(self, request):
        pool = request.query_params.get("pool") or settings.MATRAIX_DEFAULT_PERSONA_POOL
        try:
            catalog = PersonaService().get_attributes(pool)
            # The raw MatrAIx catalog is not forwarded: `attributes` is already
            # the normalized list extracted from it.
            return Response({"pool": catalog["pool"], "attributes": catalog["attributes"]})
        except MatrAIxUnavailableError:
            return Response(
                {
                    "error": "matraix_unavailable",
                    "message": "Persona service is unavailable.",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )


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
