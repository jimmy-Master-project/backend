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
            return Response(PersonaService().get_attributes(pool))
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
                "selected_attributes": result["selected_attributes"],
                "personas": result["personas"],
                "partial": result.get("partial", False),
            }
        )
