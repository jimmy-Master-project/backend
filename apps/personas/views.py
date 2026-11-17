# pyright: reportMissingImports=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false
import random
from typing import Any, cast

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from services.matraix_client import MatrAIxUnavailableError

from .protected_attributes import serialize_protected_attributes
from .models import PersonaSet
from .serializers import PersonaListQuerySerializer, PersonaSampleRequestSerializer, PersonaSetCreateSerializer
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


def _serialize_persona_set(persona_set: PersonaSet, include_personas: bool = False):
    data = {
        "id": persona_set.id,
        "name": persona_set.name,
        "pool": persona_set.pool,
        # Keys are MatrAIx dimension names (data) and are passed through as-is.
        "filters": persona_set.filters,
        "requestedSize": persona_set.requested_size,
        "size": len(persona_set.persona_ids),
        "seed": persona_set.seed,
        "personaIds": persona_set.persona_ids,
        "createdAt": persona_set.created_at.isoformat() if persona_set.created_at else None,
    }
    if include_personas:
        data["personas"] = [
            {"personaId": item.get("persona_id"), "attributes": item.get("attributes") or {}}
            for item in persona_set.personas
        ]
    return data


class PersonaSetListCreateView(APIView):
    """GET persona/sets/ : saved persona lists. POST : sample by conditions and save."""

    def get(self, request):
        return Response([_serialize_persona_set(item) for item in PersonaSet.objects.all()])

    def post(self, request):
        serializer = PersonaSetCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = cast(dict[str, Any], serializer.validated_data)
        seed = data.get("seed")
        if seed is None:
            seed = random.SystemRandom().randint(0, 2_147_483_647)
        try:
            result = PersonaService().sample_persona_set(
                sample_size=data["sample_size"],
                filters=data.get("filters") or {},
                seed=seed,
                pool=data.get("pool") or settings.MATRAIX_DEFAULT_PERSONA_POOL,
            )
        except MatrAIxUnavailableError:
            return _unavailable()
        except PersonaServiceError:
            return Response(
                {
                    "error": "not_enough_personas",
                    "message": f"Fewer than {data['sample_size']} personas match these conditions. "
                    "Loosen the conditions or lower the number of personas.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        personas = result["personas"]
        persona_set = PersonaSet.objects.create(
            name=data["name"],
            pool=result["pool"],
            filters=data.get("filters") or {},
            requested_size=data["sample_size"],
            seed=seed,
            persona_ids=[item["persona_id"] for item in personas],
            personas=[{"persona_id": item["persona_id"], "attributes": item["attributes"]} for item in personas],
        )
        return Response(_serialize_persona_set(persona_set, include_personas=True), status=status.HTTP_201_CREATED)


class PersonaSetDetailView(APIView):
    """GET persona/sets/<id>/ (with every persona's attributes), DELETE."""

    def get(self, request, set_id):
        persona_set = PersonaSet.objects.filter(pk=set_id).first()
        if persona_set is None:
            return Response({"error": "not_found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(_serialize_persona_set(persona_set, include_personas=True))

    def delete(self, request, set_id):
        deleted, _ = PersonaSet.objects.filter(pk=set_id).delete()
        if not deleted:
            return Response({"error": "not_found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(status=status.HTTP_204_NO_CONTENT)
