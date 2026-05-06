from __future__ import annotations

from typing import Any
from urllib.parse import quote

from hubspot_agent.client import HubSpotClient
from hubspot_agent.errors import HubSpotError, RateLimitError, ScopeError
from hubspot_agent.tools import tool

_VALID_OBJECT_TYPES = frozenset({"contacts", "companies", "deals", "tickets"})


def _validate_object_type(object_type: str) -> None:
    if object_type not in _VALID_OBJECT_TYPES:
        raise ValueError(
            f"Invalid object_type '{object_type}'. "
            f"Must be one of: {', '.join(sorted(_VALID_OBJECT_TYPES))}"
        )


@tool(name="hubspot_get_object", description="Retrieve a HubSpot object by ID.")
async def hubspot_get_object(
    object_id: str,
    object_type: str,
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    _validate_object_type(object_type)
    try:
        resp = await client.get(
            f"/crm/v3/objects/{object_type}/{quote(object_id, safe='')}",
            portal_id=portal_id,
            expected_scopes=[f"crm.objects.{object_type}.read"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_get_object"}


@tool(name="hubspot_search_objects", description="Search HubSpot objects using filter groups.")
async def hubspot_search_objects(
    object_type: str,
    query: dict[str, Any],
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    _validate_object_type(object_type)
    try:
        resp = await client.post(
            f"/crm/v3/objects/{object_type}/search",
            portal_id=portal_id,
            body=query,
            expected_scopes=[f"crm.objects.{object_type}.read"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_search_objects"}


@tool(name="hubspot_create_object", description="Create a new HubSpot object record.")
async def hubspot_create_object(
    object_type: str,
    properties: dict[str, Any],
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    _validate_object_type(object_type)
    try:
        resp = await client.post(
            f"/crm/v3/objects/{object_type}",
            portal_id=portal_id,
            body={"properties": properties},
            expected_scopes=[f"crm.objects.{object_type}.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_create_object"}


@tool(name="hubspot_update_object", description="Update an existing HubSpot object record.")
async def hubspot_update_object(
    object_id: str,
    object_type: str,
    properties: dict[str, Any],
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    _validate_object_type(object_type)
    try:
        resp = await client.patch(
            f"/crm/v3/objects/{object_type}/{quote(object_id, safe='')}",
            portal_id=portal_id,
            body={"properties": properties},
            expected_scopes=[f"crm.objects.{object_type}.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_update_object"}


@tool(name="hubspot_delete_object", description="Permanently delete a HubSpot object record.")
async def hubspot_delete_object(
    object_id: str,
    object_type: str,
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    _validate_object_type(object_type)
    try:
        resp = await client.delete(
            f"/crm/v3/objects/{object_type}/{quote(object_id, safe='')}",
            portal_id=portal_id,
            expected_scopes=[f"crm.objects.{object_type}.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_delete_object"}


_BATCH_SIZE = 100


def _partition_records(
    records: list[dict[str, Any]], unique_key: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seen: dict[str, dict[str, Any]] = {}
    creates: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    for record in records:
        key = str(record.get(unique_key, "")).lower().strip()
        if key and key in seen:
            continue
        if key:
            seen[key] = record
        obj_id = record.get("id") or record.get("hs_object_id")
        if obj_id:
            updates.append({"id": str(obj_id), "properties": record})
        else:
            creates.append({"properties": record})
    return seen, creates, updates


def _chunk(inputs: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [inputs[i : i + size] for i in range(0, len(inputs), size)]


@tool(name="hubspot_batch_upsert_objects", description="Batch create or update HubSpot objects with input-side deduplication.")
async def hubspot_batch_upsert_objects(
    object_type: str,
    records: list[dict[str, Any]],
    client: HubSpotClient,
    portal_id: str,
    unique_key: str = "email",
) -> dict[str, Any]:
    _validate_object_type(object_type)
    _, creates, updates = _partition_records(records, unique_key)

    created_count = 0
    updated_count = 0
    errors: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for chunk in _chunk(creates, _BATCH_SIZE):
        try:
            resp = await client.post(
                f"/crm/v3/objects/{object_type}/batch/create",
                portal_id=portal_id,
                body={"inputs": chunk},
                expected_scopes=[f"crm.objects.{object_type}.write"],
            )
            body = resp.body
            created_count += len(body.get("results", []))
            results.extend(body.get("results", []))
            errors.extend(body.get("errors", []))
        except (HubSpotError, RateLimitError, ScopeError) as exc:
            errors.append({"message": str(exc), "category": "BATCH_CREATE"})

    for chunk in _chunk(updates, _BATCH_SIZE):
        try:
            resp = await client.post(
                f"/crm/v3/objects/{object_type}/batch/update",
                portal_id=portal_id,
                body={"inputs": chunk},
                expected_scopes=[f"crm.objects.{object_type}.write"],
            )
            body = resp.body
            updated_count += len(body.get("results", []))
            results.extend(body.get("results", []))
            errors.extend(body.get("errors", []))
        except (HubSpotError, RateLimitError, ScopeError) as exc:
            errors.append({"message": str(exc), "category": "BATCH_UPDATE"})

    return {
        "succeeded": created_count + updated_count,
        "failed": len(errors),
        "total": len(records),
        "results": results,
        "errors": errors,
    }
