from __future__ import annotations

from typing import Any
from urllib.parse import quote

from hubspot_agent.client import HubSpotClient
from hubspot_agent.errors import HubSpotError, RateLimitError, ScopeError
from hubspot_agent.tools import tool


@tool(name="hubspot_get_workflow", description="Retrieve a HubSpot workflow by ID.")
async def hubspot_get_workflow(
    workflow_id: str,
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    try:
        resp = await client.get(
            f"/automation/v4/workflows/{quote(workflow_id, safe='')}",
            portal_id=portal_id,
            expected_scopes=["automation.workflows.read"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_get_workflow"}


@tool(name="hubspot_list_workflows", description="List all HubSpot workflows.")
async def hubspot_list_workflows(
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    try:
        resp = await client.get(
            "/automation/v4/workflows",
            portal_id=portal_id,
            expected_scopes=["automation.workflows.read"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_list_workflows"}


@tool(name="hubspot_create_workflow", description="Create a new HubSpot workflow.")
async def hubspot_create_workflow(
    name: str,
    workflow_type: str,
    actions: list[dict[str, Any]],
    enrollment: dict[str, Any],
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    try:
        resp = await client.post(
            "/automation/v4/workflows",
            portal_id=portal_id,
            body={"name": name, "type": workflow_type, "actions": actions, "enrollment": enrollment},
            expected_scopes=["automation.workflows.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_create_workflow"}


@tool(name="hubspot_update_workflow", description="Update an existing HubSpot workflow.")
async def hubspot_update_workflow(
    workflow_id: str,
    updates: dict[str, Any],
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    try:
        resp = await client.patch(
            f"/automation/v4/workflows/{quote(workflow_id, safe='')}",
            portal_id=portal_id,
            body=updates,
            expected_scopes=["automation.workflows.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_update_workflow"}


@tool(name="hubspot_enroll_workflow", description="Enroll records into a HubSpot workflow.")
async def hubspot_enroll_workflow(
    workflow_id: str,
    object_ids: list[str],
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    try:
        resp = await client.post(
            f"/automation/v4/workflows/{quote(workflow_id, safe='')}/enrollments",
            portal_id=portal_id,
            body={"objectIds": object_ids},
            expected_scopes=["automation.workflows.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_enroll_workflow"}


@tool(name="hubspot_toggle_workflow", description="Toggle a HubSpot workflow on or off.")
async def hubspot_toggle_workflow(
    workflow_id: str,
    enabled: bool,
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    try:
        resp = await client.post(
            f"/automation/v4/workflows/{quote(workflow_id, safe='')}/toggle",
            portal_id=portal_id,
            body={"enabled": enabled},
            expected_scopes=["automation.workflows.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_toggle_workflow"}
