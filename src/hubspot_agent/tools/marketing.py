from __future__ import annotations

from typing import Any
from urllib.parse import quote

from hubspot_agent.client import HubSpotClient
from hubspot_agent.errors import HubSpotError, RateLimitError, ScopeError
from hubspot_agent.tools import tool


@tool(name="hubspot_create_email", description="Create a marketing email in HubSpot.")
async def hubspot_create_email(
    name: str,
    subject: str,
    content: dict[str, Any],
    client: HubSpotClient,
    portal_id: str,
    from_email: str | None = None,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    try:
        body: dict[str, Any] = {
            "name": name,
            "subject": subject,
            "content": content,
        }
        if from_email:
            body["from"] = {"email": from_email}
        if campaign_id:
            body["campaignId"] = campaign_id
        resp = await client.post(
            "/marketing/v3/emails",
            portal_id=portal_id,
            body=body,
            expected_scopes=["content"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_create_email"}


@tool(name="hubspot_get_email", description="Retrieve a marketing email by ID.")
async def hubspot_get_email(
    email_id: str,
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    try:
        resp = await client.get(
            f"/marketing/v3/emails/{quote(email_id, safe='')}",
            portal_id=portal_id,
            expected_scopes=["content"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_get_email"}


@tool(name="hubspot_create_campaign", description="Create a marketing campaign in HubSpot.")
async def hubspot_create_campaign(
    name: str,
    client: HubSpotClient,
    portal_id: str,
    notes: str | None = None,
    color: str | None = None,
) -> dict[str, Any]:
    try:
        body: dict[str, Any] = {"name": name}
        if notes:
            body["notes"] = notes
        if color:
            body["color"] = color
        resp = await client.post(
            "/marketing/v3/campaigns",
            portal_id=portal_id,
            body=body,
            expected_scopes=["content"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_create_campaign"}


@tool(name="hubspot_list_campaigns", description="List marketing campaigns in HubSpot.")
async def hubspot_list_campaigns(
    client: HubSpotClient,
    portal_id: str,
    limit: int | None = None,
) -> dict[str, Any]:
    try:
        query = ""
        if limit:
            query = f"?limit={limit}"
        resp = await client.get(
            f"/marketing/v3/campaigns{query}",
            portal_id=portal_id,
            expected_scopes=["content"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_list_campaigns"}


@tool(name="hubspot_create_segment", description="Create a list/segment for marketing.")
async def hubspot_create_segment(
    name: str,
    object_type_id: str,
    processing_type: str,
    client: HubSpotClient,
    portal_id: str,
    filters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        body: dict[str, Any] = {
            "name": name,
            "objectTypeId": object_type_id,
            "processingType": processing_type,
        }
        if filters:
            body["filters"] = filters
        resp = await client.post(
            "/crm/v3/lists",
            portal_id=portal_id,
            body=body,
            expected_scopes=["crm.lists.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_create_segment"}


@tool(name="hubspot_get_segment", description="Retrieve a marketing segment/list by ID.")
async def hubspot_get_segment(
    segment_id: str,
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    try:
        resp = await client.get(
            f"/crm/v3/lists/{quote(segment_id, safe='')}",
            portal_id=portal_id,
            expected_scopes=["crm.lists.read"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_get_segment"}


@tool(name="hubspot_create_ab_test", description="Set up an A/B test for a marketing email.")
async def hubspot_create_ab_test(
    email_id: str,
    test_type: str,
    variants: list[dict[str, Any]],
    client: HubSpotClient,
    portal_id: str,
    winner_criteria: str | None = None,
    test_duration_hours: int | None = None,
) -> dict[str, Any]:
    try:
        body: dict[str, Any] = {
            "testType": test_type,
            "variants": variants,
        }
        if winner_criteria:
            body["winnerCriteria"] = winner_criteria
        if test_duration_hours is not None:
            body["testDurationHours"] = test_duration_hours
        resp = await client.post(
            f"/marketing/v3/emails/{quote(email_id, safe='')}/ab-test",
            portal_id=portal_id,
            body=body,
            expected_scopes=["content"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_create_ab_test"}


@tool(name="hubspot_get_ab_test", description="Retrieve A/B test results for an email.")
async def hubspot_get_ab_test(
    email_id: str,
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    try:
        resp = await client.get(
            f"/marketing/v3/emails/{quote(email_id, safe='')}/ab-test",
            portal_id=portal_id,
            expected_scopes=["content"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_get_ab_test"}


@tool(name="hubspot_create_suppression_list", description="Create a suppression list in HubSpot.")
async def hubspot_create_suppression_list(
    name: str,
    object_type_id: str,
    client: HubSpotClient,
    portal_id: str,
    record_ids: list[str] | None = None,
) -> dict[str, Any]:
    try:
        body: dict[str, Any] = {
            "name": name,
            "objectTypeId": object_type_id,
            "processingType": "SUPPRESSION",
        }
        resp = await client.post(
            "/crm/v3/lists",
            portal_id=portal_id,
            body=body,
            expected_scopes=["crm.lists.write"],
        )
        list_id = resp.body.get("id")
        if list_id and record_ids:
            await client.post(
                f"/crm/v3/lists/{quote(list_id, safe='')}/memberships/add",
                portal_id=portal_id,
                body={"recordIds": record_ids},
                expected_scopes=["crm.lists.write"],
            )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_create_suppression_list"}


@tool(name="hubspot_list_suppression_lists", description="List suppression lists in HubSpot.")
async def hubspot_list_suppression_lists(
    client: HubSpotClient = None,
    portal_id: str = "",
) -> dict[str, Any]:
    try:
        resp = await client.get(
            "/crm/v3/lists",
            portal_id=portal_id,
            expected_scopes=["crm.lists.read"],
        )
        results = resp.body.get("results", [])
        suppression = [r for r in results if r.get("processingType") == "SUPPRESSION"]
        return {"results": suppression}
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_list_suppression_lists"}


@tool(name="hubspot_send_email", description="Send or trigger a marketing email.")
async def hubspot_send_email(
    email_id: str,
    client: HubSpotClient,
    portal_id: str,
    send_to_list_id: str | None = None,
) -> dict[str, Any]:
    try:
        body: dict[str, Any] = {}
        if send_to_list_id:
            body["sendToListId"] = send_to_list_id
        resp = await client.post(
            f"/marketing/v3/emails/{quote(email_id, safe='')}/send",
            portal_id=portal_id,
            body=body,
            expected_scopes=["content"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_send_email"}


@tool(name="hubspot_get_email_performance", description="Retrieve performance metrics for a marketing email.")
async def hubspot_get_email_performance(
    email_id: str,
    client: HubSpotClient,
    portal_id: str,
) -> dict[str, Any]:
    try:
        resp = await client.get(
            f"/analytics/v2/reports/emails/{quote(email_id, safe='')}/performance",
            portal_id=portal_id,
            expected_scopes=["content"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_get_email_performance"}
