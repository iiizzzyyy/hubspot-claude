from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from hubspot_agent.client import HubSpotClient
from hubspot_agent.errors import HubSpotError, RateLimitError, ScopeError
from hubspot_agent.tools import tool


@tool(name="hubspot_get_page", description="Retrieve a HubSpot CMS page by ID or path.")
async def hubspot_get_page(
    page_id: str | None = None,
    path: str | None = None,
    page_type: str = "site-page",
    client: HubSpotClient = None,
    portal_id: str = "",
) -> dict[str, Any]:
    if not page_id and not path:
        return {"error": "Either page_id or path is required", "tool": "hubspot_get_page"}

    api_segment = "blog-posts" if page_type == "blog-post" else "site-pages"
    try:
        if page_id:
            resp = await client.get(
                f"/cms/v3/pages/{api_segment}/{quote(page_id, safe='')}",
                portal_id=portal_id,
                expected_scopes=["content.pages.read"],
            )
            return resp.body

        # Path-based lookup with pagination
        after = None
        while True:
            query = f"?limit=100"
            if after:
                query += f"&after={after}"
            resp = await client.get(
                f"/cms/v3/pages/{api_segment}{query}",
                portal_id=portal_id,
                expected_scopes=["content.pages.read"],
            )
            results = resp.body.get("results", [])
            for page in results:
                if page.get("url") == path or page.get("slug") == path:
                    return page
            paging = resp.body.get("paging", {})
            next_after = paging.get("next", {}).get("after")
            if not next_after:
                break
            after = next_after
        return {"error": f"No page found for path: {path}", "tool": "hubspot_get_page"}
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_get_page"}


@tool(name="hubspot_update_page", description="Update a HubSpot CMS page content.")
async def hubspot_update_page(
    page_id: str,
    updates: dict[str, Any],
    page_type: str = "site-page",
    client: HubSpotClient = None,
    portal_id: str = "",
) -> dict[str, Any]:
    api_segment = "blog-posts" if page_type == "blog-post" else "site-pages"
    try:
        resp = await client.patch(
            f"/cms/v3/pages/{api_segment}/{quote(page_id, safe='')}",
            portal_id=portal_id,
            body=updates,
            expected_scopes=["content.pages.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_update_page"}


@tool(name="hubspot_list_files", description="List files in the HubSpot file manager.")
async def hubspot_list_files(
    folder_path: str | None = None,
    limit: int = 100,
    client: HubSpotClient = None,
    portal_id: str = "",
) -> dict[str, Any]:
    try:
        params = f"?limit={limit}"
        if folder_path:
            params += f"&folderPath={quote(folder_path, safe='')}"
        resp = await client.get(
            f"/filemanager/api/v3/files{params}",
            portal_id=portal_id,
            expected_scopes=["files.files.read"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_list_files"}


@tool(
    name="hubspot_upload_file",
    description="Upload a file to the HubSpot file manager.",
)
async def hubspot_upload_file(
    file_content: bytes,
    file_name: str,
    folder_path: str = "/",
    client: HubSpotClient = None,
    portal_id: str = "",
) -> dict[str, Any]:
    try:
        options = json.dumps({"folderPath": folder_path, "access": "PUBLIC_INDEXABLE"})
        files = {"file": (file_name, file_content)}
        data = {"options": options}
        resp = await client.post_files(
            "/filemanager/api/v3/files",
            portal_id=portal_id,
            data=data,
            files=files,
            expected_scopes=["files.files.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_upload_file"}


@tool(
    name="hubspot_publish_social_post",
    description="Publish a social media post via HubSpot.",
)
async def hubspot_publish_social_post(
    channel_id: str,
    content: str,
    client: HubSpotClient = None,
    portal_id: str = "",
    scheduled_at: str | None = None,
) -> dict[str, Any]:
    try:
        body: dict[str, Any] = {
            "channelId": channel_id,
            "content": {"body": content},
        }
        if scheduled_at:
            body["scheduledAt"] = scheduled_at
        resp = await client.post(
            "/broadcast/v1/broadcasts",
            portal_id=portal_id,
            body=body,
            expected_scopes=["social.publish.write"],
        )
        return resp.body
    except (HubSpotError, RateLimitError, ScopeError) as exc:
        return {"error": str(exc), "tool": "hubspot_publish_social_post"}
