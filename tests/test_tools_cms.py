import httpx
import pytest

from hubspot_agent.client import HubSpotClient
from hubspot_agent.config import PortalConfig
from hubspot_agent.tools.cms import (
    hubspot_get_page,
    hubspot_update_page,
    hubspot_list_files,
    hubspot_upload_file,
    hubspot_publish_social_post,
)


@pytest.mark.asyncio
async def test_hubspot_get_page_by_id(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.get("https://api.hubapi.com/cms/v3/pages/site-pages/1").mock(
        return_value=httpx.Response(200, json={"id": "1", "name": "Home"})
    )
    result = await hubspot_get_page(page_id="1", client=c, portal_id="123")
    assert result["id"] == "1"
    assert result["name"] == "Home"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_get_page_by_path(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.get("https://api.hubapi.com/cms/v3/pages/site-pages?limit=100").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"id": "1", "name": "Home", "url": "/", "slug": "home"},
                    {"id": "2", "name": "About", "url": "/about", "slug": "about"},
                ]
            },
        )
    )
    result = await hubspot_get_page(path="/about", client=c, portal_id="123")
    assert result["id"] == "2"
    assert result["name"] == "About"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_get_page_not_found(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.get("https://api.hubapi.com/cms/v3/pages/site-pages?limit=100").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    result = await hubspot_get_page(path="/missing", client=c, portal_id="123")
    assert "error" in result
    assert "No page found" in result["error"]
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_get_page_requires_id_or_path(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    result = await hubspot_get_page(client=c, portal_id="123")
    assert "error" in result
    assert "Either page_id or path is required" in result["error"]
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_update_page(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.patch("https://api.hubapi.com/cms/v3/pages/site-pages/1").mock(
        return_value=httpx.Response(200, json={"id": "1", "name": "Updated"})
    )
    result = await hubspot_update_page(
        page_id="1", updates={"name": "Updated"}, client=c, portal_id="123"
    )
    assert result["id"] == "1"
    assert result["name"] == "Updated"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_list_files(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.get("https://api.hubapi.com/filemanager/api/v3/files?limit=100").mock(
        return_value=httpx.Response(200, json={"results": [{"id": "f1", "name": "logo.png"}]})
    )
    result = await hubspot_list_files(client=c, portal_id="123")
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "logo.png"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_upload_file(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.post("https://api.hubapi.com/filemanager/api/v3/files").mock(
        return_value=httpx.Response(201, json={"id": "f2", "name": "doc.pdf"})
    )
    result = await hubspot_upload_file(
        file_content=b"pdf content",
        file_name="doc.pdf",
        folder_path="/docs",
        client=c,
        portal_id="123",
    )
    assert result["id"] == "f2"
    assert result["name"] == "doc.pdf"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_publish_social_post(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.post("https://api.hubapi.com/broadcast/v1/broadcasts").mock(
        return_value=httpx.Response(201, json={"id": "b1", "status": "PUBLISHED"})
    )
    result = await hubspot_publish_social_post(
        channel_id="ch1",
        content="Hello world",
        client=c,
        portal_id="123",
    )
    assert result["id"] == "b1"
    assert result["status"] == "PUBLISHED"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_publish_social_post_scheduled(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.post("https://api.hubapi.com/broadcast/v1/broadcasts").mock(
        return_value=httpx.Response(201, json={"id": "b2", "status": "SCHEDULED"})
    )
    result = await hubspot_publish_social_post(
        channel_id="ch1",
        content="Scheduled post",
        client=c,
        portal_id="123",
        scheduled_at="2026-05-08T12:00:00Z",
    )
    assert result["id"] == "b2"
    assert result["status"] == "SCHEDULED"
    await c.close()
