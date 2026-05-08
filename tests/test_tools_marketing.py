import httpx
import pytest

from hubspot_agent.client import HubSpotClient
from hubspot_agent.config import PortalConfig
from hubspot_agent.tools.marketing import (
    hubspot_create_ab_test,
    hubspot_create_campaign,
    hubspot_create_email,
    hubspot_create_segment,
    hubspot_create_suppression_list,
    hubspot_get_ab_test,
    hubspot_get_email,
    hubspot_get_email_performance,
    hubspot_get_segment,
    hubspot_list_campaigns,
    hubspot_list_suppression_lists,
    hubspot_send_email,
)


@pytest.mark.asyncio
async def test_hubspot_create_email(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.post("https://api.hubapi.com/marketing/v3/emails").mock(
        return_value=httpx.Response(201, json={"id": "e1", "name": "Welcome"})
    )
    result = await hubspot_create_email(
        name="Welcome",
        subject="Welcome to our service",
        content={"html": "<p>Hello</p>"},
        client=c,
        portal_id="123",
    )
    assert result["id"] == "e1"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_get_email(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.get("https://api.hubapi.com/marketing/v3/emails/e1").mock(
        return_value=httpx.Response(200, json={"id": "e1", "name": "Welcome"})
    )
    result = await hubspot_get_email(email_id="e1", client=c, portal_id="123")
    assert result["id"] == "e1"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_create_campaign(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.post("https://api.hubapi.com/marketing/v3/campaigns").mock(
        return_value=httpx.Response(201, json={"id": "c1", "name": "Q1 Campaign"})
    )
    result = await hubspot_create_campaign(
        name="Q1 Campaign", client=c, portal_id="123", notes="First quarter"
    )
    assert result["id"] == "c1"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_list_campaigns(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.get("https://api.hubapi.com/marketing/v3/campaigns").mock(
        return_value=httpx.Response(200, json={"results": [{"id": "c1"}]})
    )
    result = await hubspot_list_campaigns(client=c, portal_id="123")
    assert len(result["results"]) == 1
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_create_segment(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.post("https://api.hubapi.com/crm/v3/lists").mock(
        return_value=httpx.Response(201, json={"id": "s1", "name": "New Leads"})
    )
    result = await hubspot_create_segment(
        name="New Leads",
        object_type_id="0-1",
        processing_type="DYNAMIC",
        client=c,
        portal_id="123",
    )
    assert result["id"] == "s1"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_get_segment(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.get("https://api.hubapi.com/crm/v3/lists/s1").mock(
        return_value=httpx.Response(200, json={"id": "s1", "name": "New Leads"})
    )
    result = await hubspot_get_segment(segment_id="s1", client=c, portal_id="123")
    assert result["id"] == "s1"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_create_ab_test(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.post("https://api.hubapi.com/marketing/v3/emails/e1/ab-test").mock(
        return_value=httpx.Response(201, json={"testId": "ab1", "testType": "SUBJECT"})
    )
    result = await hubspot_create_ab_test(
        email_id="e1",
        test_type="SUBJECT",
        variants=[{"name": "A"}, {"name": "B"}],
        client=c,
        portal_id="123",
    )
    assert result["testId"] == "ab1"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_get_ab_test(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.get("https://api.hubapi.com/marketing/v3/emails/e1/ab-test").mock(
        return_value=httpx.Response(200, json={"testId": "ab1", "winner": "A"})
    )
    result = await hubspot_get_ab_test(email_id="e1", client=c, portal_id="123")
    assert result["testId"] == "ab1"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_create_suppression_list(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.post("https://api.hubapi.com/crm/v3/lists").mock(
        return_value=httpx.Response(201, json={"id": "sup1", "name": "Do Not Email"})
    )
    respx_mock.post("https://api.hubapi.com/crm/v3/lists/sup1/memberships/add").mock(
        return_value=httpx.Response(200)
    )
    result = await hubspot_create_suppression_list(
        name="Do Not Email",
        object_type_id="0-1",
        client=c,
        portal_id="123",
        record_ids=["101"],
    )
    assert result["id"] == "sup1"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_list_suppression_lists(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.get("https://api.hubapi.com/crm/v3/lists").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"id": "sup1", "processingType": "SUPPRESSION"},
                    {"id": "l1", "processingType": "STATIC"},
                ]
            },
        )
    )
    result = await hubspot_list_suppression_lists(client=c, portal_id="123")
    assert len(result["results"]) == 1
    assert result["results"][0]["id"] == "sup1"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_send_email(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.post("https://api.hubapi.com/marketing/v3/emails/e1/send").mock(
        return_value=httpx.Response(200, json={"status": "SCHEDULED"})
    )
    result = await hubspot_send_email(email_id="e1", client=c, portal_id="123")
    assert result["status"] == "SCHEDULED"
    await c.close()


@pytest.mark.asyncio
async def test_hubspot_get_email_performance(respx_mock):
    c = HubSpotClient(PortalConfig(portal_id="123", token="t"))
    respx_mock.get("https://api.hubapi.com/analytics/v2/reports/emails/e1/performance").mock(
        return_value=httpx.Response(200, json={"sent": 1000, "openRate": 0.45})
    )
    result = await hubspot_get_email_performance(email_id="e1", client=c, portal_id="123")
    assert result["sent"] == 1000
    await c.close()
