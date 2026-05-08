import json

import httpx
import pytest

from hubspot_agent.client import HubSpotClient
from hubspot_agent.config import PortalConfig
from hubspot_agent.errors import ErrorCategory, HubSpotError, RateLimitError
from hubspot_agent.testing import ChaosConfig, ChaosHubSpotClient
from hubspot_agent.tools.objects import hubspot_get_object, hubspot_batch_upsert_objects


@pytest.mark.asyncio
async def test_chaos_deterministic_with_seed(respx_mock):
    """Same seed produces identical fault sequences."""
    cfg = ChaosConfig(rate_limit_rate=1.0, chaos_seed=42)
    client1 = ChaosHubSpotClient(PortalConfig(portal_id="123", token="t"), cfg)
    client2 = ChaosHubSpotClient(PortalConfig(portal_id="123", token="t"), cfg)

    faults1 = []
    faults2 = []
    for _ in range(20):
        try:
            await client1.get("/test", portal_id="123")
        except RateLimitError:
            faults1.append("rate_limit")
        except Exception as exc:
            faults1.append(type(exc).__name__)

        try:
            await client2.get("/test", portal_id="123")
        except RateLimitError:
            faults2.append("rate_limit")
        except Exception as exc:
            faults2.append(type(exc).__name__)

    assert faults1 == faults2
    await client1.close()
    await client2.close()


@pytest.mark.asyncio
async def test_chaos_rate_limit_fault():
    """Rate limit fault has retry_after between 1-30 seconds."""
    cfg = ChaosConfig(rate_limit_rate=1.0, network_error_rate=0.0, truncation_rate=0.0)
    client = ChaosHubSpotClient(PortalConfig(portal_id="123", token="t"), cfg)

    with pytest.raises(RateLimitError) as exc_info:
        await client.get("/test", portal_id="123")

    assert exc_info.value.retry_after is not None
    assert 1 <= exc_info.value.retry_after <= 30
    await client.close()


@pytest.mark.asyncio
async def test_chaos_network_fault():
    """Network fault raises HubSpotError with SERVER category."""
    cfg = ChaosConfig(rate_limit_rate=0.0, network_error_rate=1.0, truncation_rate=0.0)
    client = ChaosHubSpotClient(PortalConfig(portal_id="123", token="t"), cfg)

    with pytest.raises(HubSpotError) as exc_info:
        await client.get("/test", portal_id="123")

    assert exc_info.value.category == ErrorCategory.SERVER
    assert exc_info.value.status_code == 503
    await client.close()


@pytest.mark.asyncio
async def test_chaos_truncation_fault(respx_mock):
    """Truncation fault raises JSONDecodeError to simulate partial read."""
    cfg = ChaosConfig(rate_limit_rate=0.0, network_error_rate=0.0, truncation_rate=1.0)
    client = ChaosHubSpotClient(PortalConfig(portal_id="123", token="t"), cfg)

    respx_mock.get("https://api.hubapi.com/test").mock(
        return_value=httpx.Response(200, json={"id": "1", "properties": {"email": "a@b.com"}})
    )

    with pytest.raises(json.JSONDecodeError):
        await client.get("/test", portal_id="123")
    await client.close()


@pytest.mark.asyncio
async def test_chaos_zero_rates_no_faults(respx_mock):
    """With all rates at 0, chaos client behaves like normal client."""
    cfg = ChaosConfig(rate_limit_rate=0.0, network_error_rate=0.0, truncation_rate=0.0)
    client = ChaosHubSpotClient(PortalConfig(portal_id="123", token="t"), cfg)

    respx_mock.get("https://api.hubapi.com/crm/v3/objects/contacts/1").mock(
        return_value=httpx.Response(200, json={"id": "1"})
    )
    resp = await client.get("/crm/v3/objects/contacts/1", portal_id="123")
    assert resp.body["id"] == "1"
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 42, 99])
async def test_chaos_existing_client_pattern_with_seed(respx_mock, seed):
    """Parameterized: existing GET test pattern with chaos client at zero rates."""
    cfg = ChaosConfig(
        rate_limit_rate=0.0,
        network_error_rate=0.0,
        truncation_rate=0.0,
        chaos_seed=seed,
    )
    client = ChaosHubSpotClient(PortalConfig(portal_id="123", token="t"), cfg)

    respx_mock.get("https://api.hubapi.com/crm/v3/objects/contacts/1").mock(
        return_value=httpx.Response(200, json={"id": "1", "properties": {"email": "a@b.com"}})
    )
    resp = await client.get("/crm/v3/objects/contacts/1", portal_id="123")
    assert resp.body["id"] == "1"
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 42])
async def test_chaos_tool_function_with_seed(respx_mock, seed):
    """Parameterized: existing tool test with chaos client at zero rates."""
    cfg = ChaosConfig(
        rate_limit_rate=0.0,
        network_error_rate=0.0,
        truncation_rate=0.0,
        chaos_seed=seed,
    )
    client = ChaosHubSpotClient(PortalConfig(portal_id="123", token="t"), cfg)

    respx_mock.get("https://api.hubapi.com/crm/v3/objects/contacts/1").mock(
        return_value=httpx.Response(200, json={"id": "1", "properties": {"email": "a@b.com"}})
    )
    result = await hubspot_get_object(
        object_id="1", object_type="contacts", client=client, portal_id="123"
    )
    assert result["id"] == "1"
    await client.close()


@pytest.mark.asyncio
async def test_chaos_401_retry_still_works(respx_mock, monkeypatch, tmp_path):
    """Verify 401 refresh-and-retry behavior is preserved under chaos client."""
    import time

    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    cfg = ChaosConfig(rate_limit_rate=0.0, network_error_rate=0.0, truncation_rate=0.0)
    client = ChaosHubSpotClient(
        PortalConfig(
            portal_id="123",
            token="expired-token",
            auth_type="oauth",
            refresh_token="refresh-123",
            expires_at=time.time() + 10000,
        ),
        cfg,
    )

    call_count = {"n": 0}

    def handler(request):
        auth = request.headers.get("Authorization", "")
        call_count["n"] += 1
        if call_count["n"] == 1:
            assert "expired-token" in auth
            return httpx.Response(401, json={"message": "Token expired"})
        assert "valid-token" in auth
        return httpx.Response(200, json={"id": "1"})

    respx_mock.get("https://api.hubapi.com/crm/v3/objects/contacts/1").mock(side_effect=handler)
    respx_mock.post("https://api.hubapi.com/oauth/v1/token").mock(
        return_value=httpx.Response(
            200,
            json={
                "access_token": "valid-token",
                "refresh_token": "refresh-123",
                "expires_in": 21600,
            },
        )
    )

    resp = await client.get("/crm/v3/objects/contacts/1", portal_id="123")
    assert resp.body["id"] == "1"
    assert call_count["n"] == 2
    await client.close()


@pytest.mark.asyncio
async def test_chaos_batch_graceful_degradation_rate_limit(respx_mock):
    """Batch operations continue when chaos injects rate limits per chunk."""
    cfg = ChaosConfig(rate_limit_rate=1.0, network_error_rate=0.0, truncation_rate=0.0)
    client = ChaosHubSpotClient(PortalConfig(portal_id="123", token="t"), cfg)

    respx_mock.post("https://api.hubapi.com/crm/v3/objects/contacts/batch/create").mock(
        return_value=httpx.Response(200, json={"results": [{"id": "3"}], "errors": []})
    )

    result = await hubspot_batch_upsert_objects(
        object_type="contacts",
        records=[{"email": "batch@example.com"}],
        client=client,
        portal_id="123",
    )
    assert result["failed"] == 1
    assert result["succeeded"] == 0
    assert len(result["errors"]) == 1
    assert "Rate limit exceeded (chaos injected)" in result["errors"][0]["message"]
    await client.close()


@pytest.mark.asyncio
async def test_chaos_batch_graceful_degradation_network(respx_mock):
    """Batch operations continue when chaos injects network errors per chunk."""
    cfg = ChaosConfig(rate_limit_rate=0.0, network_error_rate=1.0, truncation_rate=0.0)
    client = ChaosHubSpotClient(PortalConfig(portal_id="123", token="t"), cfg)

    respx_mock.post("https://api.hubapi.com/crm/v3/objects/contacts/batch/create").mock(
        return_value=httpx.Response(200, json={"results": [{"id": "3"}], "errors": []})
    )

    result = await hubspot_batch_upsert_objects(
        object_type="contacts",
        records=[{"email": "batch@example.com"}],
        client=client,
        portal_id="123",
    )
    assert result["failed"] == 1
    assert result["succeeded"] == 0
    assert len(result["errors"]) == 1
    assert any(
        msg in result["errors"][0]["message"] for msg in ChaosHubSpotClient._NETWORK_MESSAGES
    )
    await client.close()
