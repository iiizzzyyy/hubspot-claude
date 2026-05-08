from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import sys
from typing import Any
from unittest.mock import patch

import pytest
from aiohttp import ClientSession

from hubspot_agent.client import APIResponse
from hubspot_agent.config import PortalConfig
from hubspot_agent.webhooks import (
    WebhookEventProcessor,
    WebhookServer,
    WebhookSubscriptionManager,
)


@pytest.fixture
def client_secret() -> str:
    return "test-client-secret"


@pytest.fixture
def portal_id() -> str:
    return "12345"


@pytest.fixture
def mock_portal() -> PortalConfig:
    return PortalConfig(portal_id="12345", token="test-token", tier="Professional")


@pytest.fixture
def webhook_payload() -> dict[str, Any]:
    return {
        "eventId": 1,
        "subscriptionType": "contact.propertyChange",
        "objectId": 101,
        "propertyName": "email",
        "propertyValue": "test@example.com",
    }


def _sign(body: bytes, secret: str) -> str:
    return base64.b64encode(
        hmac.new(secret.encode(), body, hashlib.sha256).digest()
    ).decode()


# ---------------------------------------------------------------------------
# WebhookServer tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_server_accepts_valid_signature(
    client_secret: str, portal_id: str, webhook_payload: dict[str, Any]
) -> None:
    server = WebhookServer(client_secret=client_secret, portal_id=portal_id, port=18080)
    await server.start()

    body = json.dumps(webhook_payload).encode()
    signature = _sign(body, client_secret)

    try:
        async with ClientSession() as session:
            async with session.post(
                "http://127.0.0.1:18080/webhooks/hubspot",
                data=body,
                headers={"X-HubSpot-Signature": signature},
            ) as resp:
                assert resp.status == 200
                text = await resp.text()
                assert text == "OK"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_webhook_server_rejects_invalid_signature(
    client_secret: str, portal_id: str, webhook_payload: dict[str, Any]
) -> None:
    server = WebhookServer(client_secret=client_secret, portal_id=portal_id, port=18081)
    await server.start()

    body = json.dumps(webhook_payload).encode()
    bad_signature = _sign(body, "wrong-secret")

    try:
        async with ClientSession() as session:
            async with session.post(
                "http://127.0.0.1:18081/webhooks/hubspot",
                data=body,
                headers={"X-HubSpot-Signature": bad_signature},
            ) as resp:
                assert resp.status == 401
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_webhook_server_rejects_missing_signature(
    client_secret: str, portal_id: str, webhook_payload: dict[str, Any]
) -> None:
    server = WebhookServer(client_secret=client_secret, portal_id=portal_id, port=18082)
    await server.start()

    body = json.dumps(webhook_payload).encode()

    try:
        async with ClientSession() as session:
            async with session.post(
                "http://127.0.0.1:18082/webhooks/hubspot",
                data=body,
            ) as resp:
                assert resp.status == 401
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_webhook_server_rejects_invalid_json(
    client_secret: str, portal_id: str
) -> None:
    server = WebhookServer(client_secret=client_secret, portal_id=portal_id, port=18083)
    await server.start()

    body = b"not-json"
    signature = _sign(body, client_secret)

    try:
        async with ClientSession() as session:
            async with session.post(
                "http://127.0.0.1:18083/webhooks/hubspot",
                data=body,
                headers={"X-HubSpot-Signature": signature},
            ) as resp:
                assert resp.status == 400
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_webhook_server_queues_event(
    client_secret: str, portal_id: str, webhook_payload: dict[str, Any]
) -> None:
    server = WebhookServer(client_secret=client_secret, portal_id=portal_id, port=18084)
    await server.start()

    body = json.dumps(webhook_payload).encode()
    signature = _sign(body, client_secret)

    try:
        async with ClientSession() as session:
            async with session.post(
                "http://127.0.0.1:18084/webhooks/hubspot",
                data=body,
                headers={"X-HubSpot-Signature": signature},
            ) as resp:
                assert resp.status == 200

        # Give the processor a moment to dequeue
        await asyncio.sleep(0.3)
        assert server.event_queue.empty()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_webhook_server_handles_batch_events(
    client_secret: str, portal_id: str
) -> None:
    server = WebhookServer(client_secret=client_secret, portal_id=portal_id, port=18085)
    await server.start()

    events = [
        {"eventId": 1, "subscriptionType": "contact.propertyChange", "objectId": 101},
        {"eventId": 2, "subscriptionType": "company.creation", "objectId": 102},
    ]
    body = json.dumps(events).encode()
    signature = _sign(body, client_secret)

    try:
        async with ClientSession() as session:
            async with session.post(
                "http://127.0.0.1:18085/webhooks/hubspot",
                data=body,
                headers={"X-HubSpot-Signature": signature},
            ) as resp:
                assert resp.status == 200

        await asyncio.sleep(0.3)
        assert server.event_queue.empty()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_webhook_server_graceful_shutdown(
    client_secret: str, portal_id: str
) -> None:
    server = WebhookServer(client_secret=client_secret, portal_id=portal_id, port=18086)
    await server.start()
    assert server._processor_task is not None
    assert not server._processor_task.done()

    await server.stop()
    assert server._processor_task.done() or server._processor_task.cancelled()


# ---------------------------------------------------------------------------
# WebhookEventProcessor tests
# ---------------------------------------------------------------------------


def test_event_processor_routes_contact_property_change() -> None:
    processor = WebhookEventProcessor()
    with patch("hubspot_agent.webhooks.load_portal_config") as mock_load, \
         patch("hubspot_agent.webhooks.dispatch_agent") as mock_dispatch, \
         patch("hubspot_agent.webhooks.emit_trace"):
        mock_load.return_value = PortalConfig(
            portal_id="123", token="tok", tier="Professional"
        )
        mock_dispatch.return_value.status = "success"
        mock_dispatch.return_value.model_dump.return_value = {
            "agent_name": "objects",
            "status": "success",
            "data": {},
        }

        event = {
            "subscriptionType": "contact.propertyChange",
            "objectId": 101,
        }
        result = processor.process(event, "123")

        assert result["status"] == "success"
        assert result["agent"] == "objects"
        mock_dispatch.assert_called_once()
        call_kwargs = mock_dispatch.call_args.kwargs
        assert call_kwargs["agent_name"] == "objects"
        assert call_kwargs["mode"] == "preview"


def test_event_processor_returns_error_when_no_portal_config() -> None:
    processor = WebhookEventProcessor()
    with patch("hubspot_agent.webhooks.load_portal_config", return_value=None), \
         patch("hubspot_agent.webhooks.emit_trace"):
        event = {"subscriptionType": "contact.creation", "objectId": 1}
        result = processor.process(event, "999")
        assert result["status"] == "error"
        assert "Portal config not found" in result["reason"]


def test_event_processor_returns_unrouted_for_unknown_event() -> None:
    processor = WebhookEventProcessor()
    with patch("hubspot_agent.webhooks.emit_trace"):
        event = {"subscriptionType": "unknown.event", "objectId": 1}
        result = processor.process(event, "123")
        assert result["status"] == "unrouted"
        assert result["reason"] == "No agent mapping found"


# ---------------------------------------------------------------------------
# WebhookSubscriptionManager tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscription_manager_list_subscriptions(mock_portal: PortalConfig) -> None:
    manager = WebhookSubscriptionManager()
    with patch.object(manager, "_app_id", return_value="app_123"), \
         patch("hubspot_agent.webhooks.HubSpotClient") as MockClient:
        instance = MockClient.return_value
        instance.get = AsyncMock(return_value=APIResponse(
            status_code=200,
            body={"results": [{"id": "sub_1", "subscriptionType": "contact.creation"}]},
            headers={},
        ))
        instance.close = AsyncMock()

        results = await manager.list_subscriptions(mock_portal)
        assert len(results) == 1
        assert results[0]["id"] == "sub_1"
        instance.get.assert_called_once_with(
            "/webhooks/v1/app_123/subscriptions",
            portal_id="12345",
        )


@pytest.mark.asyncio
async def test_subscription_manager_create_subscription(mock_portal: PortalConfig) -> None:
    manager = WebhookSubscriptionManager()
    with patch.object(manager, "_app_id", return_value="app_123"), \
         patch("hubspot_agent.webhooks.HubSpotClient") as MockClient:
        instance = MockClient.return_value
        instance.post = AsyncMock(return_value=APIResponse(
            status_code=200,
            body={"id": "sub_new", "subscriptionType": "contact.creation"},
            headers={},
        ))
        instance.close = AsyncMock()

        result = await manager.create_subscription(
            mock_portal, "contact.creation", "https://example.com/webhook"
        )
        assert result["id"] == "sub_new"
        assert manager._local_urls["sub_new"] == "https://example.com/webhook"
        instance.post.assert_called_once_with(
            "/webhooks/v1/app_123/subscriptions",
            portal_id="12345",
            body={"subscriptionType": "contact.creation", "enabled": True},
        )


@pytest.mark.asyncio
async def test_subscription_manager_delete_subscription(mock_portal: PortalConfig) -> None:
    manager = WebhookSubscriptionManager()
    manager._local_urls["sub_1"] = "https://example.com/webhook"
    with patch.object(manager, "_app_id", return_value="app_123"), \
         patch("hubspot_agent.webhooks.HubSpotClient") as MockClient:
        instance = MockClient.return_value
        instance.delete = AsyncMock(return_value=APIResponse(
            status_code=204,
            body={},
            headers={},
        ))
        instance.close = AsyncMock()

        result = await manager.delete_subscription(mock_portal, "sub_1")
        assert "sub_1" not in manager._local_urls
        instance.delete.assert_called_once_with(
            "/webhooks/v1/app_123/subscriptions/sub_1",
            portal_id="12345",
        )


def test_subscription_manager_raises_when_no_app_id(mock_portal: PortalConfig) -> None:
    manager = WebhookSubscriptionManager()
    with patch("hubspot_agent.webhooks.load_app_credentials", return_value=None):
        with pytest.raises(ValueError, match="App credentials not configured"):
            manager._app_id()


# ---------------------------------------------------------------------------
# server.py CLI tests
# ---------------------------------------------------------------------------


def test_server_cli_missing_portal_id() -> None:
    from hubspot_agent.server import main
    with patch.dict(os.environ, {}, clear=True), \
         patch("hubspot_agent.server.detect_default_portal", return_value=None):
        rc = main(["--port", "18090"])
    assert rc == 1


def test_server_cli_missing_credentials() -> None:
    from hubspot_agent.server import main
    with patch.dict(os.environ, {"HUBSPOT_PORTAL_ID": "123"}, clear=True), \
         patch("hubspot_agent.server.load_portal_config", return_value=PortalConfig(
             portal_id="123", token="tok", tier="Professional"
         )), \
         patch("hubspot_agent.server.load_app_credentials", return_value=None):
        rc = main(["--portal-id", "123"])
    assert rc == 1


def test_server_cli_success() -> None:
    from hubspot_agent.server import main
    with patch.dict(os.environ, {"HUBSPOT_PORTAL_ID": "123"}, clear=True), \
         patch("hubspot_agent.server.load_portal_config", return_value=PortalConfig(
             portal_id="123", token="tok", tier="Professional"
         )), \
         patch("hubspot_agent.server.load_app_credentials", return_value={
             "client_id": "cid", "client_secret": "csec", "app_id": "aid"
         }), \
         patch("hubspot_agent.server.WebhookServer") as MockServer:
        instance = MockServer.return_value
        instance.run = AsyncMock()
        rc = main(["--portal-id", "123", "--port", "18091"])
    assert rc == 0
    MockServer.assert_called_once_with(
        client_secret="csec",
        portal_id="123",
        host="0.0.0.0",
        port=18091,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AsyncMock:
    def __init__(self, return_value: Any = None):
        self.return_value = return_value
        self.call_args = None
        self.call_count = 0

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_args = type("Args", (), {"args": args, "kwargs": kwargs})()
        self.call_count += 1
        return self.return_value

    def assert_called_once(self) -> None:
        assert self.call_count == 1

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        assert self.call_count == 1
        assert self.call_args.args == args
        assert self.call_args.kwargs == kwargs
