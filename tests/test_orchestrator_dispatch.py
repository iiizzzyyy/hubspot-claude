from hubspot_agent.orchestrator import dispatch_agent, record_action_completion
from hubspot_agent.config import PortalConfig, save_portal_config


def test_dispatch_agent_unknown():
    result = dispatch_agent("nonexistent", "do something")
    assert result.status == "error"
    assert "Unknown agent" in result.error_message


def test_dispatch_agent_preview_mode():
    result = dispatch_agent("objects", "find contacts in northeast")
    assert result.status == "preview"
    assert "find contacts in northeast" in result.data["full_prompt"]
    assert "Mode: preview" in result.data["full_prompt"]


def test_dispatch_agent_execute_mode():
    result = dispatch_agent(
        "objects",
        "create a contact",
        mode="execute",
        payload={"properties": {"email": "test@example.com"}},
    )
    assert result.status == "ready"
    assert "Execute the following payload" in result.data["full_prompt"]
    assert "test@example.com" in result.data["full_prompt"]


def test_dispatch_agent_execute_records_action_id(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    save_portal_config(PortalConfig(portal_id="123", token="t"))

    result = dispatch_agent(
        "objects",
        "create a contact",
        portal_config=PortalConfig(portal_id="123", token="t"),
        mode="execute",
        payload={"properties": {"email": "test@example.com"}},
    )
    assert result.status == "ready"
    assert "action_id" in result.data


def test_dispatch_agent_duplicate_detected(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    save_portal_config(PortalConfig(portal_id="123", token="t"))
    config = PortalConfig(portal_id="123", token="t")

    payload = {"properties": {"email": "test@example.com"}}

    first = dispatch_agent("objects", "create contact", portal_config=config, mode="execute", payload=payload)
    assert first.status == "ready"

    second = dispatch_agent("objects", "create contact", portal_config=config, mode="execute", payload=payload)
    assert second.status == "duplicate"
    assert "already in flight" in second.error_message
    assert second.data.get("duplicate_action_id") == first.data["action_id"]


def test_dispatch_agent_no_duplicate_for_different_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    save_portal_config(PortalConfig(portal_id="123", token="t"))
    config = PortalConfig(portal_id="123", token="t")

    dispatch_agent("objects", "create contact", portal_config=config, mode="execute", payload={"email": "a@example.com"})
    second = dispatch_agent("objects", "create contact", portal_config=config, mode="execute", payload={"email": "b@example.com"})
    assert second.status == "ready"


def test_record_action_completion(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    from hubspot_agent.ledger import ActionLedger

    expected_dir = tmp_path / ".claude" / "hubspot" / "123"
    ledger = ActionLedger("123", base_dir=expected_dir)
    ledger.start_action("x1", "objects", "create", {})

    record_action_completion("123", "x1", {"status": "success"})

    in_flight = ledger.get_in_flight()
    assert in_flight == []
