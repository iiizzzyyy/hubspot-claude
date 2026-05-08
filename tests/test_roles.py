import json

import pytest

from hubspot_agent.models import RiskLevel
from hubspot_agent.roles import RoleConfig, RoleManager
from hubspot_agent.orchestrator import dispatch_agent, dispatch_agents_parallel
from hubspot_agent.config import PortalConfig, save_portal_config


# ---------------------------------------------------------------------------
# RoleConfig model
# ---------------------------------------------------------------------------


def test_role_config_defaults():
    role = RoleConfig(
        user_id="alice",
        allowed_agents=["objects", "properties"],
        max_risk_level=RiskLevel.MEDIUM,
        denied_tools=["delete_contact"],
    )
    assert role.user_id == "alice"
    assert role.allowed_agents == ["objects", "properties"]
    assert role.max_risk_level == RiskLevel.MEDIUM
    assert role.denied_tools == ["delete_contact"]


# ---------------------------------------------------------------------------
# RoleManager loading
# ---------------------------------------------------------------------------


def test_load_roles_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    roles = RoleManager.load_roles("123")
    assert roles == []


def test_load_roles_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    portal_dir = tmp_path / ".claude" / "hubspot" / "123"
    portal_dir.mkdir(parents=True)
    data = [
        {
            "user_id": "alice",
            "allowed_agents": ["objects"],
            "max_risk_level": "low",
            "denied_tools": [],
        },
        {
            "user_id": "bob",
            "allowed_agents": ["objects", "workflows"],
            "max_risk_level": "high",
            "denied_tools": ["delete_contact"],
        },
    ]
    (portal_dir / "roles.json").write_text(json.dumps(data))

    roles = RoleManager.load_roles("123")
    assert len(roles) == 2
    assert roles[0].user_id == "alice"
    assert roles[1].user_id == "bob"
    assert roles[1].max_risk_level == RiskLevel.HIGH


def test_load_roles_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    portal_dir = tmp_path / ".claude" / "hubspot" / "123"
    portal_dir.mkdir(parents=True)
    (portal_dir / "roles.json").write_text("not json")

    roles = RoleManager.load_roles("123")
    assert roles == []


def test_load_roles_non_list(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    portal_dir = tmp_path / ".claude" / "hubspot" / "123"
    portal_dir.mkdir(parents=True)
    (portal_dir / "roles.json").write_text('{"user_id": "alice"}')

    roles = RoleManager.load_roles("123")
    assert roles == []


# ---------------------------------------------------------------------------
# RoleManager.can_dispatch — permissive defaults
# ---------------------------------------------------------------------------


def test_can_dispatch_no_roles_loaded():
    manager = RoleManager([])
    assert manager.can_dispatch("alice", "objects", RiskLevel.LOW) is True
    assert manager.can_dispatch("alice", "objects", RiskLevel.DESTRUCTIVE) is True


def test_can_dispatch_none_user_id():
    manager = RoleManager([
        RoleConfig(user_id="alice", allowed_agents=["objects"], max_risk_level=RiskLevel.LOW, denied_tools=[]),
    ])
    assert manager.can_dispatch(None, "objects", RiskLevel.LOW) is True


def test_can_dispatch_user_not_in_roles():
    manager = RoleManager([
        RoleConfig(user_id="alice", allowed_agents=["objects"], max_risk_level=RiskLevel.LOW, denied_tools=[]),
    ])
    assert manager.can_dispatch("bob", "workflows", RiskLevel.HIGH) is True


# ---------------------------------------------------------------------------
# RoleManager.can_dispatch — restrictive checks
# ---------------------------------------------------------------------------


def test_can_dispatch_denied_agent():
    manager = RoleManager([
        RoleConfig(user_id="alice", allowed_agents=["objects"], max_risk_level=RiskLevel.HIGH, denied_tools=[]),
    ])
    assert manager.can_dispatch("alice", "objects", RiskLevel.LOW) is True
    assert manager.can_dispatch("alice", "workflows", RiskLevel.LOW) is False


def test_can_dispatch_denied_risk_level():
    manager = RoleManager([
        RoleConfig(user_id="alice", allowed_agents=["objects", "workflows"], max_risk_level=RiskLevel.MEDIUM, denied_tools=[]),
    ])
    assert manager.can_dispatch("alice", "objects", RiskLevel.LOW) is True
    assert manager.can_dispatch("alice", "objects", RiskLevel.MEDIUM) is True
    assert manager.can_dispatch("alice", "objects", RiskLevel.HIGH) is False
    assert manager.can_dispatch("alice", "objects", RiskLevel.DESTRUCTIVE) is False


def test_can_dispatch_denied_tool():
    manager = RoleManager([
        RoleConfig(user_id="alice", allowed_agents=["objects"], max_risk_level=RiskLevel.HIGH, denied_tools=["delete_contact"]),
    ])
    assert manager.can_dispatch("alice", "objects", RiskLevel.LOW, tool_name="create_contact") is True
    assert manager.can_dispatch("alice", "objects", RiskLevel.LOW, tool_name="delete_contact") is False


def test_can_dispatch_no_tool_name():
    manager = RoleManager([
        RoleConfig(user_id="alice", allowed_agents=["objects"], max_risk_level=RiskLevel.HIGH, denied_tools=["delete_contact"]),
    ])
    assert manager.can_dispatch("alice", "objects", RiskLevel.LOW) is True


# ---------------------------------------------------------------------------
# RoleManager.for_portal integration
# ---------------------------------------------------------------------------


def test_for_portal_loads_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    portal_dir = tmp_path / ".claude" / "hubspot" / "456"
    portal_dir.mkdir(parents=True)
    data = [
        {
            "user_id": "alice",
            "allowed_agents": ["objects"],
            "max_risk_level": "low",
            "denied_tools": [],
        },
    ]
    (portal_dir / "roles.json").write_text(json.dumps(data))

    manager = RoleManager.for_portal("456")
    assert manager.can_dispatch("alice", "objects", RiskLevel.LOW) is True
    assert manager.can_dispatch("alice", "workflows", RiskLevel.LOW) is False


# ---------------------------------------------------------------------------
# Orchestrator integration — dispatch_agent
# ---------------------------------------------------------------------------


def test_dispatch_agent_no_portal_config_allows_all():
    result = dispatch_agent("objects", "find contacts")
    assert result.status == "preview"


def test_dispatch_agent_missing_roles_file_allows_all(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    save_portal_config(PortalConfig(portal_id="123", token="t"))
    config = PortalConfig(portal_id="123", token="t")

    result = dispatch_agent("objects", "find contacts", portal_config=config, user_id="alice")
    assert result.status == "preview"


def test_dispatch_agent_role_denied_agent(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    save_portal_config(PortalConfig(portal_id="123", token="t"))
    config = PortalConfig(portal_id="123", token="t")

    portal_dir = tmp_path / ".claude" / "hubspot" / "123"
    portal_dir.mkdir(parents=True)
    data = [
        {
            "user_id": "alice",
            "allowed_agents": ["properties"],
            "max_risk_level": "high",
            "denied_tools": [],
        },
    ]
    (portal_dir / "roles.json").write_text(json.dumps(data))

    result = dispatch_agent("objects", "find contacts", portal_config=config, user_id="alice")
    assert result.status == "error"
    assert "Role denied" in result.error_message
    assert "objects" in result.error_message


def test_dispatch_agent_role_denied_risk_level(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    save_portal_config(PortalConfig(portal_id="123", token="t"))
    config = PortalConfig(portal_id="123", token="t")

    portal_dir = tmp_path / ".claude" / "hubspot" / "123"
    portal_dir.mkdir(parents=True)
    data = [
        {
            "user_id": "alice",
            "allowed_agents": ["objects"],
            "max_risk_level": "low",
            "denied_tools": [],
        },
    ]
    (portal_dir / "roles.json").write_text(json.dumps(data))

    result = dispatch_agent(
        "objects",
        "delete contacts",
        portal_config=config,
        user_id="alice",
        risk_level=RiskLevel.HIGH,
    )
    assert result.status == "error"
    assert "Role denied" in result.error_message
    assert "high" in result.error_message


def test_dispatch_agent_user_not_in_roles_is_permissive(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    save_portal_config(PortalConfig(portal_id="123", token="t"))
    config = PortalConfig(portal_id="123", token="t")

    portal_dir = tmp_path / ".claude" / "hubspot" / "123"
    portal_dir.mkdir(parents=True)
    data = [
        {
            "user_id": "alice",
            "allowed_agents": ["properties"],
            "max_risk_level": "low",
            "denied_tools": [],
        },
    ]
    (portal_dir / "roles.json").write_text(json.dumps(data))

    result = dispatch_agent("objects", "find contacts", portal_config=config, user_id="bob")
    assert result.status == "preview"


# ---------------------------------------------------------------------------
# Orchestrator integration — dispatch_agents_parallel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_agents_parallel_role_denied(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    save_portal_config(PortalConfig(portal_id="123", token="t"))
    config = PortalConfig(portal_id="123", token="t")

    portal_dir = tmp_path / ".claude" / "hubspot" / "123"
    portal_dir.mkdir(parents=True)
    data = [
        {
            "user_id": "alice",
            "allowed_agents": ["properties"],
            "max_risk_level": "high",
            "denied_tools": [],
        },
    ]
    (portal_dir / "roles.json").write_text(json.dumps(data))

    results = await dispatch_agents_parallel(
        ["objects", "properties"],
        "find stuff",
        portal_config=config,
        user_id="alice",
    )
    assert len(results) == 2
    objects_result = [r for r in results if r.agent_name == "objects"][0]
    properties_result = [r for r in results if r.agent_name == "properties"][0]
    assert objects_result.status == "error"
    assert "Role denied" in objects_result.error_message
    assert properties_result.status == "preview"


@pytest.mark.asyncio
async def test_dispatch_agents_parallel_no_user_id_allows_all(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    save_portal_config(PortalConfig(portal_id="123", token="t"))
    config = PortalConfig(portal_id="123", token="t")

    portal_dir = tmp_path / ".claude" / "hubspot" / "123"
    portal_dir.mkdir(parents=True)
    data = [
        {
            "user_id": "alice",
            "allowed_agents": ["properties"],
            "max_risk_level": "low",
            "denied_tools": [],
        },
    ]
    (portal_dir / "roles.json").write_text(json.dumps(data))

    results = await dispatch_agents_parallel(
        ["objects", "properties"],
        "find stuff",
        portal_config=config,
    )
    assert all(r.status == "preview" for r in results)
