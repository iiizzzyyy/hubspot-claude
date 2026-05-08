import pytest
from datetime import datetime, timezone

from hubspot_agent.hooks import (
    HookContext,
    HookEvent,
    HookRegistry,
    HookResult,
    get_registry,
    load_hooks_config,
    register_hooks_from_config,
    reset_registry,
    run_hooks,
    _resolve_handler,
)
from hubspot_agent.models import PreviewResult, RiskLevel
from hubspot_agent.orchestrator import (
    dispatch_agents_parallel,
    record_action_completion_with_hooks,
    run_post_approval_hooks,
    run_post_write_hooks,
    run_pre_approval_hooks,
    run_pre_write_hooks,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


# ---------------------------------------------------------------------------
# Hook primitives
# ---------------------------------------------------------------------------


def test_hook_event_values():
    assert HookEvent.PRE_WRITE == "pre_write"
    assert HookEvent.POST_WRITE == "post_write"
    assert HookEvent.PRE_APPROVAL == "pre_approval"
    assert HookEvent.POST_APPROVAL == "post_approval"


def test_hook_context_defaults():
    ctx = HookContext()
    assert ctx.portal_id is None
    assert ctx.agent_name is None
    assert ctx.action_id is None
    assert ctx.payload == {}
    assert ctx.preview_result is None
    assert ctx.user_id is None
    assert isinstance(ctx.timestamp, datetime)


def test_hook_context_extra_fields():
    ctx = HookContext(custom_field="extra")
    assert ctx.custom_field == "extra"


def test_hook_result_defaults():
    result = HookResult()
    assert result.allowed is True
    assert result.modified_payload is None
    assert result.message is None


# ---------------------------------------------------------------------------
# HookRegistry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_run_single_handler():
    registry = HookRegistry()

    async def handler(ctx):
        return HookResult(allowed=True, modified_payload={"modified": True})

    registry.register(HookEvent.PRE_WRITE, handler)
    ctx = HookContext(payload={"original": True})
    result = await registry.run(HookEvent.PRE_WRITE, ctx)
    assert result.allowed is True
    assert result.modified_payload == {"modified": True}


@pytest.mark.asyncio
async def test_registry_run_multiple_handlers_mutate_payload():
    registry = HookRegistry()

    async def add_field(ctx):
        payload = dict(ctx.payload)
        payload["field_a"] = 1
        return HookResult(allowed=True, modified_payload=payload)

    async def add_another(ctx):
        payload = dict(ctx.payload)
        payload["field_b"] = 2
        return HookResult(allowed=True, modified_payload=payload)

    registry.register(HookEvent.PRE_WRITE, add_field)
    registry.register(HookEvent.PRE_WRITE, add_another)

    ctx = HookContext(payload={"base": 0})
    result = await registry.run(HookEvent.PRE_WRITE, ctx)
    assert result.allowed is True
    assert result.modified_payload == {"base": 0, "field_a": 1, "field_b": 2}


@pytest.mark.asyncio
async def test_registry_run_stops_on_blocked():
    registry = HookRegistry()

    async def blocker(ctx):
        return HookResult(allowed=False, message="quota exceeded")

    async def never_called(ctx):
        return HookResult(allowed=True, modified_payload={"should": "not_exist"})

    registry.register(HookEvent.PRE_WRITE, blocker)
    registry.register(HookEvent.PRE_WRITE, never_called)

    ctx = HookContext(payload={})
    result = await registry.run(HookEvent.PRE_WRITE, ctx)
    assert result.allowed is False
    assert result.message == "quota exceeded"
    assert result.modified_payload is None


@pytest.mark.asyncio
async def test_registry_run_catches_exception_and_continues():
    registry = HookRegistry()

    async def failer(ctx):
        raise RuntimeError("boom")

    async def recover(ctx):
        return HookResult(allowed=True, modified_payload={"recovered": True})

    registry.register(HookEvent.PRE_WRITE, failer)
    registry.register(HookEvent.PRE_WRITE, recover)

    ctx = HookContext(payload={})
    result = await registry.run(HookEvent.PRE_WRITE, ctx)
    assert result.allowed is True
    assert result.modified_payload == {"recovered": True}


@pytest.mark.asyncio
async def test_registry_run_no_handlers():
    registry = HookRegistry()
    ctx = HookContext(payload={"x": 1})
    result = await registry.run(HookEvent.PRE_WRITE, ctx)
    assert result.allowed is True
    assert result.modified_payload is None


# ---------------------------------------------------------------------------
# Global registry helpers
# ---------------------------------------------------------------------------


def test_get_registry_returns_singleton():
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2


def test_reset_registry_creates_new_instance():
    r1 = get_registry()
    reset_registry()
    r2 = get_registry()
    assert r1 is not r2


@pytest.mark.asyncio
async def test_run_hooks_uses_default_registry():
    reset_registry()
    registry = get_registry()

    async def handler(ctx):
        return HookResult(allowed=True, modified_payload={"from_default": True})

    registry.register(HookEvent.POST_WRITE, handler)
    result = await run_hooks(
        HookEvent.POST_WRITE,
        portal_id="123",
        agent_name="objects",
        payload={"test": True},
    )
    assert result.allowed is True
    assert result.modified_payload == {"from_default": True}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_load_hooks_config_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config = load_hooks_config("123")
    assert config == {}


def test_load_hooks_config_valid(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    portal_dir = tmp_path / ".claude" / "hubspot" / "123"
    portal_dir.mkdir(parents=True, exist_ok=True)
    hooks_file = portal_dir / "hooks.json"
    hooks_file.write_text('{"pre_write": [{"module": "a", "func": "b"}]}')
    config = load_hooks_config("123")
    assert config == {"pre_write": [{"module": "a", "func": "b"}]}


def test_load_hooks_config_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    portal_dir = tmp_path / ".claude" / "hubspot" / "123"
    portal_dir.mkdir(parents=True, exist_ok=True)
    hooks_file = portal_dir / "hooks.json"
    hooks_file.write_text("not json")
    config = load_hooks_config("123")
    assert config == {}


# ---------------------------------------------------------------------------
# register_hooks_from_config
# ---------------------------------------------------------------------------


def test_register_hooks_from_config_skips_invalid_event():
    reset_registry()
    register_hooks_from_config("123", registry=get_registry())
    # no config file, should be no-op
    assert len(get_registry()._handlers[HookEvent.PRE_WRITE]) == 0


def test_register_hooks_from_config_resolves_handlers(monkeypatch):
    reset_registry()
    registry = get_registry()

    fake_handler = lambda ctx: None  # noqa: E731

    monkeypatch.setattr(
        "hubspot_agent.hooks._resolve_handler",
        lambda module, func: fake_handler,
    )

    monkeypatch.setattr(
        "hubspot_agent.hooks.load_hooks_config",
        lambda portal_id: {
            "pre_write": [{"module": "my_mod", "func": "my_handler"}],
        },
    )

    register_hooks_from_config("123", registry=registry)
    assert len(registry._handlers[HookEvent.PRE_WRITE]) == 1


def test_register_hooks_from_config_skips_bad_entries(monkeypatch):
    reset_registry()
    registry = get_registry()

    monkeypatch.setattr(
        "hubspot_agent.hooks.load_hooks_config",
        lambda portal_id: {
            "pre_write": [
                {"module": "my_mod"},
                "not-a-dict",
            ],
        },
    )

    register_hooks_from_config("123", registry=registry)
    assert len(registry._handlers[HookEvent.PRE_WRITE]) == 0


# ---------------------------------------------------------------------------
# Orchestrator integration — approval hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pre_approval_hooks():
    reset_registry()
    registry = get_registry()

    async def handler(ctx):
        return HookResult(
            allowed=False,
            message="needs manager approval",
            modified_payload=None,
        )

    registry.register(HookEvent.PRE_APPROVAL, handler)

    preview = PreviewResult(
        preview={"affected": [{"id": "1"}]},
        impact_count=1,
        risk_level=RiskLevel.HIGH,
        proposed_payload={"endpoint": "/test"},
    )
    result = await run_pre_approval_hooks(
        preview_result=preview,
        portal_id="123",
        agent_name="objects",
        payload={"x": 1},
    )
    assert result.allowed is False
    assert result.message == "needs manager approval"


@pytest.mark.asyncio
async def test_run_post_approval_hooks():
    reset_registry()
    registry = get_registry()

    async def handler(ctx):
        return HookResult(allowed=True, modified_payload={"notified": True})

    registry.register(HookEvent.POST_APPROVAL, handler)

    result = await run_post_approval_hooks(
        portal_id="123",
        agent_name="objects",
        action_id="abc",
        payload={"x": 1},
    )
    assert result.allowed is True
    assert result.modified_payload == {"notified": True}


# ---------------------------------------------------------------------------
# Orchestrator integration — write hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pre_write_hooks():
    reset_registry()
    registry = get_registry()

    async def handler(ctx):
        payload = dict(ctx.payload)
        payload["enriched"] = True
        return HookResult(allowed=True, modified_payload=payload)

    registry.register(HookEvent.PRE_WRITE, handler)

    result = await run_pre_write_hooks(
        portal_id="123",
        agent_name="objects",
        payload={"email": "a@example.com"},
    )
    assert result.allowed is True
    assert result.modified_payload == {"email": "a@example.com", "enriched": True}


@pytest.mark.asyncio
async def test_run_post_write_hooks():
    reset_registry()
    registry = get_registry()

    async def handler(ctx):
        return HookResult(allowed=True, message="mirrored to slack")

    registry.register(HookEvent.POST_WRITE, handler)

    result = await run_post_write_hooks(
        portal_id="123",
        agent_name="objects",
        action_id="abc",
        payload={"x": 1},
    )
    assert result.allowed is True
    assert result.message == "mirrored to slack"


@pytest.mark.asyncio
async def test_record_action_completion_with_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    reset_registry()
    registry = get_registry()

    post_write_called = False

    async def handler(ctx):
        nonlocal post_write_called
        post_write_called = True
        return HookResult(allowed=True)

    registry.register(HookEvent.POST_WRITE, handler)

    from hubspot_agent.ledger import ActionLedger

    expected_dir = tmp_path / ".claude" / "hubspot" / "123"
    ledger = ActionLedger("123", base_dir=expected_dir)
    ledger.start_action("x1", "objects", "create", {})

    await record_action_completion_with_hooks(
        "123",
        "x1",
        {"status": "success"},
        agent_name="objects",
        payload={"email": "a@example.com"},
        registry=registry,
    )

    assert post_write_called is True
    in_flight = ledger.get_in_flight()
    assert in_flight == []


# ---------------------------------------------------------------------------
# Orchestrator integration — dispatch_agents_parallel pre_write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_agents_parallel_pre_write_blocked():
    reset_registry()
    registry = get_registry()

    async def blocker(ctx):
        return HookResult(allowed=False, message="maintenance window")

    registry.register(HookEvent.PRE_WRITE, blocker)

    results = await dispatch_agents_parallel(
        ["objects"],
        "create contact",
        mode="execute",
        payload={"email": "a@example.com"},
    )
    assert len(results) == 1
    assert results[0].status == "blocked"
    assert "maintenance window" in results[0].error_message


@pytest.mark.asyncio
async def test_dispatch_agents_parallel_pre_write_payload_modified():
    reset_registry()
    registry = get_registry()

    async def enricher(ctx):
        payload = dict(ctx.payload)
        payload["lifecyclestage"] = "subscriber"
        return HookResult(allowed=True, modified_payload=payload)

    registry.register(HookEvent.PRE_WRITE, enricher)

    results = await dispatch_agents_parallel(
        ["objects"],
        "create contact",
        mode="execute",
        payload={"email": "a@example.com"},
    )
    assert len(results) == 1
    assert results[0].status == "ready"
    full_prompt = results[0].data.get("full_prompt", "")
    assert "subscriber" in full_prompt


@pytest.mark.asyncio
async def test_dispatch_agents_parallel_preview_mode_ignores_hooks():
    reset_registry()
    registry = get_registry()

    async def blocker(ctx):
        return HookResult(allowed=False, message="should not fire")

    registry.register(HookEvent.PRE_WRITE, blocker)

    results = await dispatch_agents_parallel(
        ["objects"],
        "find contacts",
        mode="preview",
    )
    assert len(results) == 1
    assert results[0].status == "preview"


# ---------------------------------------------------------------------------
# _resolve_handler
# ---------------------------------------------------------------------------


def test_resolve_handler_missing_module():
    assert _resolve_handler("nonexistent_module_xyz", "foo") is None


def test_resolve_handler_missing_attr():
    assert _resolve_handler("hubspot_agent.hooks", "nonexistent_attr_xyz") is None


def test_resolve_handler_valid():
    from hubspot_agent.hooks import run_hooks
    assert _resolve_handler("hubspot_agent.hooks", "run_hooks") is run_hooks
