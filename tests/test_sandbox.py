import os
from unittest.mock import patch

import pytest

from hubspot_agent.config import PortalConfig
from hubspot_agent.models import AgentResult, RiskLevel
from hubspot_agent.plan import DAGPlan, PlanNode
from hubspot_agent.sandbox import (
    BehaviorDiff,
    SandboxResult,
    SandboxRunner,
    _SandboxPlanExecutor,
    build_sandbox_offer_prompt,
    format_sandbox_result,
    get_sandbox_portal_config,
    should_offer_sandbox,
)


@pytest.fixture
def mock_portal():
    return PortalConfig(portal_id="123", token="test-token", tier="Professional")


@pytest.fixture
def sandbox_portal():
    return PortalConfig(portal_id="4567890123", token="sandbox-token", tier="Professional")


@pytest.fixture
def high_risk_plan():
    return DAGPlan(
        plan_id="plan-high",
        nodes=[
            PlanNode(
                node_id="node-1",
                agent="objects",
                action="create",
                payload_summary={"text": "create a contact"},
                risk_level=RiskLevel.HIGH,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=5,
    )


@pytest.fixture
def destructive_plan():
    return DAGPlan(
        plan_id="plan-del",
        nodes=[
            PlanNode(
                node_id="node-1",
                agent="objects",
                action="delete",
                payload_summary={"text": "delete contact"},
                risk_level=RiskLevel.DESTRUCTIVE,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.DESTRUCTIVE,
        estimated_duration_seconds=5,
    )


@pytest.fixture
def low_risk_plan():
    return DAGPlan(
        plan_id="plan-low",
        nodes=[
            PlanNode(
                node_id="node-1",
                agent="objects",
                action="read",
                payload_summary={"text": "find contacts"},
                risk_level=RiskLevel.LOW,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.LOW,
        estimated_duration_seconds=5,
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_behavior_diff_defaults():
    diff = BehaviorDiff()
    assert diff.matches == {}
    assert diff.mismatches == []
    assert diff.missing == []
    assert diff.extra == []


def test_behavior_diff_with_data():
    diff = BehaviorDiff(
        matches={"node-1.text": "create a contact"},
        mismatches=[
            {"node_id": "node-1", "field": "status", "expected": "active", "actual": "inactive"}
        ],
        missing=["node-1.id"],
        extra=["node-1.debug"],
    )
    assert diff.matches["node-1.text"] == "create a contact"
    assert len(diff.mismatches) == 1
    assert diff.missing == ["node-1.id"]
    assert diff.extra == ["node-1.debug"]


def test_sandbox_result_defaults():
    result = SandboxResult(
        plan_executed=True,
        behavior_diff=BehaviorDiff(),
        warnings=[],
        sandbox_portal_id="4567890123",
    )
    assert result.plan_executed is True
    assert result.warnings == []
    assert result.sandbox_portal_id == "4567890123"


# ---------------------------------------------------------------------------
# SandboxRunner tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_in_sandbox_success(sandbox_portal, high_risk_plan):
    def fake_dispatch(**kwargs):
        return AgentResult(
            agent_name="objects",
            status="success",
            data={"text": "create a contact", "id": "123"},
        )

    runner = SandboxRunner(dispatch_agent_fn=fake_dispatch)
    result = await runner.preview_in_sandbox(high_risk_plan, sandbox_portal)

    assert result.plan_executed is True
    assert result.sandbox_portal_id == "4567890123"
    assert result.warnings == []
    assert result.behavior_diff.matches == {"node-1.text": "create a contact"}
    assert result.behavior_diff.extra == ["node-1.id"]


@pytest.mark.asyncio
async def test_preview_in_sandbox_mismatch(sandbox_portal):
    plan = DAGPlan(
        plan_id="plan-mismatch",
        nodes=[
            PlanNode(
                node_id="node-1",
                agent="objects",
                action="update",
                payload_summary={"text": "update contact", "status": "active"},
                risk_level=RiskLevel.HIGH,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=5,
    )

    def fake_dispatch(**kwargs):
        return AgentResult(
            agent_name="objects",
            status="success",
            data={"text": "update contact", "status": "inactive"},
        )

    runner = SandboxRunner(dispatch_agent_fn=fake_dispatch)
    result = await runner.preview_in_sandbox(plan, sandbox_portal)

    assert result.plan_executed is True
    assert len(result.behavior_diff.mismatches) == 1
    mismatch = result.behavior_diff.mismatches[0]
    assert mismatch["field"] == "status"
    assert mismatch["expected"] == "active"
    assert mismatch["actual"] == "inactive"


@pytest.mark.asyncio
async def test_preview_in_sandbox_extra_fields(sandbox_portal):
    plan = DAGPlan(
        plan_id="plan-extra",
        nodes=[
            PlanNode(
                node_id="node-1",
                agent="objects",
                action="create",
                payload_summary={"text": "create a contact"},
                risk_level=RiskLevel.HIGH,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=5,
    )

    def fake_dispatch(**kwargs):
        return AgentResult(
            agent_name="objects",
            status="success",
            data={"text": "create a contact", "unexpected": "value"},
        )

    runner = SandboxRunner(dispatch_agent_fn=fake_dispatch)
    result = await runner.preview_in_sandbox(plan, sandbox_portal)

    assert result.behavior_diff.matches == {"node-1.text": "create a contact"}
    assert result.behavior_diff.extra == ["node-1.unexpected"]


@pytest.mark.asyncio
async def test_preview_in_sandbox_warning_on_error(sandbox_portal, high_risk_plan):
    def fake_dispatch(**kwargs):
        return AgentResult(
            agent_name="objects",
            status="error",
            error_message="API rate limited",
        )

    runner = SandboxRunner(dispatch_agent_fn=fake_dispatch)
    result = await runner.preview_in_sandbox(high_risk_plan, sandbox_portal)

    assert result.plan_executed is False
    assert len(result.warnings) == 1
    assert "API rate limited" in result.warnings[0]


@pytest.mark.asyncio
async def test_preview_in_sandbox_no_dispatch_raises(high_risk_plan, sandbox_portal):
    runner = SandboxRunner(dispatch_agent_fn=None)
    with pytest.raises(RuntimeError, match="No dispatch agent function"):
        await runner.preview_in_sandbox(high_risk_plan, sandbox_portal)


@pytest.mark.asyncio
async def test_preview_in_sandbox_async_dispatch(sandbox_portal, high_risk_plan):
    async def async_dispatch(**kwargs):
        return AgentResult(
            agent_name="objects",
            status="success",
            data={"text": "create a contact"},
        )

    runner = SandboxRunner(dispatch_agent_fn=async_dispatch)
    result = await runner.preview_in_sandbox(high_risk_plan, sandbox_portal)
    assert result.plan_executed is True


@pytest.mark.asyncio
async def test_preview_in_sandbox_multi_node(sandbox_portal):
    plan = DAGPlan(
        plan_id="plan-multi",
        nodes=[
            PlanNode(
                node_id="node-1",
                agent="objects",
                action="create",
                payload_summary={"text": "create contact"},
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="node-2",
                agent="lists",
                action="create",
                payload_summary={"text": "create list"},
                risk_level=RiskLevel.HIGH,
            ),
        ],
        edges=[("node-1", "node-2")],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=10,
    )

    def fake_dispatch(**kwargs):
        agent = kwargs["agent_name"]
        return AgentResult(
            agent_name=agent,
            status="success",
            data={"text": kwargs.get("payload", {}).get("text", "")},
        )

    runner = SandboxRunner(dispatch_agent_fn=fake_dispatch)
    result = await runner.preview_in_sandbox(plan, sandbox_portal)

    assert result.plan_executed is True
    assert len(result.behavior_diff.matches) == 2
    assert "node-1.text" in result.behavior_diff.matches
    assert "node-2.text" in result.behavior_diff.matches


# ---------------------------------------------------------------------------
# _SandboxPlanExecutor tests
# ---------------------------------------------------------------------------


def test_sandbox_plan_executor_never_buffers(high_risk_plan):
    executor = _SandboxPlanExecutor()
    ordered = executor._topological_sort(high_risk_plan.nodes, high_risk_plan.edges)
    assert not executor._should_start_buffer(
        high_risk_plan.nodes[0], ordered, 0
    )


def test_sandbox_plan_executor_injects_node_id():
    executor = _SandboxPlanExecutor()
    node = PlanNode(
        node_id="node-x",
        agent="objects",
        action="create",
        payload_summary={"text": "test"},
        risk_level=RiskLevel.HIGH,
    )
    payload = executor._build_node_payload(node, {})
    assert payload["_sandbox_node_id"] == "node-x"
    assert payload["text"] == "test"


# ---------------------------------------------------------------------------
# Orchestrator integration tests
# ---------------------------------------------------------------------------


def test_get_sandbox_portal_config_with_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("HUBSPOT_SANDBOX_PORTAL_ID", "1234567890")
    monkeypatch.setenv("HUBSPOT_TOKEN_1234567890", "token-abc")
    config = get_sandbox_portal_config()
    assert config is not None
    assert config.portal_id == "1234567890"
    assert config.token == "token-abc"


def test_get_sandbox_portal_config_without_env_var(monkeypatch):
    monkeypatch.delenv("HUBSPOT_SANDBOX_PORTAL_ID", raising=False)
    assert get_sandbox_portal_config() is None


def test_should_offer_sandbox_high_risk_with_config(monkeypatch, high_risk_plan):
    monkeypatch.setenv("HUBSPOT_SANDBOX_PORTAL_ID", "1234567890")
    monkeypatch.setenv("HUBSPOT_TOKEN_1234567890", "token-abc")
    assert should_offer_sandbox(high_risk_plan) is True


def test_should_offer_sandbox_destructive_with_config(monkeypatch, destructive_plan):
    monkeypatch.setenv("HUBSPOT_SANDBOX_PORTAL_ID", "1234567890")
    monkeypatch.setenv("HUBSPOT_TOKEN_1234567890", "token-abc")
    assert should_offer_sandbox(destructive_plan) is True


def test_should_offer_sandbox_low_risk(monkeypatch, low_risk_plan):
    monkeypatch.setenv("HUBSPOT_SANDBOX_PORTAL_ID", "1234567890")
    monkeypatch.setenv("HUBSPOT_TOKEN_1234567890", "token-abc")
    assert should_offer_sandbox(low_risk_plan) is False


def test_should_offer_sandbox_no_config(monkeypatch, high_risk_plan):
    monkeypatch.delenv("HUBSPOT_SANDBOX_PORTAL_ID", raising=False)
    assert should_offer_sandbox(high_risk_plan) is False


def test_build_sandbox_offer_prompt_with_config(monkeypatch, high_risk_plan):
    monkeypatch.setenv("HUBSPOT_SANDBOX_PORTAL_ID", "1234567890")
    monkeypatch.setenv("HUBSPOT_TOKEN_1234567890", "token-abc")
    prompt = build_sandbox_offer_prompt(high_risk_plan)
    assert "Sandbox Preview Available" in prompt
    assert "HIGH" in prompt
    assert "1234567890" in prompt
    assert "Approve sandbox preview?" in prompt


def test_build_sandbox_offer_prompt_without_config(monkeypatch, high_risk_plan):
    monkeypatch.delenv("HUBSPOT_SANDBOX_PORTAL_ID", raising=False)
    prompt = build_sandbox_offer_prompt(high_risk_plan)
    assert prompt == ""


def test_format_sandbox_result():
    result = SandboxResult(
        plan_executed=True,
        behavior_diff=BehaviorDiff(
            matches={"node-1.text": "create"},
            mismatches=[
                {"node_id": "node-1", "field": "status", "expected": "active", "actual": "inactive"}
            ],
            missing=["node-1.id"],
            extra=["node-1.debug"],
        ),
        warnings=["Node node-1 failed: timeout"],
        sandbox_portal_id="7890123456",
    )
    text = format_sandbox_result(result)
    assert "Sandbox Preview Result" in text
    assert "7890123456" in text
    assert "Plan executed" in text
    assert "Yes" in text
    assert "timeout" in text
    assert "Matches" in text
    assert "Mismatches" in text
    assert "node-1.status" in text
    assert "Missing" in text
    assert "node-1.id" in text
    assert "Extra" in text
    assert "node-1.debug" in text


def test_format_sandbox_result_no_warnings():
    result = SandboxResult(
        plan_executed=True,
        behavior_diff=BehaviorDiff(),
        warnings=[],
        sandbox_portal_id="7890123456",
    )
    text = format_sandbox_result(result)
    assert "Warnings:" not in text


# ---------------------------------------------------------------------------
# Orchestrator execute_plan integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_plan_with_sandbox_runs_preview_first(
    mock_portal, sandbox_portal, high_risk_plan
):
    from hubspot_agent.orchestrator import execute_plan

    call_log = []

    def fake_dispatch(**kwargs):
        portal_id = kwargs.get("portal_config", mock_portal).portal_id
        call_log.append(portal_id)
        return AgentResult(
            agent_name=kwargs["agent_name"],
            status="success",
            data={"text": kwargs.get("payload", {}).get("text", "")},
        )

    with patch("hubspot_agent.orchestrator.dispatch_agent", new=fake_dispatch):
        results = await execute_plan(
            high_risk_plan,
            mock_portal,
            trace_id="trace-1",
            sandbox_portal_config=sandbox_portal,
        )

    assert call_log[0] == "4567890123"
    assert call_log[1] == "123"
    assert len(results) == 1
    assert results[0].status == "success"


@pytest.mark.asyncio
async def test_execute_plan_without_sandbox_skips_preview(mock_portal, high_risk_plan):
    from hubspot_agent.orchestrator import execute_plan

    call_log = []

    def fake_dispatch(**kwargs):
        portal_id = kwargs.get("portal_config", mock_portal).portal_id
        call_log.append(portal_id)
        return AgentResult(
            agent_name=kwargs["agent_name"],
            status="success",
            data={"text": kwargs.get("payload", {}).get("text", "")},
        )

    with patch("hubspot_agent.orchestrator.dispatch_agent", new=fake_dispatch):
        results = await execute_plan(
            high_risk_plan,
            mock_portal,
            trace_id="trace-1",
        )

    assert call_log == ["123"]
    assert len(results) == 1
