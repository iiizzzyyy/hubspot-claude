import pytest
from unittest.mock import AsyncMock

from hubspot_agent.config import PortalConfig
from hubspot_agent.models import AgentResult, RiskLevel
from hubspot_agent.plan import DAGPlan, PlanExecutor, PlanNode, WriteBuffer


@pytest.fixture
def mock_portal():
    return PortalConfig(portal_id="123", token="test-token", tier="Professional")


# ---------------------------------------------------------------------------
# WriteBuffer basics
# ---------------------------------------------------------------------------


def test_buffer_write_stores_operations():
    buffer = WriteBuffer()
    node = PlanNode(
        node_id="node-1",
        agent="objects",
        action="create",
        inputs={"object_type": "contacts"},
        risk_level=RiskLevel.HIGH,
    )
    payload = {"properties": {"email": "a@example.com"}}
    buffer.buffer_write(node, payload)
    assert len(buffer._buffer) == 1
    assert buffer._buffer[0][0].node_id == "node-1"
    assert buffer._buffer[0][1] == payload


def test_can_coalesce_objects_create():
    assert WriteBuffer.can_coalesce(
        PlanNode(
            node_id="n1", agent="objects", action="create", risk_level=RiskLevel.HIGH
        )
    )


def test_can_coalesce_lists_update():
    assert WriteBuffer.can_coalesce(
        PlanNode(
            node_id="n1", agent="lists", action="update", risk_level=RiskLevel.HIGH
        )
    )


def test_can_coalesce_properties_delete():
    assert WriteBuffer.can_coalesce(
        PlanNode(
            node_id="n1",
            agent="properties",
            action="delete",
            risk_level=RiskLevel.HIGH,
        )
    )


def test_can_coalesce_rejects_read():
    assert not WriteBuffer.can_coalesce(
        PlanNode(
            node_id="n1", agent="objects", action="read", risk_level=RiskLevel.LOW
        )
    )


def test_can_coalesce_rejects_workflows():
    assert not WriteBuffer.can_coalesce(
        PlanNode(
            node_id="n1", agent="workflows", action="create", risk_level=RiskLevel.HIGH
        )
    )


def test_can_coalesce_rejects_trigger():
    assert not WriteBuffer.can_coalesce(
        PlanNode(
            node_id="n1",
            agent="objects",
            action="trigger",
            risk_level=RiskLevel.MEDIUM,
        )
    )


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_group_by_endpoint_groups_by_object_type_and_action():
    buffer = WriteBuffer()
    buffer.buffer_write(
        PlanNode(
            node_id="n1",
            agent="objects",
            action="create",
            inputs={"object_type": "contacts"},
            risk_level=RiskLevel.HIGH,
        ),
        {"properties": {"email": "a@example.com"}},
    )
    buffer.buffer_write(
        PlanNode(
            node_id="n2",
            agent="objects",
            action="create",
            inputs={"object_type": "contacts"},
            risk_level=RiskLevel.HIGH,
        ),
        {"properties": {"email": "b@example.com"}},
    )
    buffer.buffer_write(
        PlanNode(
            node_id="n3",
            agent="objects",
            action="update",
            inputs={"object_type": "contacts"},
            risk_level=RiskLevel.HIGH,
        ),
        {"properties": {"email": "c@example.com"}},
    )
    buffer.buffer_write(
        PlanNode(
            node_id="n4",
            agent="objects",
            action="create",
            inputs={"object_type": "companies"},
            risk_level=RiskLevel.HIGH,
        ),
        {"properties": {"name": "Acme"}},
    )
    groups = buffer._group_by_endpoint()
    assert len(groups) == 3
    assert len(groups["contacts:/crm/v3/objects/contacts/batch/create"]) == 2
    assert len(groups["contacts:/crm/v3/objects/contacts/batch/update"]) == 1
    assert len(groups["companies:/crm/v3/objects/companies/batch/create"]) == 1


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_returns_results_for_all_buffered_nodes(mock_portal):
    mock_client = AsyncMock()
    mock_client.post.return_value = AsyncMock()
    mock_client.post.return_value.body = {
        "results": [{"id": "1"}, {"id": "2"}],
        "errors": [],
    }

    buffer = WriteBuffer(client=mock_client)
    node1 = PlanNode(
        node_id="n1",
        agent="objects",
        action="create",
        inputs={"object_type": "contacts"},
        risk_level=RiskLevel.HIGH,
    )
    node2 = PlanNode(
        node_id="n2",
        agent="objects",
        action="create",
        inputs={"object_type": "contacts"},
        risk_level=RiskLevel.HIGH,
    )
    buffer.buffer_write(node1, {"properties": {"email": "a@example.com"}})
    buffer.buffer_write(node2, {"properties": {"email": "b@example.com"}})

    results = await buffer.flush(mock_portal)
    assert len(results) == 2
    assert results[0].status == "success"
    assert results[0].data == {"id": "1"}
    assert results[1].status == "success"
    assert results[1].data == {"id": "2"}

    mock_client.post.assert_awaited_once_with(
        "/crm/v3/objects/contacts/batch/create",
        portal_id="123",
        body={
            "inputs": [
                {"properties": {"email": "a@example.com"}},
                {"properties": {"email": "b@example.com"}},
            ]
        },
        expected_scopes=["crm.objects.contacts.write"],
    )


@pytest.mark.asyncio
async def test_flush_error_attribution_by_index(mock_portal):
    mock_client = AsyncMock()
    mock_client.post.return_value = AsyncMock()
    mock_client.post.return_value.body = {
        "results": [{"id": "1"}, {}],
        "errors": [
            {"message": "Invalid email", "index": 1},
        ],
    }

    buffer = WriteBuffer(client=mock_client)
    node1 = PlanNode(
        node_id="n1",
        agent="objects",
        action="create",
        inputs={"object_type": "contacts"},
        risk_level=RiskLevel.HIGH,
    )
    node2 = PlanNode(
        node_id="n2",
        agent="objects",
        action="create",
        inputs={"object_type": "contacts"},
        risk_level=RiskLevel.HIGH,
    )
    buffer.buffer_write(node1, {"properties": {"email": "a@example.com"}})
    buffer.buffer_write(node2, {"properties": {"email": "bad"}})

    results = await buffer.flush(mock_portal)
    assert results[0].status == "success"
    assert results[0].data == {"id": "1"}
    assert results[1].status == "error"
    assert results[1].error_message == "Invalid email"


@pytest.mark.asyncio
async def test_flush_error_attribution_by_id(mock_portal):
    mock_client = AsyncMock()
    mock_client.post.return_value = AsyncMock()
    mock_client.post.return_value.body = {
        "results": [{"id": "1"}, {"id": "2"}],
        "errors": [
            {"message": "Not found", "id": "2"},
        ],
    }

    buffer = WriteBuffer(client=mock_client)
    node1 = PlanNode(
        node_id="n1",
        agent="objects",
        action="update",
        inputs={"object_type": "contacts", "object_id": "1"},
        risk_level=RiskLevel.HIGH,
    )
    node2 = PlanNode(
        node_id="n2",
        agent="objects",
        action="update",
        inputs={"object_type": "contacts", "object_id": "2"},
        risk_level=RiskLevel.HIGH,
    )
    buffer.buffer_write(
        node1, {"object_id": "1", "properties": {"email": "a@example.com"}}
    )
    buffer.buffer_write(
        node2, {"object_id": "2", "properties": {"email": "b@example.com"}}
    )

    results = await buffer.flush(mock_portal)
    assert results[0].status == "success"
    assert results[1].status == "error"
    assert results[1].error_message == "Not found"


@pytest.mark.asyncio
async def test_flush_all_nodes_error_on_exception(mock_portal):
    mock_client = AsyncMock()
    mock_client.post.side_effect = RuntimeError("Network failure")

    buffer = WriteBuffer(client=mock_client)
    node1 = PlanNode(
        node_id="n1",
        agent="objects",
        action="create",
        inputs={"object_type": "contacts"},
        risk_level=RiskLevel.HIGH,
    )
    node2 = PlanNode(
        node_id="n2",
        agent="objects",
        action="create",
        inputs={"object_type": "contacts"},
        risk_level=RiskLevel.HIGH,
    )
    buffer.buffer_write(node1, {})
    buffer.buffer_write(node2, {})

    results = await buffer.flush(mock_portal)
    assert len(results) == 2
    assert all(r.status == "error" for r in results)
    assert all("Network failure" in r.error_message for r in results)


@pytest.mark.asyncio
async def test_flush_empty_buffer(mock_portal):
    buffer = WriteBuffer()
    results = await buffer.flush(mock_portal)
    assert results == []


# ---------------------------------------------------------------------------
# PlanExecutor integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_buffers_consecutive_same_type_writes(mock_portal):
    mock_client = AsyncMock()
    mock_client.post.return_value = AsyncMock()
    mock_client.post.return_value.body = {
        "results": [{"id": "1"}, {"id": "2"}],
        "errors": [],
    }

    call_order = []

    def fake_dispatch(agent_name, user_request, portal_config, mode, payload, trace_id):
        call_order.append(agent_name)
        return AgentResult(agent_name=agent_name, status="success")

    executor = PlanExecutor(dispatch_agent_fn=fake_dispatch, client=mock_client)
    plan = DAGPlan(
        plan_id="plan-test",
        nodes=[
            PlanNode(
                node_id="a",
                agent="objects",
                action="create",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="b",
                agent="objects",
                action="create",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="c",
                agent="lists",
                action="add",
                risk_level=RiskLevel.MEDIUM,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=10,
    )

    results = await executor.execute(plan, mock_portal, "trace-123")
    assert call_order == ["lists"]
    assert len(results) == 3
    assert results[0].status == "success"
    assert results[0].data == {"id": "1"}
    assert results[1].status == "success"
    assert results[1].data == {"id": "2"}
    assert results[2].agent_name == "lists"
    assert results[2].status == "success"

    mock_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_executor_no_coalescing_for_non_write_nodes(mock_portal):
    call_order = []

    def fake_dispatch(agent_name, user_request, portal_config, mode, payload, trace_id):
        call_order.append(agent_name)
        return AgentResult(agent_name=agent_name, status="success")

    executor = PlanExecutor(dispatch_agent_fn=fake_dispatch)
    plan = DAGPlan(
        plan_id="plan-test",
        nodes=[
            PlanNode(
                node_id="a",
                agent="objects",
                action="read",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.LOW,
            ),
            PlanNode(
                node_id="b",
                agent="objects",
                action="read",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.LOW,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.LOW,
        estimated_duration_seconds=10,
    )

    results = await executor.execute(plan, mock_portal, "trace-123")
    assert call_order == ["objects", "objects"]
    assert len(results) == 2


@pytest.mark.asyncio
async def test_executor_no_coalescing_when_actions_are_different(mock_portal):
    mock_client = AsyncMock()
    mock_client.post.return_value = AsyncMock()
    mock_client.post.return_value.body = {
        "results": [{"id": "1"}],
        "errors": [],
    }

    call_order = []

    def fake_dispatch(agent_name, user_request, portal_config, mode, payload, trace_id):
        call_order.append(agent_name)
        return AgentResult(agent_name=agent_name, status="success")

    executor = PlanExecutor(dispatch_agent_fn=fake_dispatch, client=mock_client)
    plan = DAGPlan(
        plan_id="plan-test",
        nodes=[
            PlanNode(
                node_id="a",
                agent="objects",
                action="create",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="b",
                agent="objects",
                action="delete",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.DESTRUCTIVE,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.DESTRUCTIVE,
        estimated_duration_seconds=10,
    )

    results = await executor.execute(plan, mock_portal, "trace-123")
    assert call_order == []
    assert len(results) == 2
    assert mock_client.post.await_count == 2


@pytest.mark.asyncio
async def test_executor_no_coalescing_for_solitary_write(mock_portal):
    call_order = []

    def fake_dispatch(agent_name, user_request, portal_config, mode, payload, trace_id):
        call_order.append(agent_name)
        return AgentResult(agent_name=agent_name, status="success")

    executor = PlanExecutor(dispatch_agent_fn=fake_dispatch)
    plan = DAGPlan(
        plan_id="plan-test",
        nodes=[
            PlanNode(
                node_id="a",
                agent="objects",
                action="create",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="b",
                agent="lists",
                action="add",
                risk_level=RiskLevel.MEDIUM,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=10,
    )

    results = await executor.execute(plan, mock_portal, "trace-123")
    assert call_order == ["objects", "lists"]
    assert len(results) == 2


@pytest.mark.asyncio
async def test_executor_flushes_buffer_at_plan_boundary(mock_portal):
    mock_client = AsyncMock()
    mock_client.post.return_value = AsyncMock()
    mock_client.post.return_value.body = {
        "results": [{"id": "1"}, {"id": "2"}],
        "errors": [],
    }

    call_order = []

    def fake_dispatch(agent_name, user_request, portal_config, mode, payload, trace_id):
        call_order.append(agent_name)
        return AgentResult(agent_name=agent_name, status="success")

    executor = PlanExecutor(dispatch_agent_fn=fake_dispatch, client=mock_client)
    plan = DAGPlan(
        plan_id="plan-test",
        nodes=[
            PlanNode(
                node_id="a",
                agent="objects",
                action="create",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="b",
                agent="objects",
                action="create",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="c",
                agent="workflows",
                action="trigger",
                risk_level=RiskLevel.MEDIUM,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=15,
    )

    results = await executor.execute(plan, mock_portal, "trace-123")
    assert call_order == ["workflows"]
    assert len(results) == 3
    assert results[0].status == "success"
    assert results[1].status == "success"
    assert results[2].agent_name == "workflows"

    mock_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_executor_handles_buffer_errors_and_stops(mock_portal):
    mock_client = AsyncMock()
    mock_client.post.return_value = AsyncMock()
    mock_client.post.return_value.body = {
        "results": [{"id": "1"}, {}],
        "errors": [
            {"message": "Invalid property", "index": 1},
        ],
    }

    call_order = []

    def fake_dispatch(agent_name, user_request, portal_config, mode, payload, trace_id):
        call_order.append(agent_name)
        return AgentResult(agent_name=agent_name, status="success")

    executor = PlanExecutor(dispatch_agent_fn=fake_dispatch, client=mock_client)
    plan = DAGPlan(
        plan_id="plan-test",
        nodes=[
            PlanNode(
                node_id="a",
                agent="objects",
                action="create",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="b",
                agent="objects",
                action="create",
                inputs={"object_type": "contacts"},
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="c",
                agent="lists",
                action="add",
                risk_level=RiskLevel.MEDIUM,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=10,
    )

    results = await executor.execute(plan, mock_portal, "trace-123")
    assert len(results) == 2
    assert results[0].status == "success"
    assert results[0].data == {"id": "1"}
    assert results[1].status == "error"
    assert results[1].error_message == "Invalid property"
    assert call_order == []
