import pytest

from hubspot_agent.config import PortalConfig
from hubspot_agent.models import AgentResult, RiskLevel
from hubspot_agent.plan import DAGPlan, DAGPlanner, PlanExecutor, PlanNode
from hubspot_agent.preview import render_dag_plan


@pytest.fixture
def mock_portal():
    return PortalConfig(portal_id="123", token="test-token", tier="Professional")


@pytest.fixture
def planner():
    return DAGPlanner(
        fast_path_keywords={
            "objects": ["contact", "company", "deal", "ticket"],
            "properties": ["property", "field", "schema", "custom field"],
            "workflows": ["workflow", "automation", "enroll", "trigger"],
            "lists": ["list", "segment", "add to list"],
            "engagements": ["note", "task", "meeting", "call", "activity", "log"],
        },
        static_dependencies={
            "workflows": ["properties"],
            "lists": ["objects"],
            "engagements": ["objects"],
        },
        agent_getters={
            "objects": lambda: None,
            "properties": lambda: None,
            "workflows": lambda: None,
            "lists": lambda: None,
            "engagements": lambda: None,
        },
    )


# ---------------------------------------------------------------------------
# Model serialization
# ---------------------------------------------------------------------------


def test_plan_node_serialization():
    node = PlanNode(
        node_id="node-1",
        agent="objects",
        action="create",
        inputs={"object_type": "contacts"},
        outputs=["contact", "created_id"],
        dependencies=[],
        payload_summary={"text": "create a contact"},
        risk_level=RiskLevel.HIGH,
    )
    data = node.model_dump()
    assert data["node_id"] == "node-1"
    assert data["agent"] == "objects"
    assert data["action"] == "create"
    assert data["risk_level"] == "high"


def test_dag_plan_serialization():
    node = PlanNode(
        node_id="node-1",
        agent="objects",
        action="create",
        risk_level=RiskLevel.HIGH,
    )
    plan = DAGPlan(
        plan_id="plan-abc",
        nodes=[node],
        edges=[],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=5,
    )
    data = plan.model_dump()
    assert data["plan_id"] == "plan-abc"
    assert len(data["nodes"]) == 1
    assert data["overall_risk"] == "high"


# ---------------------------------------------------------------------------
# Compound request detection
# ---------------------------------------------------------------------------


def test_is_compound_request_sequential_phrases(planner):
    assert planner._is_compound_request("create a contact and then add them to a list")
    assert planner._is_compound_request("first create a property then build a workflow")
    assert planner._is_compound_request("create a contact next send an email")
    assert planner._is_compound_request("add a deal after that log a call")


def test_is_compound_request_numbered_steps(planner):
    assert planner._is_compound_request("1. create a contact\n2. add to list")
    assert planner._is_compound_request("1) create a deal 2) update the deal")


def test_is_compound_request_multiple_agents(planner):
    assert planner._is_compound_request("create a contact and enroll in workflow")
    assert planner._is_compound_request("find contacts and build a list segment")


def test_is_compound_request_single_step(planner):
    assert not planner._is_compound_request("find contacts in northeast")
    assert not planner._is_compound_request("create a contact")
    assert not planner._is_compound_request("log a call")


# ---------------------------------------------------------------------------
# Operation extraction
# ---------------------------------------------------------------------------


def test_extract_operations_splits_on_and_then(planner):
    ops = planner._extract_operations("create a contact and then add them to a list")
    assert len(ops) == 2
    assert ops[0]["agent"] == "objects"
    assert ops[1]["agent"] == "lists"


def test_extract_operations_splits_on_numbered_list(planner):
    ops = planner._extract_operations("1. create a contact\n2. add to list\n3. log a call")
    assert len(ops) == 3
    agents = [op["agent"] for op in ops]
    assert "objects" in agents
    assert "lists" in agents
    assert "engagements" in agents


def test_extract_operations_single_segment(planner):
    ops = planner._extract_operations("create a contact")
    assert len(ops) == 1
    assert ops[0]["agent"] == "objects"
    assert ops[0]["action"] == "create"


def test_extract_operations_infers_outputs(planner):
    ops = planner._extract_operations("create a contact and then add them to a list")
    create_op = next(op for op in ops if op["action"] == "create")
    assert "created_id" in create_op["outputs"]
    assert "contact" in create_op["outputs"]


def test_extract_operations_infers_inputs(planner):
    ops = planner._extract_operations("update contact id 123")
    assert ops[0]["inputs"]["needs_id"] is True
    assert ops[0]["inputs"]["object_type"] == "contacts"


# ---------------------------------------------------------------------------
# Edge derivation
# ---------------------------------------------------------------------------


def test_derive_edges_static_dependencies(planner):
    nodes = [
        PlanNode(node_id="node-1", agent="objects", action="create", risk_level=RiskLevel.HIGH),
        PlanNode(node_id="node-2", agent="lists", action="create", risk_level=RiskLevel.HIGH),
    ]
    edges = planner._derive_edges(nodes)
    assert ("node-1", "node-2") in edges


def test_derive_edges_data_flow(planner):
    nodes = [
        PlanNode(
            node_id="node-1",
            agent="objects",
            action="create",
            outputs=["created_id"],
            risk_level=RiskLevel.HIGH,
        ),
        PlanNode(
            node_id="node-2",
            agent="lists",
            action="add",
            inputs={"needs_id": True},
            risk_level=RiskLevel.HIGH,
        ),
    ]
    edges = planner._derive_edges(nodes)
    assert ("node-1", "node-2") in edges


def test_derive_edges_no_self_loops(planner):
    nodes = [
        PlanNode(node_id="node-1", agent="objects", action="create", risk_level=RiskLevel.HIGH),
    ]
    edges = planner._derive_edges(nodes)
    assert all(src != dst for src, dst in edges)


def test_derive_edges_preserves_existing_dependencies(planner):
    nodes = [
        PlanNode(
            node_id="node-1",
            agent="objects",
            action="create",
            risk_level=RiskLevel.HIGH,
        ),
        PlanNode(
            node_id="node-2",
            agent="lists",
            action="create",
            dependencies=["node-1"],
            risk_level=RiskLevel.HIGH,
        ),
    ]
    edges = planner._derive_edges(nodes)
    assert ("node-1", "node-2") in edges
    assert len(edges) == 1


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def test_topological_sort_linear_chain():
    nodes = [
        PlanNode(node_id="a", agent="objects", action="create", risk_level=RiskLevel.HIGH),
        PlanNode(node_id="b", agent="lists", action="create", risk_level=RiskLevel.HIGH),
        PlanNode(node_id="c", agent="workflows", action="trigger", risk_level=RiskLevel.MEDIUM),
    ]
    edges = [("a", "b"), ("b", "c")]
    executor = PlanExecutor()
    sorted_nodes = executor._topological_sort(nodes, edges)
    ids = [n.node_id for n in sorted_nodes]
    assert ids.index("a") < ids.index("b") < ids.index("c")


def test_topological_sort_parallel_nodes():
    nodes = [
        PlanNode(node_id="a", agent="objects", action="create", risk_level=RiskLevel.HIGH),
        PlanNode(node_id="b", agent="properties", action="create", risk_level=RiskLevel.HIGH),
        PlanNode(node_id="c", agent="workflows", action="trigger", risk_level=RiskLevel.MEDIUM),
    ]
    edges = [("a", "c"), ("b", "c")]
    executor = PlanExecutor()
    sorted_nodes = executor._topological_sort(nodes, edges)
    ids = [n.node_id for n in sorted_nodes]
    assert ids.index("c") > ids.index("a")
    assert ids.index("c") > ids.index("b")


def test_topological_sort_rejects_cycle():
    nodes = [
        PlanNode(node_id="a", agent="objects", action="create", risk_level=RiskLevel.HIGH),
        PlanNode(node_id="b", agent="lists", action="create", risk_level=RiskLevel.HIGH),
    ]
    edges = [("a", "b"), ("b", "a")]
    executor = PlanExecutor()
    with pytest.raises(ValueError, match="Cyclic dependency"):
        executor._topological_sort(nodes, edges)


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def test_resolve_inputs_wires_created_id():
    node = PlanNode(
        node_id="b",
        agent="lists",
        action="add",
        inputs={"needs_id": True},
        dependencies=["a"],
        risk_level=RiskLevel.HIGH,
    )
    completed = {
        "a": AgentResult(
            agent_name="objects",
            status="success",
            data={"created_id": "123"},
        ),
    }
    executor = PlanExecutor()
    resolved = executor._resolve_inputs(node, completed)
    assert resolved["source_id"] == "123"


def test_resolve_inputs_keeps_existing():
    node = PlanNode(
        node_id="b",
        agent="lists",
        action="add",
        inputs={"object_type": "contacts"},
        dependencies=[],
        risk_level=RiskLevel.HIGH,
    )
    executor = PlanExecutor()
    resolved = executor._resolve_inputs(node, {})
    assert resolved["object_type"] == "contacts"


# ---------------------------------------------------------------------------
# Render DAG plan
# ---------------------------------------------------------------------------


def test_render_dag_plan_markdown_table():
    nodes = [
        PlanNode(
            node_id="node-1",
            agent="objects",
            action="create",
            payload_summary={"text": "create a contact"},
            risk_level=RiskLevel.HIGH,
        ),
        PlanNode(
            node_id="node-2",
            agent="lists",
            action="add",
            payload_summary={"text": "add contact to list"},
            risk_level=RiskLevel.MEDIUM,
            dependencies=["node-1"],
        ),
    ]
    plan = DAGPlan(
        plan_id="plan-abc",
        nodes=nodes,
        edges=[("node-1", "node-2")],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=13,
    )
    markdown = render_dag_plan(plan)
    assert "## Execution Plan: plan-abc" in markdown
    assert "**Overall Risk:** HIGH" in markdown
    assert "**Estimated Duration:** 13s" in markdown
    assert "| Step | Agent | Action | Risk | Dependencies | Summary |" in markdown
    assert "| 1 | objects | create | HIGH | None | create a contact |" in markdown
    assert "| 2 | lists | add | MEDIUM | node-1 | add contact to list |" in markdown
    assert "Approve this full plan" in markdown


def test_render_dag_plan_long_summary_truncated():
    nodes = [
        PlanNode(
            node_id="node-1",
            agent="objects",
            action="create",
            payload_summary={"text": "a" * 100},
            risk_level=RiskLevel.LOW,
        ),
    ]
    plan = DAGPlan(
        plan_id="plan-x",
        nodes=nodes,
        edges=[],
        overall_risk=RiskLevel.LOW,
        estimated_duration_seconds=5,
    )
    markdown = render_dag_plan(plan)
    assert "a..." in markdown


# ---------------------------------------------------------------------------
# DAGPlanner.generate integration
# ---------------------------------------------------------------------------


def test_dag_planner_generate_creates_plan(planner, mock_portal):
    plan = planner.generate(
        "create a contact and then add them to a list", mock_portal
    )
    assert isinstance(plan, DAGPlan)
    assert plan.plan_id.startswith("plan-")
    assert len(plan.nodes) == 2
    assert len(plan.edges) >= 1
    assert plan.overall_risk == RiskLevel.HIGH
    assert plan.estimated_duration_seconds > 0


def test_dag_planner_single_step_still_returns_plan(planner, mock_portal):
    plan = planner.generate("create a contact", mock_portal)
    assert len(plan.nodes) == 1
    assert plan.nodes[0].agent == "objects"


# ---------------------------------------------------------------------------
# Orchestrator integration helpers
# ---------------------------------------------------------------------------


def test_is_compound_request_from_orchestrator():
    from hubspot_agent.orchestrator import is_compound_request

    assert is_compound_request("create a contact and then add them to a list")
    assert not is_compound_request("find contacts in northeast")


@pytest.mark.asyncio
async def test_route_and_plan_bypasses_for_simple_request(mock_portal):
    from hubspot_agent.orchestrator import route_and_plan

    result = await route_and_plan("find contacts in northeast", mock_portal)
    assert isinstance(result, list)
    assert "objects" in result


@pytest.mark.asyncio
async def test_route_and_plan_returns_dag_for_compound_request(mock_portal):
    from hubspot_agent.orchestrator import route_and_plan

    result = await route_and_plan(
        "create a contact and then add them to a list", mock_portal
    )
    assert isinstance(result, DAGPlan)
    assert len(result.nodes) == 2


# ---------------------------------------------------------------------------
# PlanExecutor execute integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_executor_execute_runs_in_order(mock_portal):
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
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="b",
                agent="lists",
                action="add",
                dependencies=["a"],
                risk_level=RiskLevel.HIGH,
            ),
        ],
        edges=[("a", "b")],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=10,
    )

    results = await executor.execute(plan, mock_portal, "trace-123")
    assert call_order == ["objects", "lists"]
    assert len(results) == 2
    assert all(r.status == "success" for r in results)


@pytest.mark.asyncio
async def test_plan_executor_no_dispatch_fn_raises(mock_portal):
    executor = PlanExecutor()
    plan = DAGPlan(
        plan_id="plan-test",
        nodes=[
            PlanNode(
                node_id="a",
                agent="objects",
                action="create",
                risk_level=RiskLevel.HIGH,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.LOW,
        estimated_duration_seconds=5,
    )
    with pytest.raises(RuntimeError, match="No dispatch agent function"):
        await executor.execute(plan, mock_portal, "trace-123")


# ---------------------------------------------------------------------------
# Overall risk computation
# ---------------------------------------------------------------------------


def test_compute_overall_risk_destructive_wins(planner):
    nodes = [
        PlanNode(node_id="a", agent="objects", action="read", risk_level=RiskLevel.LOW),
        PlanNode(node_id="b", agent="objects", action="delete", risk_level=RiskLevel.DESTRUCTIVE),
    ]
    assert planner._compute_overall_risk(nodes) == RiskLevel.DESTRUCTIVE


def test_compute_overall_risk_high_wins_over_medium(planner):
    nodes = [
        PlanNode(node_id="a", agent="objects", action="read", risk_level=RiskLevel.MEDIUM),
        PlanNode(node_id="b", agent="objects", action="create", risk_level=RiskLevel.HIGH),
    ]
    assert planner._compute_overall_risk(nodes) == RiskLevel.HIGH


def test_compute_overall_risk_low_only(planner):
    nodes = [
        PlanNode(node_id="a", agent="objects", action="read", risk_level=RiskLevel.LOW),
    ]
    assert planner._compute_overall_risk(nodes) == RiskLevel.LOW


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------


def test_estimate_duration_basic(planner):
    nodes = [
        PlanNode(node_id="a", agent="objects", action="create", risk_level=RiskLevel.HIGH),
        PlanNode(node_id="b", agent="lists", action="create", risk_level=RiskLevel.HIGH),
    ]
    edges = [("a", "b")]
    duration = planner._estimate_duration(nodes, edges)
    assert duration == 10 + 3  # 2*5 + 1*3


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------


def test_assess_risk_delete(planner):
    assert planner._assess_risk({"action": "delete"}) == RiskLevel.DESTRUCTIVE


def test_assess_risk_create(planner):
    assert planner._assess_risk({"action": "create"}) == RiskLevel.HIGH


def test_assess_risk_update(planner):
    assert planner._assess_risk({"action": "update"}) == RiskLevel.HIGH


def test_assess_risk_trigger(planner):
    assert planner._assess_risk({"action": "trigger"}) == RiskLevel.MEDIUM


def test_assess_risk_read(planner):
    assert planner._assess_risk({"action": "read"}) == RiskLevel.LOW
