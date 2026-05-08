import pytest

from hubspot_agent.models import RiskLevel
from hubspot_agent.plan import DAGPlan, InteractivePlanModifier, PlanModification, PlanNode
from hubspot_agent.preview import render_dag_plan


@pytest.fixture
def modifier():
    return InteractivePlanModifier()


@pytest.fixture
def sample_plan():
    return DAGPlan(
        plan_id="plan-test",
        nodes=[
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
                dependencies=["node-1"],
                risk_level=RiskLevel.MEDIUM,
            ),
            PlanNode(
                node_id="node-3",
                agent="workflows",
                action="trigger",
                payload_summary={"text": "enroll in workflow"},
                dependencies=["node-2"],
                risk_level=RiskLevel.MEDIUM,
            ),
        ],
        edges=[("node-1", "node-2"), ("node-2", "node-3")],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=21,
    )


# ---------------------------------------------------------------------------
# parse_instruction
# ---------------------------------------------------------------------------


def test_parse_instruction_skip_node_id(modifier):
    mod = modifier.parse_instruction("skip node-2")
    assert mod.skip_nodes == ["node-2"]
    assert mod.parameter_edits == {}


def test_parse_instruction_skip_short_id(modifier):
    mod = modifier.parse_instruction("skip n2")
    assert mod.skip_nodes == ["node-2"]
    assert mod.parameter_edits == {}


def test_parse_instruction_skip_multiple(modifier):
    mod = modifier.parse_instruction("skip node-1 and skip n3")
    assert mod.skip_nodes == ["node-1", "node-3"]


def test_parse_instruction_skip_with_punctuation(modifier):
    mod = modifier.parse_instruction("skip node-1, skip node-2;")
    assert mod.skip_nodes == ["node-1", "node-2"]


def test_parse_instruction_edit_node_id(modifier):
    mod = modifier.parse_instruction("edit node-2 property=new_value")
    assert mod.skip_nodes == []
    assert mod.parameter_edits == {"node-2": {"property": "new_value"}}


def test_parse_instruction_edit_short_id(modifier):
    mod = modifier.parse_instruction("edit n2 property=new_value")
    assert mod.parameter_edits == {"node-2": {"property": "new_value"}}


def test_parse_instruction_mixed(modifier):
    mod = modifier.parse_instruction("skip node-1, edit n2 property=X")
    assert mod.skip_nodes == ["node-1"]
    assert mod.parameter_edits == {"node-2": {"property": "X"}}


def test_parse_instruction_case_insensitive(modifier):
    mod = modifier.parse_instruction("SKIP N2, EDIT NODE-1 FOO=bar")
    assert mod.skip_nodes == ["node-2"]
    assert mod.parameter_edits == {"node-1": {"foo": "bar"}}


def test_parse_instruction_empty(modifier):
    mod = modifier.parse_instruction("just approve it")
    assert mod.skip_nodes == []
    assert mod.parameter_edits == {}


# ---------------------------------------------------------------------------
# apply — skip and edit
# ---------------------------------------------------------------------------


def test_apply_skip_leaf_node(modifier, sample_plan):
    modification = PlanModification(skip_nodes=["node-3"])
    new_plan = modifier.apply(sample_plan, modification)
    assert len(new_plan.nodes) == 2
    assert all(n.node_id != "node-3" for n in new_plan.nodes)
    assert ("node-2", "node-3") not in new_plan.edges


def test_apply_edit_node(modifier, sample_plan):
    modification = PlanModification(parameter_edits={"node-2": {"property": "X"}})
    new_plan = modifier.apply(sample_plan, modification)
    node_2 = next(n for n in new_plan.nodes if n.node_id == "node-2")
    assert node_2.payload_summary["property"] == "X"
    assert node_2.payload_summary["text"] == "add contact to list"


def test_apply_skip_and_edit(modifier, sample_plan):
    modification = PlanModification(
        skip_nodes=["node-3"],
        parameter_edits={"node-2": {"foo": "bar"}},
    )
    new_plan = modifier.apply(sample_plan, modification)
    assert len(new_plan.nodes) == 2
    node_2 = next(n for n in new_plan.nodes if n.node_id == "node-2")
    assert node_2.payload_summary["foo"] == "bar"


def test_apply_recomputes_risk(modifier, sample_plan):
    modification = PlanModification(skip_nodes=["node-3"])
    new_plan = modifier.apply(sample_plan, modification)
    assert new_plan.overall_risk == RiskLevel.HIGH


def test_apply_recomputes_duration(modifier, sample_plan):
    modification = PlanModification(skip_nodes=["node-3"])
    new_plan = modifier.apply(sample_plan, modification)
    # 2 nodes * 5 + 1 edge * 3 = 13
    assert new_plan.estimated_duration_seconds == 13


def test_apply_preserves_plan_id(modifier, sample_plan):
    modification = PlanModification(skip_nodes=["node-3"])
    new_plan = modifier.apply(sample_plan, modification)
    assert new_plan.plan_id == sample_plan.plan_id


# ---------------------------------------------------------------------------
# validate_invariants
# ---------------------------------------------------------------------------


def test_validate_invariants_passes(modifier, sample_plan):
    violations = modifier.validate_invariants(sample_plan)
    assert violations == []


def test_validate_invariants_empty_plan(modifier):
    plan = DAGPlan(
        plan_id="plan-empty",
        nodes=[],
        edges=[],
        overall_risk=RiskLevel.LOW,
        estimated_duration_seconds=0,
    )
    violations = modifier.validate_invariants(plan)
    assert any("no nodes" in v.lower() for v in violations)


def test_validate_invariants_detects_orphaned_dependency(modifier):
    plan = DAGPlan(
        plan_id="plan-bad",
        nodes=[
            PlanNode(
                node_id="node-1",
                agent="objects",
                action="create",
                dependencies=["missing-node"],
                risk_level=RiskLevel.HIGH,
            ),
        ],
        edges=[],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=5,
    )
    violations = modifier.validate_invariants(plan)
    assert any("missing node" in v for v in violations)


def test_validate_invariants_detects_cycle(modifier):
    plan = DAGPlan(
        plan_id="plan-bad",
        nodes=[
            PlanNode(
                node_id="node-1",
                agent="objects",
                action="create",
                risk_level=RiskLevel.HIGH,
            ),
            PlanNode(
                node_id="node-2",
                agent="lists",
                action="add",
                risk_level=RiskLevel.HIGH,
            ),
        ],
        edges=[("node-1", "node-2"), ("node-2", "node-1")],
        overall_risk=RiskLevel.HIGH,
        estimated_duration_seconds=13,
    )
    violations = modifier.validate_invariants(plan)
    assert any("Cyclic" in v for v in violations)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_apply_invalid_skip_node_raises(modifier, sample_plan):
    modification = PlanModification(skip_nodes=["node-99"])
    with pytest.raises(ValueError, match="unknown node"):
        modifier.apply(sample_plan, modification)


def test_apply_invalid_edit_node_raises(modifier, sample_plan):
    modification = PlanModification(parameter_edits={"node-99": {"foo": "bar"}})
    with pytest.raises(ValueError, match="unknown node"):
        modifier.apply(sample_plan, modification)


def test_apply_skip_all_nodes_raises(modifier, sample_plan):
    modification = PlanModification(skip_nodes=["node-1", "node-2", "node-3"])
    with pytest.raises(ValueError, match="empty"):
        modifier.apply(sample_plan, modification)


def test_apply_orphaned_downstream_raises(modifier, sample_plan):
    modification = PlanModification(skip_nodes=["node-1"])
    with pytest.raises(ValueError, match="missing node"):
        modifier.apply(sample_plan, modification)


# ---------------------------------------------------------------------------
# preview rendering with modifications
# ---------------------------------------------------------------------------


def test_render_dag_plan_shows_modifications(modifier, sample_plan):
    modification = PlanModification(
        skip_nodes=["node-3"],
        parameter_edits={"node-2": {"foo": "bar"}},
    )
    new_plan = modifier.apply(sample_plan, modification)
    markdown = render_dag_plan(new_plan, modification=modification)
    assert "Modified Execution Plan" in markdown
    assert "**Skipped Nodes:** node-3" in markdown
    assert "EDITED: foo=bar" in markdown


def test_render_dag_plan_without_modification(modifier, sample_plan):
    markdown = render_dag_plan(sample_plan)
    assert "Execution Plan:" in markdown
    assert "Skipped Nodes" not in markdown
    assert "EDITED" not in markdown
