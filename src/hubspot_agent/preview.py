from __future__ import annotations

from typing import Any

from hubspot_agent.plan import DAGPlan, PlanModification


def render_field_diff(
    old_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    max_records: int = 10,
) -> str:
    """Render a markdown table of per-field diffs for the first N records."""
    lines: list[str] = []
    count = min(len(old_records), len(new_records), max_records)
    if count == 0:
        return ""

    for i in range(count):
        old = old_records[i]
        new = new_records[i]
        record_id = old.get("id", new.get("id", f"record_{i}"))
        lines.append(f"**Record {record_id}**")
        diff_lines: list[str] = []
        for key in sorted(set(old.keys()) | set(new.keys())):
            old_val = old.get(key)
            new_val = new.get(key)
            if old_val != new_val:
                diff_lines.append(f"- `{key}`: `{old_val}` -> `{new_val}`")
        if diff_lines:
            lines.extend(diff_lines)
        else:
            lines.append("- (no changes)")
        lines.append("")

    return "\n".join(lines)


def render_pattern_summary(
    old_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    start_index: int = 10,
) -> str:
    """Summarize remaining records by identical change pattern."""
    remaining = min(len(old_records), len(new_records)) - start_index
    if remaining <= 0:
        return ""

    # Identify fields that change in every remaining record with the same old->new mapping
    if start_index >= len(old_records) or start_index >= len(new_records):
        return f"*Remaining {remaining} records: same change pattern.*"

    first_old = old_records[start_index]
    first_new = new_records[start_index]
    changed_fields: dict[str, tuple[Any, Any]] = {}

    for key in sorted(set(first_old.keys()) | set(first_new.keys())):
        old_val = first_old.get(key)
        new_val = first_new.get(key)
        if old_val != new_val:
            changed_fields[key] = (old_val, new_val)

    # Verify pattern holds across all remaining records
    consistent = True
    for i in range(start_index + 1, min(len(old_records), len(new_records))):
        old = old_records[i]
        new = new_records[i]
        for key, (expected_old, expected_new) in changed_fields.items():
            if old.get(key) != expected_old or new.get(key) != expected_new:
                consistent = False
                break
        if not consistent:
            break

    if consistent and changed_fields:
        parts = [f"`{k}`: `{v[0]}` -> `{v[1]}`" for k, v in sorted(changed_fields.items())]
        return f"*Remaining {remaining} records: all have {'; '.join(parts)}.*"

    return f"*Remaining {remaining} records: same change pattern.*"


def format_preview(
    old_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    impact_count: int,
    mode: str = "diff",
) -> str:
    """Format a preview with inline diffs or a summary."""
    if mode != "diff":
        return f"**Impact:** {impact_count} records"

    diff = render_field_diff(old_records, new_records, max_records=10)
    summary = render_pattern_summary(old_records, new_records, start_index=10)
    lines = [f"**Impact:** {impact_count} records", ""]
    if diff:
        lines.append(diff)
    if summary:
        lines.append(summary)
    return "\n".join(lines)


def render_dag_plan(
    plan: DAGPlan,
    modification: PlanModification | None = None,
) -> str:
    """Render a DAGPlan as a markdown table for HITL approval.

    Steps are shown in topological (execution) order so the user sees
    the actual sequence the executor will follow.
    """
    from collections import deque

    title = "Modified Execution Plan" if modification else "Execution Plan"
    lines: list[str] = [
        f"## {title}: {plan.plan_id}",
        f"- **Overall Risk:** {plan.overall_risk.value.upper()}",
        f"- **Estimated Duration:** {plan.estimated_duration_seconds}s",
        f"- **Nodes:** {len(plan.nodes)} | **Edges:** {len(plan.edges)}",
    ]

    if modification and modification.skip_nodes:
        lines.append(f"- **Skipped Nodes:** {', '.join(modification.skip_nodes)}")

    lines.extend([
        "",
        "| Step | Agent | Action | Risk | Dependencies | Summary |",
        "|------|-------|--------|------|--------------|---------|",
    ])

    # Build dependency map for display
    dep_map: dict[str, list[str]] = {n.node_id: [] for n in plan.nodes}
    for src, dst in plan.edges:
        dep_map.setdefault(dst, []).append(src)

    # Topological sort for rendering order
    in_degree: dict[str, int] = {n.node_id: 0 for n in plan.nodes}
    for src, dst in plan.edges:
        if src in in_degree and dst in in_degree:
            in_degree[dst] += 1

    node_map = {n.node_id: n for n in plan.nodes}
    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    sorted_ids: list[str] = []
    adjacency: dict[str, list[str]] = {n.node_id: [] for n in plan.nodes}
    for src, dst in plan.edges:
        if src in adjacency:
            adjacency[src].append(dst)

    while queue:
        current = queue.popleft()
        sorted_ids.append(current)
        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Fallback: if cycle or incomplete sort, use original order
    if len(sorted_ids) != len(plan.nodes):
        sorted_ids = [n.node_id for n in plan.nodes]

    for i, node_id in enumerate(sorted_ids, start=1):
        node = node_map[node_id]
        deps = ", ".join(dep_map.get(node.node_id, [])) or "None"
        summary = node.payload_summary.get("text", "")[:60]
        if len(summary) == 60:
            summary += "..."

        if modification and node.node_id in modification.parameter_edits:
            edits = modification.parameter_edits[node.node_id]
            edit_str = "; ".join(f"{k}={v}" for k, v in edits.items())
            summary += f" **(EDITED: {edit_str})**"

        lines.append(
            f"| {i} | {node.agent} | {node.action} | {node.risk_level.value.upper()} | {deps} | {summary} |"
        )

    lines.append("")
    lines.append("Approve this full plan to execute all steps in dependency order? (y/n)")
    return "\n".join(lines)
