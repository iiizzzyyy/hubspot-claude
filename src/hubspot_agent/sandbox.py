from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any

from pydantic import BaseModel

from hubspot_agent.config import PortalConfig, load_portal_config
from hubspot_agent.models import AgentResult, RiskLevel
from hubspot_agent.plan import DAGPlan, PlanExecutor, PlanNode
from hubspot_agent.trace import emit_trace, new_trace_id


class BehaviorDiff(BaseModel):
    matches: dict[str, Any] = {}
    mismatches: list[dict] = []
    missing: list[str] = []
    extra: list[str] = []


class SandboxResult(BaseModel):
    plan_executed: bool
    behavior_diff: BehaviorDiff
    warnings: list[str]
    sandbox_portal_id: str


_IGNORED_KEYS = {
    "system_prompt",
    "full_prompt",
    "tool_names",
    "batch_mode",
    "action_id",
    "resolved_inputs",
    "_sandbox_node_id",
}


class _SandboxPlanExecutor(PlanExecutor):
    """PlanExecutor variant that disables write buffering for sandbox runs."""

    def _should_start_buffer(
        self,
        node: PlanNode,
        ordered_nodes: list[PlanNode],
        idx: int,
    ) -> bool:
        return False

    def _build_node_payload(
        self,
        node: PlanNode,
        completed_results: dict[str, AgentResult],
    ) -> dict[str, Any]:
        payload = super()._build_node_payload(node, completed_results)
        payload["_sandbox_node_id"] = node.node_id
        return payload


class SandboxRunner:
    def __init__(
        self,
        dispatch_agent_fn: Any | None = None,
    ) -> None:
        self._dispatch_agent = dispatch_agent_fn

    async def preview_in_sandbox(
        self,
        plan: DAGPlan,
        sandbox_portal: PortalConfig,
    ) -> SandboxResult:
        trace_id = new_trace_id()
        emit_trace(
            sandbox_portal.portal_id,
            "tool_call",
            trace_id,
            {
                "plan_id": plan.plan_id,
                "event": "sandbox_execution_start",
                "node_count": len(plan.nodes),
            },
        )

        if self._dispatch_agent is None:
            raise RuntimeError(
                "No dispatch agent function configured for SandboxRunner"
            )

        execution_log: list[tuple[str, AgentResult]] = []

        async def tracked_dispatch(**kwargs: Any) -> AgentResult:
            payload = kwargs.get("payload", {})
            node_id = payload.get("_sandbox_node_id")
            result = self._dispatch_agent(**kwargs)
            if inspect.iscoroutine(result):
                result = await result
            if node_id:
                execution_log.append((str(node_id), result))
            return result

        executor = _SandboxPlanExecutor(dispatch_agent_fn=tracked_dispatch)
        all_results = await executor.execute(plan, sandbox_portal, trace_id)

        node_map = {n.node_id: n for n in plan.nodes}

        if not execution_log:
            ordered_nodes = executor._topological_sort(plan.nodes, plan.edges)
            for node, result in zip(ordered_nodes, all_results):
                execution_log.append((node.node_id, result))

        warnings: list[str] = []
        for node_id, result in execution_log:
            if result.status in ("error", "duplicate", "blocked"):
                warnings.append(
                    f"Node {node_id} ({result.agent_name}) failed: "
                    f"{result.error_message or result.status}"
                )

        behavior_diff = self._compute_behavior_diff(execution_log, node_map)

        emit_trace(
            sandbox_portal.portal_id,
            "completion",
            trace_id,
            {
                "plan_id": plan.plan_id,
                "event": "sandbox_execution_complete",
                "warnings_count": len(warnings),
                "diff_matches": len(behavior_diff.matches),
                "diff_mismatches": len(behavior_diff.mismatches),
            },
        )

        plan_executed = (
            all(
                result.status in ("success", "preview", "ready", "corrected")
                for _, result in execution_log
            )
            and len(execution_log) == len(plan.nodes)
        )

        return SandboxResult(
            plan_executed=plan_executed,
            behavior_diff=behavior_diff,
            warnings=warnings,
            sandbox_portal_id=sandbox_portal.portal_id,
        )

    def _compute_behavior_diff(
        self,
        execution_log: list[tuple[str, AgentResult]],
        node_map: dict[str, PlanNode],
    ) -> BehaviorDiff:
        matches: dict[str, Any] = {}
        mismatches: list[dict] = []
        missing: list[str] = []
        extra: list[str] = []

        for node_id, result in execution_log:
            node = node_map.get(node_id)
            if node is None:
                continue

            expected = dict(node.payload_summary)
            actual = dict(result.data) if result.data else {}
            actual_clean = {
                k: v for k, v in actual.items() if k not in _IGNORED_KEYS
            }

            for key, expected_val in expected.items():
                full_key = f"{node_id}.{key}"
                if key not in actual_clean:
                    missing.append(full_key)
                elif actual_clean[key] == expected_val:
                    matches[full_key] = actual_clean[key]
                else:
                    mismatches.append({
                        "node_id": node_id,
                        "field": key,
                        "expected": expected_val,
                        "actual": actual_clean[key],
                    })

            for key in actual_clean:
                if key not in expected:
                    extra.append(f"{node_id}.{key}")

        return BehaviorDiff(
            matches=matches,
            mismatches=mismatches,
            missing=missing,
            extra=extra,
        )


def get_sandbox_portal_config() -> PortalConfig | None:
    """Load the configured sandbox portal from the environment."""
    portal_id = os.environ.get("HUBSPOT_SANDBOX_PORTAL_ID")
    if not portal_id:
        return None
    return load_portal_config(portal_id)


def should_offer_sandbox(plan: DAGPlan) -> bool:
    """Return True if the plan is high-risk and a sandbox portal is configured."""
    if plan.overall_risk not in (RiskLevel.HIGH, RiskLevel.DESTRUCTIVE):
        return False
    return get_sandbox_portal_config() is not None


def build_sandbox_offer_prompt(plan: DAGPlan) -> str:
    """Build a prompt offering sandbox preview before production execution."""
    sandbox_config = get_sandbox_portal_config()
    if sandbox_config is None:
        return ""
    lines = [
        "### Sandbox Preview Available",
        f"This plan has **{plan.overall_risk.value.upper()}** risk. "
        f"A sandbox portal is configured (`{sandbox_config.portal_id}`).",
        "",
        "Run this plan in the sandbox first to preview the behavior "
        "before applying to production?",
        "",
        "Approve sandbox preview? (y/n)",
    ]
    return "\n".join(lines)


def format_sandbox_result(result: SandboxResult) -> str:
    """Format a sandbox result for human review."""
    lines = [
        f"### Sandbox Preview Result (Portal {result.sandbox_portal_id})",
        f"- **Plan executed:** {'Yes' if result.plan_executed else 'No'}",
    ]
    if result.warnings:
        lines.append("- **Warnings:**")
        for w in result.warnings:
            lines.append(f"  - {w}")

    diff = result.behavior_diff
    lines.append(f"- **Matches:** {len(diff.matches)}")
    lines.append(f"- **Mismatches:** {len(diff.mismatches)}")
    if diff.mismatches:
        lines.append("  - Mismatched fields:")
        for m in diff.mismatches:
            lines.append(
                f"    - {m['node_id']}.{m['field']}: "
                f"expected {m['expected']!r}, got {m['actual']!r}"
            )
    if diff.missing:
        lines.append(f"- **Missing:** {', '.join(diff.missing)}")
    if diff.extra:
        lines.append(f"- **Extra:** {', '.join(diff.extra)}")

    return "\n".join(lines)
