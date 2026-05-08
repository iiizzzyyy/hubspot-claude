from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from hubspot_agent.agents.analytics import get_analytics_agent_prompt
from hubspot_agent.agents.associations import get_associations_agent_prompt
from hubspot_agent.agents.engagements import get_engagements_agent_prompt
from hubspot_agent.agents.hygiene import get_hygiene_agent_prompt
from hubspot_agent.agents.lists import get_lists_agent_prompt
from hubspot_agent.agents.objects import get_objects_agent_prompt
from hubspot_agent.agents.pipelines import get_pipelines_agent_prompt
from hubspot_agent.agents.properties import get_properties_agent_prompt
from hubspot_agent.agents.raw_api import get_raw_api_agent_prompt
from hubspot_agent.agents.users import get_users_agent_prompt
from hubspot_agent.agents.workflows import get_workflows_agent_prompt
from hubspot_agent.capabilities import (
    CapabilityMatrix,
    capability_explanation,
    probe_portal,
    validate_capabilities,
)
from hubspot_agent.config import PortalConfig
from hubspot_agent.models import AgentResult, PreviewResult, RiskLevel
from hubspot_agent.research import classify_url
from hubspot_agent.cache import warm_standard_schemas
from hubspot_agent.config import load_portal_config
from hubspot_agent.maintenance import run_maintenance
from hubspot_agent.snapshot import save_undo_snapshot
from hubspot_agent.ledger import ActionLedger


async def initialize_session(portal_id: str) -> None:
    try:
        await asyncio.wait_for(run_maintenance(portal_id), timeout=10.0)
    except asyncio.TimeoutError:
        pass
    portal_config = load_portal_config(portal_id)
    if portal_config is not None:
        try:
            await asyncio.wait_for(warm_standard_schemas(portal_config), timeout=15.0)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

_AGENT_KEYWORDS: dict[str, list[str]] = {
    "objects": ["contact", "company", "deal", "ticket", "record"],
    "properties": ["property", "field", "schema", "custom field"],
    "workflows": ["workflow", "automation", "enroll", "trigger"],
    "lists": ["list", "segment", "add to list"],
    "pipelines": ["pipeline", "stage", "move to"],
    "users": ["user", "permission", "team", "owner", "onboard"],
    "hygiene": ["duplicate", "merge", "dedup", "clean"],
    "analytics": ["report", "metric", "analytics", "how many"],
    "associations": ["associate", "link", "relationship", "related to"],
    "engagements": ["note", "task", "email", "meeting", "call", "activity", "log"],
    "raw_api": ["raw api", "custom endpoint", "direct api", "not covered", "escape hatch"],
}

_STATIC_DEPENDENCIES: dict[str, list[str]] = {
    "workflows": ["properties"],
    "lists": ["objects"],
    "engagements": ["objects"],
}

_AGENT_GETTERS: dict[str, Any] = {
    "objects": get_objects_agent_prompt,
    "properties": get_properties_agent_prompt,
    "workflows": get_workflows_agent_prompt,
    "lists": get_lists_agent_prompt,
    "pipelines": get_pipelines_agent_prompt,
    "users": get_users_agent_prompt,
    "hygiene": get_hygiene_agent_prompt,
    "analytics": get_analytics_agent_prompt,
    "associations": get_associations_agent_prompt,
    "engagements": get_engagements_agent_prompt,
    "raw_api": get_raw_api_agent_prompt,
}


def route_request(request_text: str) -> list[str]:
    text = request_text.lower()
    scored: dict[str, int] = {}

    for agent, keywords in _AGENT_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        if score > 0:
            scored[agent] = score

    if not scored:
        return []

    primary = sorted(scored, key=lambda a: scored[a], reverse=True)

    # Dependency ordering
    ordered: list[str] = []
    for agent in primary:
        deps = _STATIC_DEPENDENCIES.get(agent, [])
        for dep in deps:
            if dep in scored and dep not in ordered:
                ordered.append(dep)
        if agent not in ordered:
            ordered.append(agent)

    return ordered


# ---------------------------------------------------------------------------
# Scope validation
# ---------------------------------------------------------------------------


def validate_scopes(
    agent_names: list[str], portal_scopes: list[str]
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    portal_scope_set = set(portal_scopes)

    for name in agent_names:
        getter = _AGENT_GETTERS.get(name)
        if getter is None:
            continue
        prompt = getter()
        required: set[str] = set()
        for tname in prompt.tool_names:
            from hubspot_agent.tools import get_tool
            tool_def = get_tool(tname)
            if tool_def and hasattr(tool_def.func, "__defaults__"):
                # expected_scopes is typically the last kwarg default
                import inspect
                sig = inspect.signature(tool_def.func)
                for param in sig.parameters.values():
                    if param.name == "expected_scopes" and param.default is not inspect.Parameter.empty:
                        if isinstance(param.default, list):
                            required.update(param.default)
            # fallback: look at closure cell defaults
            if tool_def and hasattr(tool_def.func, "__wrapped__"):
                import inspect
                sig = inspect.signature(tool_def.func)
                for param in sig.parameters.values():
                    if param.name == "expected_scopes" and param.default is not inspect.Parameter.empty:
                        if isinstance(param.default, list):
                            required.update(param.default)

        missing_for_agent = sorted(required - portal_scope_set)
        if missing_for_agent:
            missing[name] = missing_for_agent

    return missing


# ---------------------------------------------------------------------------
# Capability validation
# ---------------------------------------------------------------------------


async def check_dispatch_readiness(
    agent_names: list[str],
    portal_config: PortalConfig,
) -> dict[str, Any]:
    """Validate scopes and capabilities before dispatching agents.

    Returns a dict with:
      - 'missing_scopes': dict[str, list[str]] — per-agent missing scopes
      - 'missing_capabilities': dict[str, list[str]] — per-agent missing features
      - 'ready': bool — True if nothing blocks dispatch
      - 'decline_reason': str | None — human-readable explanation if not ready
    """
    scope_result = validate_scopes(agent_names, portal_config.scopes_granted or [])
    matrix = await probe_portal(portal_config)
    capability_result = validate_capabilities(agent_names, matrix)

    ready = not scope_result and not capability_result
    decline_reason: str | None = None
    if not ready:
        parts: list[str] = []
        if capability_result:
            for agent, features in capability_result.items():
                for feature in features:
                    parts.append(capability_explanation(feature))
        if scope_result:
            for agent, scopes in scope_result.items():
                parts.append(f"{agent} requires scopes: {', '.join(scopes)}")
        decline_reason = "Cannot dispatch: " + "; ".join(parts)

    return {
        "missing_scopes": scope_result,
        "missing_capabilities": capability_result,
        "ready": ready,
        "decline_reason": decline_reason,
    }


# ---------------------------------------------------------------------------
# HITL approval
# ---------------------------------------------------------------------------


def needs_approval(risk_level: RiskLevel) -> bool:
    return risk_level != RiskLevel.LOW


def normalize_informing_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize informing_sources using URL classification.

    Overrides any entry whose self-reported (source, trust_tier) disagrees
    with the URL-derived classification, except it never upgrades a
    self-reported 'community-accepted' to 'official' based on URL alone.
    """
    normalized: list[dict[str, Any]] = []
    for entry in sources:
        url = entry.get("url", "")
        url_source, url_tier = classify_url(url)
        reported_source = entry.get("source", "")
        reported_tier = entry.get("trust_tier", "")
        # If URL says official, force source/tier to official
        if url_source == "official":
            entry["source"] = "official"
            entry["trust_tier"] = "official"
        else:
            # URL says community.  Preserve richer sub-agent context unless
            # the agent claimed something impossible (e.g. official on a
            # non-official domain).
            if reported_source == "official":
                entry["source"] = url_source
                # Downgrade tier but keep accepted-answer context if present
                if reported_tier == "community-accepted":
                    entry["trust_tier"] = "community-accepted"
                else:
                    entry["trust_tier"] = url_tier
            # If agent said community and URL says community, trust the agent
            # unless it claimed a tier we can't justify from the URL.
            if reported_source == "community" and reported_tier == "official":
                entry["trust_tier"] = url_tier
        normalized.append(entry)
    return normalized


def present_preview(result: PreviewResult, mode: str = "summary") -> str:
    lines = [
        f"### Proposed Change ({result.risk_level.value.upper()})",
        f"- **Impact:** {result.impact_count} records",
    ]
    if mode == "details" and result.preview:
        lines.append("- **Affected records:**")
        for item in result.preview.get("affected", []):
            lines.append(f"  - ID: {item.get('id')} | Name: {item.get('name', 'N/A')}")
        lines.append(f"- **Exact API call:** POST {result.proposed_payload.get('endpoint', 'N/A')}")
        lines.append("- **Backup advised:** This action cannot be undone.")
    elif result.preview:
        lines.append("- **Preview:**")
        for key, value in result.preview.items():
            lines.append(f"  - {key}: {value}")
    if result.informing_sources:
        lines.append("\n**Informed by:**")
        for src in normalize_informing_sources(result.informing_sources):
            tier_label = src.get("trust_tier", "")
            title = src.get("title", "Untitled")
            url = src.get("url", "")
            if tier_label == "official":
                lines.append(f"- [Official: {title}]({url})")
            else:
                display_tier = tier_label.replace("-", " ")
                lines.append(f"- [{display_tier.title()}: {title}]({url})")

    if result.risk_level == RiskLevel.DESTRUCTIVE:
        lines.append(f"\n**Destructive action.** Type `{result.impact_count}` to confirm, or `details` for full record list.")
    else:
        lines.append("\nApprove? (y/n/details)")
    return "\n".join(lines)


def store_preview_for_execution(
    portal_id: str,
    action_id: str,
    result: PreviewResult,
) -> Path:
    snapshot_dir = Path.home() / ".claude" / "hubspot" / portal_id / "undo_snapshots"
    return save_undo_snapshot(str(snapshot_dir), action_id, result.original_values)


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def dispatch_agent(
    agent_name: str,
    user_request: str,
    portal_config: PortalConfig | None = None,
    mode: str = "preview",
    payload: dict[str, Any] | None = None,
) -> AgentResult:
    getter = _AGENT_GETTERS.get(agent_name)
    if getter is None:
        return AgentResult(
            agent_name=agent_name,
            status="error",
            error_message=f"Unknown agent: {agent_name}",
        )

    action_id = str(uuid.uuid4())[:8]

    # Idempotency check for writes
    if mode == "execute" and payload is not None and portal_config is not None:
        ledger = ActionLedger(portal_config.portal_id)
        action_label = user_request.strip().splitlines()[0][:120]
        duplicate = ledger.find_similar_in_flight(agent_name, action_label, payload)
        if duplicate is not None:
            return AgentResult(
                agent_name=agent_name,
                status="duplicate",
                error_message=(
                    f"Similar action already in flight (started at {duplicate.get('timestamp')}). "
                    f"Wait for it to complete or cancel before retrying."
                ),
                data={"duplicate_action_id": duplicate.get("action_id")},
            )
        ledger.start_action(action_id, agent_name, action_label, payload)

    prompt = getter(portal_config)
    full_prompt_parts = [prompt.system_prompt, f"\nUser request: {user_request}", f"\nMode: {mode}"]

    if mode == "execute" and payload is not None:
        full_prompt_parts.append(f"\nExecute the following payload:\n```json\n{json.dumps(payload, indent=2)}\n```")

    full_prompt = "\n".join(full_prompt_parts)

    data: dict[str, Any] = {
        "system_prompt": prompt.system_prompt,
        "full_prompt": full_prompt,
        "tool_names": prompt.tool_names,
    }
    if mode == "execute" and payload is not None:
        data["action_id"] = action_id

    return AgentResult(
        agent_name=agent_name,
        status="preview" if mode == "preview" else "ready",
        data=data,
    )


def record_action_completion(portal_id: str, action_id: str, result: dict[str, Any]) -> None:
    ledger = ActionLedger(portal_id)
    ledger.complete_action(action_id, result)


# ---------------------------------------------------------------------------
# Post-timeout reconciliation
# ---------------------------------------------------------------------------


def reconcile_after_timeout(
    portal_id: str,
    expected_action: str,
    expected_payload: dict[str, Any],
) -> dict[str, Any]:
    action_id = str(uuid.uuid4())[:8]
    return {
        "action_id": action_id,
        "portal_id": portal_id,
        "expected_action": expected_action,
        "expected_payload": expected_payload,
        "reconciliation_needed": True,
        "instruction": (
            f"A previous write operation timed out. "
            f"Dispatch HygieneAgent to verify state for action '{expected_action}'. "
            f"Compare expected payload against actual HubSpot state and report discrepancies."
        ),
    }
