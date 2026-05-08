from __future__ import annotations

import asyncio
import inspect
import re
import uuid
from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from hubspot_agent.client import HubSpotClient
from hubspot_agent.config import PortalConfig
from hubspot_agent.models import AgentResult, RiskLevel
from hubspot_agent.trace import emit_trace


class PlanNode(BaseModel):
    node_id: str
    agent: str
    action: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    payload_summary: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel


class DAGPlan(BaseModel):
    plan_id: str
    nodes: list[PlanNode]
    edges: list[tuple[str, str]]
    overall_risk: RiskLevel
    estimated_duration_seconds: int


class DAGPlanner:
    _COMPOUND_SEQUENCERS: list[str] = [
        r"\band then\b",
        r"\bthen\b",
        r"\bnext\b",
        r"\bafter that\b",
        r"\bfirst\b.*?\b(?:then|next|after)\b",
        r"\bsecondly\b",
        r"\bthirdly\b",
        r"\bfollowed by\b",
        r"\bsubsequently\b",
        r"\blater\b",
        r"\bonce that(?: is|'s)? done\b",
        r"\bafterwards\b",
        r"\bmeanwhile\b",
        r"\bwhile\b",
        r"\bupon completion\b",
        r"\bstep\s+\d+",
        r"^\s*\d+\.\s+",
        r"^\s*\d+\)\s+",
    ]

    _OUTPUT_KEYWORDS: dict[str, list[str]] = {
        "objects": ["contact", "company", "deal", "ticket", "record"],
        "properties": ["property", "field", "schema"],
        "workflows": ["workflow", "automation"],
        "lists": ["list", "segment"],
        "engagements": ["note", "task", "meeting", "call", "activity"],
        "pipelines": ["pipeline", "stage"],
        "users": ["user"],
    }

    def __init__(
        self,
        fast_path_keywords: dict[str, list[str]] | None = None,
        static_dependencies: dict[str, list[str]] | None = None,
        agent_getters: dict[str, Any] | None = None,
    ) -> None:
        self._fast_path_keywords = fast_path_keywords or {}
        self._static_dependencies = static_dependencies or {}
        self._agent_getters = agent_getters or {}

    def _is_compound_request(self, request_text: str) -> bool:
        text = request_text.lower()
        sequencer_hits = sum(
            1 for pattern in self._COMPOUND_SEQUENCERS
            if re.search(pattern, text, re.IGNORECASE)
        )
        if sequencer_hits > 0:
            return True

        agent_hits: set[str] = set()
        all_keywords = dict(self._fast_path_keywords)
        for agent, keywords in self._OUTPUT_KEYWORDS.items():
            if agent not in all_keywords:
                all_keywords[agent] = keywords

        for agent, keywords in all_keywords.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}s?\b", text, re.IGNORECASE):
                    agent_hits.add(agent)
                    break

        if len(agent_hits) >= 2:
            return True

        numbered = re.findall(r"(?:^|\n)\s*\d+[.\)]\s+\w+", text, re.IGNORECASE)
        if len(numbered) >= 2:
            return True

        return False

    def _extract_operations(self, request_text: str) -> list[dict[str, Any]]:
        text = request_text.strip()
        segments = self._split_into_segments(text)
        operations: list[dict[str, Any]] = []
        unmapped: list[str] = []
        for seg in segments:
            agent = self._map_segment_to_agent(seg)
            if agent is None:
                unmapped.append(seg)
                continue
            action = self._infer_action(seg)
            outputs = self._infer_outputs(seg, agent)
            inputs = self._infer_inputs(seg)
            operations.append({
                "agent": agent,
                "action": action,
                "text": seg,
                "outputs": outputs,
                "inputs": inputs,
            })
        if unmapped:
            raise ValueError(
                f"Could not map {len(unmapped)} segment(s) to agents: "
                f"{unmapped!r}. Please rephrase or split into separate requests."
            )
        return operations

    def _split_into_segments(self, text: str) -> list[str]:
        delimiters = (
            r"(?:\band then\b|\bthen\b|\bnext\b|\bafter that\b|\bfollowed by\b"
            r"|\bsubsequently\b|\bafterwards\b|\bonce that(?: is|'s)? done\b"
            r"|\bupon completion\b|\bmeanwhile\b)"
        )
        parts = re.split(delimiters, text, flags=re.IGNORECASE)

        numbered_split: list[str] = []
        for part in parts:
            subparts = re.split(
                r"(?:\n|^)\s*(?:\d+[.\)]\s+)(?=\w)", part.strip(), flags=re.IGNORECASE
            )
            numbered_split.extend([p.strip() for p in subparts if p.strip()])

        if len(numbered_split) <= 1:
            sentences = re.split(r"(?<=[.!?])\s+(?=\w)", text)
            numbered_split = [s.strip() for s in sentences if s.strip()]

        return numbered_split

    def _map_segment_to_agent(self, segment: str) -> str | None:
        text = segment.lower()
        scores: dict[str, int] = {}
        for agent, keywords in self._fast_path_keywords.items():
            score = sum(
                1 for kw in keywords
                if re.search(rf"\b{re.escape(kw)}s?\b", text)
            )
            if score > 0:
                scores[agent] = score
        if not scores:
            for agent in self._agent_getters:
                if agent == "associations" and re.search(r"\blink\b", text):
                    scores["associations"] = 1
                elif agent == "hygiene" and re.search(r"\bduplicate\b", text):
                    scores["hygiene"] = 1
                elif agent == "analytics" and re.search(r"\breport\b", text):
                    scores["analytics"] = 1
                elif agent == "raw_api" and re.search(r"\bapi\b", text):
                    scores["raw_api"] = 1
        if not scores:
            return None
        return max(scores, key=lambda a: scores[a])

    def _infer_action(self, segment: str) -> str:
        text = segment.lower()
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in ("create", "add", "new", "insert")):
            return "create"
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in ("update", "change", "modify", "edit", "set")):
            return "update"
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in ("delete", "remove", "drop", "clear")):
            return "delete"
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in ("find", "search", "get", "list", "lookup", "fetch")):
            return "read"
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in ("enroll", "trigger", "activate")):
            return "trigger"
        return "read"

    def _infer_outputs(self, segment: str, agent: str) -> list[str]:
        text = segment.lower()
        outputs: list[str] = []
        keywords = self._OUTPUT_KEYWORDS.get(agent, [])
        for kw in keywords:
            if kw in text:
                outputs.append(kw)
        if "create" in text or "add" in text or "new" in text:
            outputs.append("created_id")
        return list(set(outputs))

    def _infer_inputs(self, segment: str) -> dict[str, Any]:
        text = segment.lower()
        inputs: dict[str, Any] = {}
        if "id" in text:
            inputs["needs_id"] = True
        if "contact" in text:
            inputs["object_type"] = "contacts"
        elif "company" in text:
            inputs["object_type"] = "companies"
        elif "deal" in text:
            inputs["object_type"] = "deals"
        elif "ticket" in text:
            inputs["object_type"] = "tickets"
        return inputs

    def _build_nodes(
        self,
        operations: list[dict[str, Any]],
        portal_config: PortalConfig,
    ) -> list[PlanNode]:
        nodes: list[PlanNode] = []
        for i, op in enumerate(operations):
            risk = self._assess_risk(op)
            node = PlanNode(
                node_id=f"node-{i + 1}",
                agent=op["agent"],
                action=op["action"],
                inputs=op.get("inputs", {}),
                outputs=op.get("outputs", []),
                dependencies=[],
                payload_summary={"text": op["text"]},
                risk_level=risk,
            )
            nodes.append(node)
        return nodes

    def _assess_risk(self, op: dict[str, Any]) -> RiskLevel:
        action = op.get("action", "read")
        if action == "delete":
            return RiskLevel.DESTRUCTIVE
        if action in ("create", "update"):
            return RiskLevel.HIGH
        if action == "trigger":
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _derive_edges(self, nodes: list[PlanNode]) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        node_ids = {n.node_id for n in nodes}

        # Preserve pre-existing dependencies as edges
        for n in nodes:
            for dep_id in n.dependencies:
                if dep_id in node_ids and (dep_id, n.node_id) not in edges:
                    edges.append((dep_id, n.node_id))

        agent_to_nodes: dict[str, list[PlanNode]] = {}
        for n in nodes:
            agent_to_nodes.setdefault(n.agent, []).append(n)

        for n in nodes:
            deps = self._static_dependencies.get(n.agent, [])
            for dep_agent in deps:
                dep_nodes = agent_to_nodes.get(dep_agent, [])
                for dep in dep_nodes:
                    if dep.node_id != n.node_id and dep.node_id not in n.dependencies:
                        if (dep.node_id, n.node_id) not in edges:
                            edges.append((dep.node_id, n.node_id))

        for i, n in enumerate(nodes):
            for j in range(i):
                prev = nodes[j]
                if self._has_data_flow(prev, n):
                    if (prev.node_id, n.node_id) not in edges:
                        edges.append((prev.node_id, n.node_id))

        edges = [(src, dst) for src, dst in edges if src != dst]
        return edges

    def _has_data_flow(self, source: PlanNode, target: PlanNode) -> bool:
        if target.inputs.get("needs_id") and "created_id" in source.outputs:
            return True
        target_obj = target.inputs.get("object_type")
        if target_obj:
            singular = target_obj.rstrip("s")
            if singular in source.outputs or target_obj in source.outputs:
                return True
        return False

    def _compute_overall_risk(self, nodes: list[PlanNode]) -> RiskLevel:
        risks = [n.risk_level for n in nodes]
        if RiskLevel.DESTRUCTIVE in risks:
            return RiskLevel.DESTRUCTIVE
        if RiskLevel.HIGH in risks:
            return RiskLevel.HIGH
        if RiskLevel.MEDIUM in risks:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _estimate_duration(self, nodes: list[PlanNode], edges: list[tuple[str, str]]) -> int:
        base = len(nodes) * 5
        overhead = len(edges) * 3
        return base + overhead

    def generate(self, request_text: str, portal_config: PortalConfig) -> DAGPlan:
        operations = self._extract_operations(request_text)
        nodes = self._build_nodes(operations, portal_config)
        if not nodes:
            raise ValueError("No actionable operations found in request")
        edges = self._derive_edges(nodes)
        overall_risk = self._compute_overall_risk(nodes)
        estimated_duration = self._estimate_duration(nodes, edges)
        return DAGPlan(
            plan_id=f"plan-{uuid.uuid4().hex[:8]}",
            nodes=nodes,
            edges=edges,
            overall_risk=overall_risk,
            estimated_duration_seconds=estimated_duration,
        )


class PlanExecutor:
    def __init__(
        self,
        dispatch_agent_fn: Any | None = None,
        client: HubSpotClient | None = None,
    ) -> None:
        self._dispatch_agent = dispatch_agent_fn
        self._client = client

    def _topological_sort(
        self,
        nodes: list[PlanNode],
        edges: list[tuple[str, str]],
    ) -> list[PlanNode]:
        adjacency: dict[str, list[str]] = {n.node_id: [] for n in nodes}
        in_degree: dict[str, int] = {n.node_id: 0 for n in nodes}
        for src, dst in edges:
            if src in adjacency and dst in adjacency:
                adjacency[src].append(dst)
                in_degree[dst] += 1

        queue = deque([n for n in nodes if in_degree[n.node_id] == 0])
        result: list[PlanNode] = []
        node_map = {n.node_id: n for n in nodes}

        while queue:
            current = queue.popleft()
            result.append(current)
            for neighbor_id in adjacency[current.node_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(node_map[neighbor_id])

        if len(result) != len(nodes):
            raise ValueError("Cyclic dependency detected in DAG plan")

        return result

    def _resolve_inputs(
        self,
        node: PlanNode,
        completed_results: dict[str, AgentResult],
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(node.inputs)
        for dep_id in node.dependencies:
            dep_result = completed_results.get(dep_id)
            if dep_result is None:
                continue
            dep_data = dep_result.data or {}
            if "created_id" in dep_data:
                resolved.setdefault("source_id", dep_data["created_id"])
            if "records" in dep_data:
                resolved.setdefault("source_records", dep_data["records"])
            if "object_type" in dep_data:
                resolved.setdefault("source_object_type", dep_data["object_type"])
        return resolved

    def _get_object_type(self, node: PlanNode) -> str:
        return (
            node.inputs.get("object_type")
            or node.inputs.get("object_type_id")
            or node.agent
        )

    def _should_start_buffer(
        self,
        node: PlanNode,
        ordered_nodes: list[PlanNode],
        idx: int,
    ) -> bool:
        object_type = self._get_object_type(node)
        for j in range(idx + 1, len(ordered_nodes)):
            next_node = ordered_nodes[j]
            if WriteBuffer.can_coalesce(next_node):
                return self._get_object_type(next_node) == object_type
        return False

    def _build_node_payload(
        self,
        node: PlanNode,
        completed_results: dict[str, AgentResult],
    ) -> dict[str, Any]:
        resolved_inputs = self._resolve_inputs(node, completed_results)
        payload = dict(node.payload_summary)
        payload["resolved_inputs"] = resolved_inputs
        return payload

    async def execute(
        self,
        plan: DAGPlan,
        portal_config: PortalConfig,
        trace_id: str,
    ) -> list[AgentResult]:
        emit_trace(
            portal_config.portal_id,
            "tool_call",
            trace_id,
            {
                "plan_id": plan.plan_id,
                "node_count": len(plan.nodes),
                "event": "plan_execution_start",
            },
        )

        ordered_nodes = self._topological_sort(plan.nodes, plan.edges)
        completed_results: dict[str, AgentResult] = {}
        all_results: list[AgentResult] = []
        buffer = WriteBuffer(client=self._client)
        current_buffer_type: str | None = None

        async def _flush_buffer() -> bool:
            if not buffer._buffer:
                return False
            batch_results = await buffer.flush(portal_config)
            for batch_result, (buf_node, _) in zip(batch_results, buffer._buffer):
                completed_results[buf_node.node_id] = batch_result
                all_results.append(batch_result)
            has_error = any(r.status in ("error", "duplicate") for r in batch_results)
            buffer._buffer.clear()
            if has_error:
                emit_trace(
                    portal_config.portal_id,
                    "error",
                    trace_id,
                    {
                        "plan_id": plan.plan_id,
                        "event": "batch_flush_error",
                        "error_count": sum(
                            1 for r in batch_results if r.status == "error"
                        ),
                    },
                )
            return has_error

        for i, node in enumerate(ordered_nodes):
            node_type = (
                self._get_object_type(node)
                if WriteBuffer.can_coalesce(node)
                else None
            )

            if current_buffer_type is not None:
                if node_type == current_buffer_type:
                    payload = self._build_node_payload(node, completed_results)
                    buffer.buffer_write(node, payload)
                    continue
                else:
                    if await _flush_buffer():
                        break
                    current_buffer_type = None

            if node_type is not None:
                if self._should_start_buffer(node, ordered_nodes, i):
                    current_buffer_type = node_type
                    payload = self._build_node_payload(node, completed_results)
                    buffer.buffer_write(node, payload)
                    continue

            if self._dispatch_agent is None:
                raise RuntimeError(
                    "No dispatch agent function configured for PlanExecutor"
                )

            payload = self._build_node_payload(node, completed_results)
            raw_result = self._dispatch_agent(
                agent_name=node.agent,
                user_request=node.payload_summary.get("text", ""),
                portal_config=portal_config,
                mode="execute",
                payload=payload,
                trace_id=trace_id,
            )

            if inspect.iscoroutine(raw_result):
                result = await raw_result
            else:
                result = raw_result

            completed_results[node.node_id] = result
            all_results.append(result)

            if result.status in ("error", "duplicate"):
                emit_trace(
                    portal_config.portal_id,
                    "error",
                    trace_id,
                    {
                        "plan_id": plan.plan_id,
                        "node_id": node.node_id,
                        "agent": node.agent,
                        "status": result.status,
                        "error": result.error_message,
                    },
                )
                break

            emit_trace(
                portal_config.portal_id,
                "tool_call",
                trace_id,
                {
                    "plan_id": plan.plan_id,
                    "node_id": node.node_id,
                    "agent": node.agent,
                    "status": result.status,
                },
            )

        if buffer._buffer:
            batch_results = await buffer.flush(portal_config)
            for batch_result, (buf_node, _) in zip(batch_results, buffer._buffer):
                completed_results[buf_node.node_id] = batch_result
                all_results.append(batch_result)
            buffer._buffer.clear()

        emit_trace(
            portal_config.portal_id,
            "completion",
            trace_id,
            {
                "plan_id": plan.plan_id,
                "event": "plan_execution_complete",
                "results_count": len(all_results),
                "completed_count": len(
                    [r for r in all_results if r.status == "success"]
                ),
            },
        )

        return all_results


class WriteBuffer:
    def __init__(self, client: HubSpotClient | None = None) -> None:
        self._buffer: list[tuple[PlanNode, dict[str, Any]]] = []
        self._client = client

    def buffer_write(self, node: PlanNode, payload: dict[str, Any]) -> None:
        self._buffer.append((node, payload))

    @staticmethod
    def can_coalesce(node: PlanNode) -> bool:
        return (
            node.agent in ("objects", "lists", "properties")
            and node.action in ("create", "update", "delete")
        )

    def _get_object_type(self, node: PlanNode) -> str:
        return (
            node.inputs.get("object_type")
            or node.inputs.get("object_type_id")
            or node.agent
        )

    def _derive_endpoint(self, node: PlanNode) -> str:
        object_type = self._get_object_type(node)
        if node.agent == "objects":
            if node.action == "create":
                return f"/crm/v3/objects/{object_type}/batch/create"
            elif node.action == "update":
                return f"/crm/v3/objects/{object_type}/batch/update"
            elif node.action == "delete":
                return f"/crm/v3/objects/{object_type}/batch/archive"
        elif node.agent == "properties":
            if node.action == "create":
                return f"/crm/v3/properties/{object_type}/batch/create"
            elif node.action == "update":
                return f"/crm/v3/properties/{object_type}/batch/update"
            elif node.action == "delete":
                return f"/crm/v3/properties/{object_type}/batch/archive"
        elif node.agent == "lists":
            if node.action == "create":
                return "/crm/v3/lists/batch/create"
            elif node.action == "update":
                return "/crm/v3/lists/batch/update"
            elif node.action == "delete":
                return "/crm/v3/lists/batch/archive"
        return f"{node.agent}/{node.action}"

    def _build_batch_input(
        self, node: PlanNode, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if node.agent == "objects":
            if node.action == "create":
                return {"properties": payload.get("properties", {})}
            elif node.action == "update":
                return {
                    "id": payload.get("object_id"),
                    "properties": payload.get("properties", {}),
                }
            elif node.action == "delete":
                return {"id": payload.get("object_id")}
        elif node.agent == "properties":
            if node.action == "create":
                return {
                    "name": payload.get("name"),
                    "label": payload.get("label"),
                    "type": payload.get("property_type"),
                    "fieldType": payload.get("field_type"),
                    "groupName": payload.get("group_name", "contactinformation"),
                }
            elif node.action == "update":
                return {
                    "name": payload.get("property_name"),
                    **payload.get("updates", {}),
                }
            elif node.action == "delete":
                return {"name": payload.get("property_name")}
        elif node.agent == "lists":
            if node.action == "create":
                return {
                    "name": payload.get("name"),
                    "objectTypeId": payload.get("object_type_id"),
                    "processingType": payload.get("processing_type"),
                }
            elif node.action == "update":
                return {
                    "listId": payload.get("list_id"),
                    **payload.get("updates", {}),
                }
            elif node.action == "delete":
                return {"listId": payload.get("list_id")}
        return payload

    def _expected_scopes(self, node: PlanNode) -> list[str]:
        object_type = self._get_object_type(node)
        if node.agent == "objects":
            return [f"crm.objects.{object_type}.write"]
        elif node.agent == "properties":
            return [f"crm.schemas.{object_type}.write"]
        elif node.agent == "lists":
            return ["crm.lists.write"]
        return []

    def _group_by_endpoint(
        self,
    ) -> dict[str, list[tuple[PlanNode, dict[str, Any]]]]:
        groups: dict[str, list[tuple[PlanNode, dict[str, Any]]]] = {}
        for node, payload in self._buffer:
            key = f"{self._get_object_type(node)}:{self._derive_endpoint(node)}"
            groups.setdefault(key, []).append((node, payload))
        return groups

    def _map_batch_error_to_nodes(
        self, batch_error: dict, nodes: list[PlanNode]
    ) -> dict[str, str]:
        errors = batch_error.get("errors", [])
        node_errors: dict[str, str] = {}
        for error in errors:
            obj_id = (
                error.get("id")
                or error.get("object_id")
                or error.get("listId")
                or error.get("name")
            )
            if obj_id:
                for node in nodes:
                    candidate = (
                        node.inputs.get("object_id")
                        or node.inputs.get("property_name")
                        or node.inputs.get("list_id")
                        or node.payload_summary.get("object_id")
                        or node.payload_summary.get("property_name")
                        or node.payload_summary.get("list_id")
                        or node.payload_summary.get("name")
                    )
                    if candidate and str(candidate) == str(obj_id):
                        node_errors[node.node_id] = error.get(
                            "message", "Unknown error"
                        )
                        break
            index = (
                error.get("inputIndex")
                if "inputIndex" in error
                else error.get("index")
            )
            if (
                index is not None
                and 0 <= index < len(nodes)
                and nodes[index].node_id not in node_errors
            ):
                node_errors[nodes[index].node_id] = error.get(
                    "message", "Unknown error"
                )
        return node_errors

    async def flush(self, portal_config: PortalConfig) -> list[AgentResult]:
        if not self._buffer:
            return []

        client = self._client
        owns_client = False
        if client is None:
            client = HubSpotClient(portal_config)
            owns_client = True

        try:
            groups = self._group_by_endpoint()
            results_by_node: dict[str, AgentResult] = {}

            for key, items in groups.items():
                nodes = [item[0] for item in items]
                payloads = [item[1] for item in items]
                inputs = [
                    self._build_batch_input(n, p) for n, p in zip(nodes, payloads)
                ]
                endpoint = self._derive_endpoint(nodes[0])
                body = {"inputs": inputs}

                try:
                    resp = await client.post(
                        endpoint,
                        portal_id=portal_config.portal_id,
                        body=body,
                        expected_scopes=self._expected_scopes(nodes[0]),
                    )
                    response_body = resp.body
                except Exception as exc:
                    for node in nodes:
                        results_by_node[node.node_id] = AgentResult(
                            agent_name=node.agent,
                            status="error",
                            error_message=str(exc),
                        )
                    continue

                errors = response_body.get("errors", [])
                results = response_body.get("results", [])
                node_errors = self._map_batch_error_to_nodes(
                    {"errors": errors}, nodes
                )

                for idx, node in enumerate(nodes):
                    if node.node_id in node_errors:
                        results_by_node[node.node_id] = AgentResult(
                            agent_name=node.agent,
                            status="error",
                            error_message=node_errors[node.node_id],
                            data={},
                        )
                    else:
                        data = results[idx] if idx < len(results) else {}
                        results_by_node[node.node_id] = AgentResult(
                            agent_name=node.agent,
                            status="success",
                            data=data if data else {},
                        )

            return [
                results_by_node[node.node_id] for node, _ in self._buffer
            ]
        finally:
            if owns_client:
                await client.close()


class PlanModification(BaseModel):
    skip_nodes: list[str] = Field(default_factory=list)
    parameter_edits: dict[str, dict[str, Any]] = Field(default_factory=dict)


class InteractivePlanModifier:
    _SKIP_RE = re.compile(
        r"\bskip\s+(\S+)",
        re.IGNORECASE,
    )
    _EDIT_RE = re.compile(
        r"\bedit\s+(\S+)\s+(\w+)=([^\s,;]+)",
        re.IGNORECASE,
    )

    @staticmethod
    def _normalize_node_id(raw: str) -> str:
        raw = raw.strip(",;.")
        if re.match(r"^n\d+$", raw, re.IGNORECASE):
            return f"node-{raw[1:]}"
        return raw

    def parse_instruction(self, instruction_text: str) -> PlanModification:
        skip_nodes: list[str] = []
        parameter_edits: dict[str, dict[str, Any]] = {}

        for match in self._SKIP_RE.finditer(instruction_text):
            node_id = self._normalize_node_id(match.group(1))
            if node_id not in skip_nodes:
                skip_nodes.append(node_id)

        for match in self._EDIT_RE.finditer(instruction_text):
            node_id = self._normalize_node_id(match.group(1)).lower()
            field = match.group(2).lower()
            value = match.group(3)
            if node_id not in parameter_edits:
                parameter_edits[node_id] = {}
            parameter_edits[node_id][field] = value

        return PlanModification(
            skip_nodes=skip_nodes,
            parameter_edits=parameter_edits,
        )

    def apply(self, plan: DAGPlan, modification: PlanModification) -> DAGPlan:
        node_ids = {n.node_id for n in plan.nodes}

        for node_id in modification.skip_nodes:
            if node_id not in node_ids:
                raise ValueError(
                    f"Cannot skip unknown node: {node_id!r}. "
                    f"Valid nodes: {sorted(node_ids)}"
                )

        for node_id in modification.parameter_edits:
            if node_id not in node_ids:
                raise ValueError(
                    f"Cannot edit unknown node: {node_id!r}. "
                    f"Valid nodes: {sorted(node_ids)}"
                )

        skipped_set = set(modification.skip_nodes)

        if len(skipped_set) == len(node_ids):
            raise ValueError("Cannot skip all nodes: plan would be empty")

        new_nodes: list[PlanNode] = []
        for node in plan.nodes:
            if node.node_id in skipped_set:
                continue
            edits = modification.parameter_edits.get(node.node_id, {})
            if edits:
                new_payload = dict(node.payload_summary)
                new_payload.update(edits)
                node = node.model_copy(update={"payload_summary": new_payload})
            new_nodes.append(node)

        new_edges = [
            (src, dst)
            for src, dst in plan.edges
            if src not in skipped_set and dst not in skipped_set
        ]

        remaining_risks = [n.risk_level for n in new_nodes]
        overall_risk = RiskLevel.LOW
        if RiskLevel.DESTRUCTIVE in remaining_risks:
            overall_risk = RiskLevel.DESTRUCTIVE
        elif RiskLevel.HIGH in remaining_risks:
            overall_risk = RiskLevel.HIGH
        elif RiskLevel.MEDIUM in remaining_risks:
            overall_risk = RiskLevel.MEDIUM

        estimated_duration = len(new_nodes) * 5 + len(new_edges) * 3

        new_plan = DAGPlan(
            plan_id=plan.plan_id,
            nodes=new_nodes,
            edges=new_edges,
            overall_risk=overall_risk,
            estimated_duration_seconds=estimated_duration,
        )

        violations = self.validate_invariants(new_plan)
        if violations:
            raise ValueError(
                "Plan modification violates invariants:\n- " + "\n- ".join(violations)
            )

        return new_plan

    def validate_invariants(self, plan: DAGPlan) -> list[str]:
        violations: list[str] = []
        node_ids = {n.node_id for n in plan.nodes}

        if not node_ids:
            violations.append("Plan contains no nodes")

        for node in plan.nodes:
            for dep_id in node.dependencies:
                if dep_id not in node_ids:
                    violations.append(
                        f"Node {node.node_id!r} depends on missing node {dep_id!r}"
                    )

        adjacency: dict[str, list[str]] = {n.node_id: [] for n in plan.nodes}
        for src, dst in plan.edges:
            if src in adjacency and dst in adjacency:
                adjacency[src].append(dst)

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n.node_id: WHITE for n in plan.nodes}

        def dfs(node_id: str) -> bool:
            color[node_id] = GRAY
            for neighbor in adjacency[node_id]:
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node_id] = BLACK
            return False

        for node_id in list(adjacency.keys()):
            if color[node_id] == WHITE:
                if dfs(node_id):
                    violations.append("Cyclic dependency detected in DAG plan")
                    break

        return violations
