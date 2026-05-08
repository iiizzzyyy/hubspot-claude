from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hubspot_agent.client import APIResponse, HubSpotClient
from hubspot_agent.config import PortalConfig
from hubspot_agent.trace import TraceEvent, _parse_trace_id_events


@dataclass
class ReplayResult:
    matched: bool
    divergences: list[str] = field(default_factory=list)
    tool_call_comparison: dict[str, Any] = field(default_factory=dict)


class MockHubSpotClient(HubSpotClient):
    """Mock client that replays recorded APIResponses by intercepting _request().

    Records divergences when the actual request signature does not match the
    recorded expectation, or when too many / too few requests are made.
    """

    def __init__(
        self,
        portal: PortalConfig,
        recorded_responses: list[tuple[dict[str, Any], APIResponse]],
    ) -> None:
        # Intentionally skip HubSpotClient.__init__ to avoid creating an
        # httpx.AsyncClient.
        self.portal = portal
        self._recorded = recorded_responses
        self._index = 0
        self._divergences: list[str] = []
        self._actual_calls: list[dict[str, Any]] = []

    @property
    def has_unused_responses(self) -> bool:
        return self._index < len(self._recorded)

    @property
    def unused_response_count(self) -> int:
        return len(self._recorded) - self._index

    async def _request(
        self,
        method: str,
        path: str,
        portal_id: str,
        body: dict[str, Any] | None = None,
        expected_scopes: list[str] | None = None,
    ) -> APIResponse:
        if self._index >= len(self._recorded):
            self._divergences.append(
                f"Unexpected request #{self._index + 1}: {method} {path}"
            )
            return APIResponse(status_code=200, body={}, headers={})

        expected_req, expected_resp = self._recorded[self._index]
        mismatches: list[str] = []

        if expected_req.get("method") != method:
            mismatches.append(
                f"method expected {expected_req.get('method')}, got {method}"
            )
        if expected_req.get("path") != path:
            mismatches.append(
                f"path expected {expected_req.get('path')}, got {path}"
            )
        if expected_req.get("body") != body:
            mismatches.append(
                f"body expected {expected_req.get('body')}, got {body}"
            )

        if mismatches:
            self._divergences.append(
                f"Request #{self._index + 1} mismatch: {'; '.join(mismatches)}"
            )

        self._actual_calls.append({
            "method": method,
            "path": path,
            "body": body,
            "response_status": expected_resp.status_code,
        })
        self._index += 1
        return expected_resp

    async def close(self) -> None:
        pass


class ReplayEngine:
    @staticmethod
    def load_trace(trace_id: str, portal_id: str) -> list[TraceEvent]:
        """Load all events for a given trace from traces.jsonl."""
        return _parse_trace_id_events(portal_id, trace_id)

    @staticmethod
    def _extract_recorded_responses(
        events: list[TraceEvent],
    ) -> list[tuple[dict[str, Any], APIResponse]]:
        """Extract (request, response) pairs from trace events."""
        responses: list[tuple[dict[str, Any], APIResponse]] = []
        for event in events:
            req = event.data.get("recorded_request")
            resp = event.data.get("recorded_response")
            if req is not None and resp is not None:
                responses.append((
                    req,
                    APIResponse(
                        status_code=resp["status_code"],
                        body=resp.get("body", {}),
                        headers=resp.get("headers", {}),
                    ),
                ))
        return responses

    @classmethod
    def mock_client_from_trace(
        cls,
        events: list[TraceEvent],
        portal: PortalConfig,
    ) -> HubSpotClient:
        """Build a mock client that replays recorded responses from a trace."""
        recorded = cls._extract_recorded_responses(events)
        return MockHubSpotClient(portal, recorded)

    @classmethod
    async def replay(
        cls,
        events: list[TraceEvent],
        portal_config: PortalConfig,
    ) -> ReplayResult:
        """Replay a trace against a mock client and assert identical behavior.

        Drives the mock client through every recorded API call in the trace,
        collecting divergences when requests or responses do not match.
        """
        mock = cls.mock_client_from_trace(events, portal_config)
        divergences: list[str] = list(mock._divergences)

        recorded = cls._extract_recorded_responses(events)

        for i, (req, expected_resp) in enumerate(recorded):
            actual = await mock._request(
                req["method"],
                req["path"],
                portal_config.portal_id,
                req.get("body"),
                req.get("expected_scopes"),
            )
            if actual.status_code != expected_resp.status_code:
                divergences.append(
                    f"Response {i + 1} status mismatch: "
                    f"expected {expected_resp.status_code}, got {actual.status_code}"
                )
            if actual.body != expected_resp.body:
                divergences.append(f"Response {i + 1} body mismatch")

        if mock.has_unused_responses:
            divergences.append(
                f"{mock.unused_response_count} recorded response(s) were not consumed"
            )

        original_tool_calls = [e for e in events if e.event_type == "tool_call"]
        tool_call_comparison: dict[str, Any] = {
            "original_count": len(original_tool_calls),
            "replayed_count": mock._index,
            "by_tool_name": {},
        }
        for tc in original_tool_calls:
            name = tc.data.get("tool_name", tc.data.get("agent", "unknown"))
            tool_call_comparison["by_tool_name"][name] = (
                tool_call_comparison["by_tool_name"].get(name, 0) + 1
            )

        if not recorded and not original_tool_calls:
            divergences.append("Trace is empty: no events to replay")

        if not recorded and original_tool_calls:
            divergences.append(
                "Trace has tool_call events but no recorded_request/recorded_response pairs"
            )

        matched = len(divergences) == 0

        return ReplayResult(
            matched=matched,
            divergences=divergences,
            tool_call_comparison=tool_call_comparison,
        )
