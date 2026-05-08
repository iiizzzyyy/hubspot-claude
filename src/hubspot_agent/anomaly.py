from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from hubspot_agent.maintenance import _portal_dir
from hubspot_agent.trace import get_recent_traces


class ToolBaseline(BaseModel):
    tool_name: str
    median_duration_ms: float
    failure_rate: float
    sample_size: int
    duration_std: float = 0.0
    failure_rate_std: float = 0.0


class AnomalyCheckResult(BaseModel):
    paused: bool
    deviation_sigma: float
    reason: str | None


class PortalBaselines(BaseModel):
    sigma_threshold: float = 3.0
    computed_at: str | None = None
    tools: dict[str, ToolBaseline] = Field(default_factory=dict)


class AnomalyDetector:
    def __init__(
        self,
        default_sigma_threshold: float = 3.0,
        recent_window_seconds: int = 3600,
    ) -> None:
        self.default_sigma_threshold = default_sigma_threshold
        self.recent_window_seconds = recent_window_seconds

    def _baselines_path(self, portal_id: str) -> Path:
        base = _portal_dir(portal_id)
        base.mkdir(parents=True, exist_ok=True)
        return base / "baselines.json"

    def _load_portal_baselines(self, portal_id: str) -> PortalBaselines:
        path = self._baselines_path(portal_id)
        if not path.exists():
            return PortalBaselines(sigma_threshold=self.default_sigma_threshold)
        try:
            raw = json.loads(path.read_text())
            return PortalBaselines.model_validate(raw)
        except (json.JSONDecodeError, Exception):
            return PortalBaselines(sigma_threshold=self.default_sigma_threshold)

    def _save_portal_baselines(self, portal_id: str, baselines: PortalBaselines) -> None:
        path = self._baselines_path(portal_id)
        path.write_text(baselines.model_dump_json(indent=2))

    @staticmethod
    def _extract_identifier(event_data: dict[str, Any]) -> str | None:
        agent = event_data.get("agent")
        if agent:
            return agent
        tool_name = event_data.get("tool_name")
        if tool_name:
            return tool_name
        return None

    def _collect_trace_metrics(
        self, events: list[Any], cutoff: float
    ) -> dict[str, list[tuple[float, bool]]]:
        """Group trace events by trace_id and collect duration/error per identifier."""
        trace_events: dict[str, list[Any]] = {}
        for e in events:
            if e.timestamp.timestamp() < cutoff:
                continue
            trace_events.setdefault(e.trace_id, []).append(e)

        tool_data: dict[str, list[tuple[float, bool]]] = {}
        for trace_id, evs in trace_events.items():
            if not evs:
                continue
            evs.sort(key=lambda e: e.timestamp)
            duration_ms = (evs[-1].timestamp - evs[0].timestamp).total_seconds() * 1000
            has_error = any(e.event_type == "error" for e in evs)

            identifiers: set[str] = set()
            for e in evs:
                if e.event_type == "tool_call":
                    ident = self._extract_identifier(e.data)
                    if ident:
                        identifiers.add(ident)

            for ident in identifiers:
                tool_data.setdefault(ident, []).append((duration_ms, has_error))

        return tool_data

    def compute_baselines(
        self, portal_id: str, window_hours: int = 168
    ) -> dict[str, ToolBaseline]:
        """Compute rolling median duration + failure rate per tool from traces."""
        events = get_recent_traces(portal_id, limit=5000)
        cutoff = datetime.now(timezone.utc).timestamp() - (window_hours * 3600)

        tool_data = self._collect_trace_metrics(events, cutoff)

        baselines: dict[str, ToolBaseline] = {}
        for tool_name, data in tool_data.items():
            durations = [d for d, _ in data]
            errors = [err for _, err in data]
            sample_size = len(data)

            if sample_size == 0:
                continue

            durations.sort()
            median_duration = float(durations[sample_size // 2]) if durations else 0.0
            failure_rate = sum(errors) / sample_size

            if sample_size > 1:
                mean_duration = sum(durations) / sample_size
                duration_std = math.sqrt(
                    sum((d - mean_duration) ** 2 for d in durations) / sample_size
                )
            else:
                duration_std = 0.0

            failure_rate_std = (
                math.sqrt(failure_rate * (1 - failure_rate) / sample_size)
                if sample_size > 0
                else 0.0
            )

            baselines[tool_name] = ToolBaseline(
                tool_name=tool_name,
                median_duration_ms=median_duration,
                failure_rate=failure_rate,
                sample_size=sample_size,
                duration_std=duration_std,
                failure_rate_std=failure_rate_std,
            )

        portal_baselines = PortalBaselines(
            sigma_threshold=self.default_sigma_threshold,
            computed_at=datetime.now(timezone.utc).isoformat(),
            tools=baselines,
        )
        self._save_portal_baselines(portal_id, portal_baselines)
        return baselines

    def check_request(
        self, portal_id: str, agent_name: str, tool_name: str
    ) -> AnomalyCheckResult:
        """Compare recent request patterns against baseline and pause if anomalous."""
        portal_baselines = self._load_portal_baselines(portal_id)
        threshold = portal_baselines.sigma_threshold

        baseline = portal_baselines.tools.get(tool_name)
        if baseline is None:
            baseline = portal_baselines.tools.get(agent_name)
        if baseline is None or baseline.sample_size < 5:
            return AnomalyCheckResult(paused=False, deviation_sigma=0.0, reason=None)

        events = get_recent_traces(portal_id, limit=5000)
        if not events:
            return AnomalyCheckResult(paused=False, deviation_sigma=0.0, reason=None)

        recent_cutoff = datetime.now(timezone.utc).timestamp() - self.recent_window_seconds
        recent_tool_data = self._collect_trace_metrics(events, recent_cutoff)

        recent_data = recent_tool_data.get(tool_name) or recent_tool_data.get(agent_name)
        if not recent_data or len(recent_data) < 3:
            return AnomalyCheckResult(paused=False, deviation_sigma=0.0, reason=None)

        recent_durations = [d for d, _ in recent_data]
        recent_errors = [err for _, err in recent_data]
        recent_n = len(recent_data)

        recent_durations.sort()
        recent_median = float(recent_durations[recent_n // 2])
        recent_failure_rate = sum(recent_errors) / recent_n

        max_sigma = 0.0
        reasons: list[str] = []

        # Duration check (only upward deviations)
        min_duration_std = max(baseline.duration_std, baseline.median_duration_ms * 0.05, 1.0)
        duration_sigma = (
            (recent_median - baseline.median_duration_ms) / min_duration_std
            if recent_median > baseline.median_duration_ms
            else 0.0
        )
        if duration_sigma > threshold:
            max_sigma = max(max_sigma, duration_sigma)
            reasons.append(
                f"duration median {recent_median:.1f}ms exceeds baseline "
                f"{baseline.median_duration_ms:.1f}ms by {duration_sigma:.1f} sigma"
            )

        # Failure rate check (only upward deviations)
        min_failure_std = max(baseline.failure_rate_std, 0.005)
        failure_sigma = (
            (recent_failure_rate - baseline.failure_rate) / min_failure_std
            if recent_failure_rate > baseline.failure_rate
            else 0.0
        )
        if failure_sigma > threshold:
            max_sigma = max(max_sigma, failure_sigma)
            reasons.append(
                f"failure rate {recent_failure_rate:.1%} exceeds baseline "
                f"{baseline.failure_rate:.1%} by {failure_sigma:.1f} sigma"
            )

        if max_sigma > threshold:
            return AnomalyCheckResult(
                paused=True,
                deviation_sigma=max_sigma,
                reason="; ".join(reasons),
            )

        return AnomalyCheckResult(
            paused=False, deviation_sigma=max(max_sigma, 0.0), reason=None
        )

    def set_sigma_threshold(self, portal_id: str, threshold: float) -> None:
        """Update the per-portal sigma threshold."""
        portal_baselines = self._load_portal_baselines(portal_id)
        portal_baselines.sigma_threshold = threshold
        self._save_portal_baselines(portal_id, portal_baselines)
