from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import hubspot_agent.config as _config


class SessionSummary(BaseModel):
    session_id: str
    portal_id: str
    started_at: datetime
    ended_at: datetime
    summary: str
    active_agents: list[str] = Field(default_factory=list)
    pending_approvals: int = 0
    custom_objects_discovered: list[str] = Field(default_factory=list)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return super().model_dump_json(**kwargs)


def _sessions_dir(portal_id: str) -> Path:
    return _config.CONFIG_DIR / portal_id / "sessions"


def _summary_file(portal_id: str, session_id: str) -> Path:
    return _sessions_dir(portal_id) / f"{session_id}.json"


def generate_summary(
    session_id: str,
    portal_id: str,
    started_at: datetime,
    active_agents: list[str],
    pending_approvals: int,
    custom_objects_discovered: list[str],
) -> str:
    """Generate a concise session summary text."""
    duration = datetime.now(timezone.utc) - started_at
    duration_str = f"{duration.total_seconds() / 60:.1f}m" if duration.total_seconds() < 3600 else f"{duration.total_seconds() / 3600:.1f}h"
    parts = [
        f"Session {session_id} for portal {portal_id} lasted {duration_str}.",
        f"Agents used: {', '.join(active_agents) if active_agents else 'none'}.",
    ]
    if pending_approvals:
        parts.append(f"Pending approvals at close: {pending_approvals}.")
    if custom_objects_discovered:
        parts.append(f"Custom objects discovered: {', '.join(custom_objects_discovered)}.")
    return " ".join(parts)


class SessionMemory:
    @staticmethod
    def save_summary(portal_id: str, session_id: str, summary: SessionSummary) -> None:
        path = _summary_file(portal_id, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary.model_dump_json())

    @staticmethod
    def load_last_summary(portal_id: str) -> SessionSummary | None:
        sessions_dir = _sessions_dir(portal_id)
        if not sessions_dir.exists():
            return None

        files = sorted(
            [f for f in sessions_dir.iterdir() if f.suffix == ".json"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return None

        try:
            data = json.loads(files[0].read_text())
            return SessionSummary(**data)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def list_sessions(portal_id: str, limit: int = 10) -> list[SessionSummary]:
        sessions_dir = _sessions_dir(portal_id)
        if not sessions_dir.exists():
            return []

        files = sorted(
            [f for f in sessions_dir.iterdir() if f.suffix == ".json"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        results: list[SessionSummary] = []
        for path in files[:limit]:
            try:
                data = json.loads(path.read_text())
                results.append(SessionSummary(**data))
            except (json.JSONDecodeError, ValueError):
                continue

        return results
