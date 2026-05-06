from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _audit_file_path(portal_id: str) -> Path:
    base = Path.home() / ".claude" / "hubspot" / portal_id
    base.mkdir(parents=True, exist_ok=True)
    return base / "audit.log"


def log_write(
    portal_id: str,
    action: str,
    agent: str,
    result_summary: dict[str, Any],
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "agent": agent,
        "result_summary": result_summary,
    }
    file_path = _audit_file_path(portal_id)
    with file_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def get_recent_audits(portal_id: str, limit: int = 50) -> list[dict[str, Any]]:
    file_path = _audit_file_path(portal_id)
    if not file_path.exists():
        return []
    lines = file_path.read_text().strip().splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    return entries[-limit:]
