from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_undo_snapshot(
    snapshot_dir: str,
    action_id: str,
    original_values: dict[str, Any],
) -> Path:
    dir_path = Path(snapshot_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{action_id}.json"
    file_path.write_text(
        json.dumps({"action_id": action_id, "original_values": original_values}, indent=2)
    )
    return file_path


def load_undo_snapshot(snapshot_dir: str, action_id: str) -> dict[str, Any] | None:
    file_path = Path(snapshot_dir) / f"{action_id}.json"
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text())


def delete_undo_snapshot(snapshot_dir: str, action_id: str) -> None:
    file_path = Path(snapshot_dir) / f"{action_id}.json"
    if file_path.exists():
        file_path.unlink()
