import os
import time
from pathlib import Path

import pytest

from hubspot_agent.maintenance import (
    MaintenanceReport,
    PruneReport,
    prune_completed_checkpoints,
    prune_snapshots,
    rotate_jsonl,
    run_maintenance,
)


def test_prune_snapshots_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    report = prune_snapshots("123")
    assert report.pruned_count == 0
    assert report.pruned_files == []


def test_prune_snapshots_removes_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    snapshot_dir = tmp_path / ".claude" / "hubspot" / "123" / "undo_snapshots"
    snapshot_dir.mkdir(parents=True)

    old_file = snapshot_dir / "old.json"
    old_file.write_text('{"action_id": "old"}')
    os.utime(old_file, (time.time() - 31 * 86400, time.time() - 31 * 86400))

    young_file = snapshot_dir / "young.json"
    young_file.write_text('{"action_id": "young"}')

    report = prune_snapshots("123", max_age_days=30)
    assert report.pruned_count == 1
    assert report.pruned_files == ["old.json"]
    assert not old_file.exists()
    assert young_file.exists()


def test_prune_checkpoints_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    report = prune_completed_checkpoints("123")
    assert report.pruned_count == 0
    assert report.pruned_files == []


def test_prune_checkpoints_removes_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    completed_dir = tmp_path / ".claude" / "hubspot" / "123" / "completed"
    completed_dir.mkdir(parents=True)

    old_file = completed_dir / "old.jsonl"
    old_file.write_text('{}')
    os.utime(old_file, (time.time() - 8 * 86400, time.time() - 8 * 86400))

    young_file = completed_dir / "young.jsonl"
    young_file.write_text('{}')

    report = prune_completed_checkpoints("123", max_age_days=7)
    assert report.pruned_count == 1
    assert report.pruned_files == ["old.jsonl"]
    assert not old_file.exists()
    assert young_file.exists()


def test_rotate_jsonl_missing_file(tmp_path):
    missing = tmp_path / "traces.jsonl"
    assert rotate_jsonl(missing) is False


def test_rotate_jsonl_below_threshold(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("small line\n")
    assert rotate_jsonl(path, max_size_mb=1) is False
    assert path.exists()


def test_rotate_jsonl_above_threshold(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("x" * (1024 * 1024 + 1))
    result = rotate_jsonl(path, max_size_mb=1)
    assert result is True
    assert not path.exists()
    rotated = tmp_path / "traces.1.jsonl"
    assert rotated.exists()


def test_rotate_jsonl_overwrites_existing_rotated(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("x" * (1024 * 1024 + 1))
    existing_rotated = tmp_path / "traces.1.jsonl"
    existing_rotated.write_text("old")
    rotate_jsonl(path, max_size_mb=1)
    assert existing_rotated.exists()
    assert existing_rotated.read_text() == "x" * (1024 * 1024 + 1)


@pytest.mark.asyncio
async def test_run_maintenance_integration(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    snapshot_dir = tmp_path / ".claude" / "hubspot" / "123" / "undo_snapshots"
    snapshot_dir.mkdir(parents=True)
    old_snapshot = snapshot_dir / "old.json"
    old_snapshot.write_text('{}')
    os.utime(old_snapshot, (time.time() - 31 * 86400, time.time() - 31 * 86400))

    completed_dir = tmp_path / ".claude" / "hubspot" / "123" / "completed"
    completed_dir.mkdir(parents=True)
    old_checkpoint = completed_dir / "old.jsonl"
    old_checkpoint.write_text('{}')
    os.utime(old_checkpoint, (time.time() - 8 * 86400, time.time() - 8 * 86400))

    traces_path = tmp_path / ".claude" / "hubspot" / "123" / "traces.jsonl"
    traces_path.write_text("x" * (1024 * 1024 + 1))

    report = await run_maintenance("123")
    assert isinstance(report, MaintenanceReport)
    assert report.snapshots.pruned_count == 1
    assert report.checkpoints.pruned_count == 1
    assert report.rotated is False
    assert not old_snapshot.exists()
    assert not old_checkpoint.exists()
    assert traces_path.exists()
    assert not (tmp_path / ".claude" / "hubspot" / "123" / "traces.1.jsonl").exists()


def test_validate_portal_id_rejects_invalid():
    from hubspot_agent.maintenance import _validate_portal_id

    _validate_portal_id("123")
    with pytest.raises(ValueError):
        _validate_portal_id("../../../etc")
    with pytest.raises(ValueError):
        _validate_portal_id("")
    with pytest.raises(ValueError):
        _validate_portal_id("abc")


def test_prune_snapshots_boundary_exactly_30_days(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    snapshot_dir = tmp_path / ".claude" / "hubspot" / "123" / "undo_snapshots"
    snapshot_dir.mkdir(parents=True)

    exact_file = snapshot_dir / "exact.json"
    exact_file.write_text('{}')
    # Fix time so boundary is deterministic
    fixed_now = 1000000000.0
    monkeypatch.setattr(time, "time", lambda: fixed_now)
    os.utime(exact_file, (fixed_now - 30 * 86400, fixed_now - 30 * 86400))

    report = prune_snapshots("123", max_age_days=30)
    assert report.pruned_count == 0
    assert exact_file.exists()


def test_rotate_jsonl_boundary_exactly_100_mb(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("x" * (100 * 1024 * 1024))
    assert rotate_jsonl(path, max_size_mb=100) is False
    assert path.exists()


def test_rotate_jsonl_with_string_path(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("x" * (1024 * 1024 + 1))
    result = rotate_jsonl(str(path), max_size_mb=1)
    assert result is True
    assert not path.exists()
    assert (tmp_path / "traces.1.jsonl").exists()


@pytest.mark.asyncio
async def test_run_maintenance_isolates_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    calls = []
    original_prune = prune_snapshots

    def failing_prune(pid, max_age_days=30):
        calls.append("prune_snapshots")
        raise PermissionError("denied")

    def ok_prune(pid, max_age_days=30):
        calls.append("prune_completed_checkpoints")
        return original_prune(pid, max_age_days)

    monkeypatch.setattr("hubspot_agent.maintenance.prune_snapshots", failing_prune)
    monkeypatch.setattr("hubspot_agent.maintenance.prune_completed_checkpoints", ok_prune)

    report = await run_maintenance("123")
    assert report.snapshots.pruned_count == 0
    assert report.checkpoints.pruned_count == 0
    assert report.rotated is False
    assert "prune_snapshots" in calls
    assert "prune_completed_checkpoints" in calls


@pytest.mark.asyncio
async def test_initialize_session_runs_maintenance(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    snapshot_dir = tmp_path / ".claude" / "hubspot" / "123" / "undo_snapshots"
    snapshot_dir.mkdir(parents=True)
    old_snapshot = snapshot_dir / "old.json"
    old_snapshot.write_text('{}')
    os.utime(old_snapshot, (time.time() - 31 * 86400, time.time() - 31 * 86400))

    from hubspot_agent.orchestrator import initialize_session

    await initialize_session("123")
    assert not old_snapshot.exists()


def test_prune_snapshots_with_invalid_portal_id():
    with pytest.raises(ValueError):
        prune_snapshots("../../../etc")


def test_rotate_jsonl_stat_oserror(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    path.write_text("x")
    monkeypatch.setattr("pathlib.Path.stat", lambda self: (_ for _ in ()).throw(OSError("stat failed")))
    assert rotate_jsonl(path, max_size_mb=1) is False


def test_prune_snapshots_unlink_oserror(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    snapshot_dir = tmp_path / ".claude" / "hubspot" / "123" / "undo_snapshots"
    snapshot_dir.mkdir(parents=True)

    old_file = snapshot_dir / "old.json"
    old_file.write_text('{}')
    monkeypatch.setattr(time, "time", lambda: 1000000000.0)
    os.utime(old_file, (1000000000.0 - 31 * 86400, 1000000000.0 - 31 * 86400))

    call_count = 0
    original_unlink = Path.unlink

    def failing_unlink(self, missing_ok=False):
        nonlocal call_count
        call_count += 1
        if self.name == "old.json":
            raise PermissionError("denied")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    report = prune_snapshots("123", max_age_days=30)
    assert report.pruned_count == 0
    assert call_count == 1
    assert old_file.exists()
