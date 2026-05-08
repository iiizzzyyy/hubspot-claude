import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hubspot_agent.memory import (
    SessionMemory,
    SessionSummary,
    _sessions_dir,
    _summary_file,
    generate_summary,
)


def test_session_summary_model():
    now = datetime.now(timezone.utc)
    summary = SessionSummary(
        session_id="sess-1",
        portal_id="123",
        started_at=now,
        ended_at=now,
        summary="Test summary",
        active_agents=["objects", "properties"],
        pending_approvals=2,
        custom_objects_discovered=["custom_a"],
    )
    assert summary.session_id == "sess-1"
    assert summary.portal_id == "123"
    assert summary.started_at == now
    assert summary.summary == "Test summary"
    assert summary.active_agents == ["objects", "properties"]
    assert summary.pending_approvals == 2
    assert summary.custom_objects_discovered == ["custom_a"]


def test_session_summary_json_roundtrip():
    now = datetime.now(timezone.utc)
    summary = SessionSummary(
        session_id="sess-1",
        portal_id="123",
        started_at=now,
        ended_at=now,
        summary="Test summary",
        active_agents=["objects"],
        pending_approvals=0,
        custom_objects_discovered=[],
    )
    raw = summary.model_dump_json()
    data = json.loads(raw)
    restored = SessionSummary(**data)
    assert restored.session_id == summary.session_id
    assert restored.portal_id == summary.portal_id
    assert restored.started_at.isoformat() == summary.started_at.isoformat()
    assert restored.summary == summary.summary


def test_save_summary_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    now = datetime.now(timezone.utc)
    summary = SessionSummary(
        session_id="abc",
        portal_id="456",
        started_at=now,
        ended_at=now,
        summary="summary text",
    )
    SessionMemory.save_summary("456", "abc", summary)

    expected = tmp_path / ".claude" / "hubspot" / "456" / "sessions" / "abc.json"
    assert expected.exists()
    data = json.loads(expected.read_text())
    assert data["session_id"] == "abc"
    assert data["portal_id"] == "456"
    assert data["summary"] == "summary text"


def test_load_last_summary_returns_most_recent(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    now = datetime.now(timezone.utc)
    old = SessionSummary(session_id="old", portal_id="789", started_at=now, ended_at=now, summary="old")
    new = SessionSummary(session_id="new", portal_id="789", started_at=now, ended_at=now, summary="new")

    SessionMemory.save_summary("789", "old", old)
    SessionMemory.save_summary("789", "new", new)

    # Ensure mtime ordering: touch old first, then new
    old_path = _summary_file("789", "old")
    new_path = _summary_file("789", "new")
    os.utime(old_path, (time.time() - 10, time.time() - 10))
    os.utime(new_path, (time.time(), time.time()))

    result = SessionMemory.load_last_summary("789")
    assert result is not None
    assert result.session_id == "new"


def test_load_last_summary_missing_dir_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    result = SessionMemory.load_last_summary("999")
    assert result is None


def test_load_last_summary_empty_dir_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    sessions_dir = _sessions_dir("999")
    sessions_dir.mkdir(parents=True)
    result = SessionMemory.load_last_summary("999")
    assert result is None


def test_load_last_summary_skips_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    sessions_dir = _sessions_dir("111")
    sessions_dir.mkdir(parents=True)
    corrupt = sessions_dir / "bad.json"
    corrupt.write_text("not json")

    result = SessionMemory.load_last_summary("111")
    assert result is None


def test_list_sessions_orders_by_mtime(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    now = datetime.now(timezone.utc)
    s1 = SessionSummary(session_id="first", portal_id="222", started_at=now, ended_at=now, summary="1")
    s2 = SessionSummary(session_id="second", portal_id="222", started_at=now, ended_at=now, summary="2")
    s3 = SessionSummary(session_id="third", portal_id="222", started_at=now, ended_at=now, summary="3")

    SessionMemory.save_summary("222", "first", s1)
    SessionMemory.save_summary("222", "second", s2)
    SessionMemory.save_summary("222", "third", s3)

    os.utime(_summary_file("222", "first"), (time.time() - 30, time.time() - 30))
    os.utime(_summary_file("222", "second"), (time.time() - 20, time.time() - 20))
    os.utime(_summary_file("222", "third"), (time.time() - 10, time.time() - 10))

    results = SessionMemory.list_sessions("222")
    assert [r.session_id for r in results] == ["third", "second", "first"]


def test_list_sessions_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    now = datetime.now(timezone.utc)
    for i in range(5):
        s = SessionSummary(session_id=f"s{i}", portal_id="333", started_at=now, ended_at=now, summary=str(i))
        SessionMemory.save_summary("333", f"s{i}", s)
        os.utime(_summary_file("333", f"s{i}"), (time.time() - i, time.time() - i))

    results = SessionMemory.list_sessions("333", limit=2)
    assert len(results) == 2
    assert results[0].session_id == "s0"


def test_list_sessions_skips_corrupt_files(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    sessions_dir = _sessions_dir("444")
    sessions_dir.mkdir(parents=True)
    good = sessions_dir / "good.json"
    now = datetime.now(timezone.utc)
    good.write_text(
        json.dumps(
            {
                "session_id": "good",
                "portal_id": "444",
                "started_at": now.isoformat(),
                "ended_at": now.isoformat(),
                "summary": "ok",
                "active_agents": [],
                "pending_approvals": 0,
                "custom_objects_discovered": [],
            }
        )
    )
    bad = sessions_dir / "bad.json"
    bad.write_text("broken")

    results = SessionMemory.list_sessions("444")
    assert len(results) == 1
    assert results[0].session_id == "good"


def test_list_sessions_missing_dir_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    assert SessionMemory.list_sessions("555") == []


def test_generate_summary_placeholder():
    now = datetime.now(timezone.utc)
    text = generate_summary(
        session_id="x",
        portal_id="123",
        started_at=now,
        active_agents=["objects", "lists"],
        pending_approvals=1,
        custom_objects_discovered=["custom_obj"],
    )
    assert "Session x for portal 123" in text
    assert "objects" in text
    assert "lists" in text
    assert "Pending approvals at close: 1" in text
    assert "custom_obj" in text


def test_generate_summary_no_agents():
    now = datetime.now(timezone.utc)
    text = generate_summary("y", "123", now, [], 0, [])
    assert "Agents used: none" in text


@pytest.mark.asyncio
async def test_initialize_session_loads_last_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    now = datetime.now(timezone.utc)
    summary = SessionSummary(session_id="prev", portal_id="777", started_at=now, ended_at=now, summary="previous")
    SessionMemory.save_summary("777", "prev", summary)

    from hubspot_agent.orchestrator import initialize_session, get_session_context

    await initialize_session("777")
    ctx = get_session_context()
    assert ctx is not None
    assert ctx.session_id == "prev"
    assert ctx.summary == "previous"


@pytest.mark.asyncio
async def test_initialize_session_no_prior_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    from hubspot_agent.orchestrator import initialize_session, get_session_context

    await initialize_session("888")
    ctx = get_session_context()
    assert ctx is None


def test_end_session_persists_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    from hubspot_agent.orchestrator import end_session, get_session_context

    started = datetime.now(timezone.utc)
    end_session(
        portal_id="999",
        session_id="sess-99",
        started_at=started,
        active_agents=["objects"],
        pending_approvals=0,
        custom_objects_discovered=["c1"],
    )

    result = SessionMemory.load_last_summary("999")
    assert result is not None
    assert result.session_id == "sess-99"
    assert result.portal_id == "999"
    assert result.active_agents == ["objects"]
    assert result.pending_approvals == 0
    assert result.custom_objects_discovered == ["c1"]
    assert result.ended_at >= started


def test_end_session_without_custom_objects(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hubspot_agent import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".claude" / "hubspot")

    from hubspot_agent.orchestrator import end_session

    started = datetime.now(timezone.utc)
    end_session(
        portal_id="999",
        session_id="sess-100",
        started_at=started,
        active_agents=["properties"],
    )

    result = SessionMemory.load_last_summary("999")
    assert result is not None
    assert result.custom_objects_discovered == []
    assert result.pending_approvals == 0
