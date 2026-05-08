import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hubspot_agent.anomaly import (
    AnomalyCheckResult,
    AnomalyDetector,
    PortalBaselines,
    ToolBaseline,
)
from hubspot_agent.trace import emit_trace, new_trace_id


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_tool_baseline_defaults():
    baseline = ToolBaseline(
        tool_name="objects",
        median_duration_ms=100.0,
        failure_rate=0.05,
        sample_size=10,
    )
    assert baseline.duration_std == 0.0
    assert baseline.failure_rate_std == 0.0


def test_anomaly_check_result():
    result = AnomalyCheckResult(paused=False, deviation_sigma=0.0, reason=None)
    assert result.paused is False
    assert result.deviation_sigma == 0.0
    assert result.reason is None


def test_portal_baselines_defaults():
    portal = PortalBaselines()
    assert portal.sigma_threshold == 3.0
    assert portal.computed_at is None
    assert portal.tools == {}


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------


def test_compute_baselines_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    detector = AnomalyDetector()
    baselines = detector.compute_baselines("123", window_hours=168)
    assert baselines == {}
    path = tmp_path / ".claude" / "hubspot" / "123" / "baselines.json"
    assert path.exists()
    raw = __import__("json").loads(path.read_text())
    assert raw["sigma_threshold"] == 3.0
    assert raw["tools"] == {}


def test_compute_baselines_from_traces(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for _ in range(5):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {"request": "find contacts"})
        emit_trace("123", "tool_call", tid, {"agent": "objects", "mode": "preview"})
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector()
    baselines = detector.compute_baselines("123", window_hours=168)

    assert "objects" in baselines
    baseline = baselines["objects"]
    assert baseline.tool_name == "objects"
    assert baseline.sample_size == 5
    assert baseline.failure_rate == 0.0
    assert baseline.median_duration_ms >= 0.0


def test_compute_baselines_includes_failure_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for i in range(10):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {"request": "find contacts"})
        emit_trace("123", "tool_call", tid, {"agent": "objects", "mode": "preview"})
        if i == 7:
            emit_trace("123", "error", tid, {"message": "timeout"})
        else:
            emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector()
    baselines = detector.compute_baselines("123", window_hours=168)

    assert baselines["objects"].sample_size == 10
    assert baselines["objects"].failure_rate == 0.1


def test_compute_baselines_per_tool_name(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for _ in range(3):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"tool_name": "search"})
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector()
    baselines = detector.compute_baselines("123", window_hours=168)
    assert "search" in baselines
    assert baselines["search"].sample_size == 3


# ---------------------------------------------------------------------------
# check_request
# ---------------------------------------------------------------------------


def test_check_request_no_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    detector = AnomalyDetector()
    result = detector.check_request("123", "objects", "objects")
    assert result.paused is False
    assert result.deviation_sigma == 0.0
    assert result.reason is None


def test_check_request_insufficient_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for _ in range(3):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector()
    detector.compute_baselines("123", window_hours=168)

    result = detector.check_request("123", "objects", "objects")
    assert result.paused is False
    assert result.reason is None


def test_check_request_normal_behavior(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for _ in range(10):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector()
    detector.compute_baselines("123", window_hours=168)

    result = detector.check_request("123", "objects", "objects")
    assert result.paused is False


def test_check_request_duration_anomaly(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for _ in range(10):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector(recent_window_seconds=1)
    detector.compute_baselines("123", window_hours=168)

    time.sleep(1.1)

    for _ in range(3):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        time.sleep(0.15)
        emit_trace("123", "completion", tid, {})

    result = detector.check_request("123", "objects", "objects")
    assert result.paused is True
    assert result.deviation_sigma > 3.0
    assert "duration" in result.reason.lower()


def test_check_request_failure_rate_anomaly(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for _ in range(10):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector(recent_window_seconds=1)
    detector.compute_baselines("123", window_hours=168)

    time.sleep(1.1)

    for _ in range(3):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "error", tid, {"message": "timeout"})

    result = detector.check_request("123", "objects", "objects")
    assert result.paused is True
    assert result.deviation_sigma > 3.0
    assert "failure rate" in result.reason.lower()


def test_check_request_tool_name_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for _ in range(10):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector()
    detector.compute_baselines("123", window_hours=168)

    result = detector.check_request("123", "objects", "unknown_tool")
    assert result.paused is False
    assert result.reason is None


def test_check_request_both_checks_trigger(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for _ in range(10):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector(recent_window_seconds=1)
    detector.compute_baselines("123", window_hours=168)

    time.sleep(1.1)

    for _ in range(3):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        time.sleep(0.15)
        emit_trace("123", "error", tid, {"message": "timeout"})

    result = detector.check_request("123", "objects", "objects")
    assert result.paused is True
    assert "duration" in result.reason.lower()
    assert "failure rate" in result.reason.lower()


# ---------------------------------------------------------------------------
# Configurable threshold
# ---------------------------------------------------------------------------


def test_set_sigma_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    detector = AnomalyDetector()
    detector.set_sigma_threshold("123", 5.0)

    portal = detector._load_portal_baselines("123")
    assert portal.sigma_threshold == 5.0


def test_custom_threshold_prevents_pause(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for _ in range(10):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector(default_sigma_threshold=200.0, recent_window_seconds=1)
    detector.compute_baselines("123", window_hours=168)

    time.sleep(1.1)

    for _ in range(3):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        time.sleep(0.15)
        emit_trace("123", "completion", tid, {})

    result = detector.check_request("123", "objects", "objects")
    assert result.paused is False


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


def test_dispatch_agent_paused_on_anomaly(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    from hubspot_agent.config import PortalConfig, save_portal_config
    from hubspot_agent.orchestrator import dispatch_agent

    save_portal_config(PortalConfig(portal_id="123", token="t"))
    config = PortalConfig(portal_id="123", token="t")

    for _ in range(10):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector(recent_window_seconds=1)
    detector.compute_baselines("123", window_hours=168)

    time.sleep(1.1)

    for _ in range(3):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "error", tid, {"message": "boom"})

    result = dispatch_agent("objects", "find contacts", portal_config=config, mode="preview")
    assert result.status == "error"
    assert "Anomaly detected" in result.error_message
    assert result.data.get("paused") is True
    assert result.data.get("deviation_sigma") > 3.0


def test_dispatch_agent_allows_normal(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    from hubspot_agent.config import PortalConfig, save_portal_config
    from hubspot_agent.orchestrator import dispatch_agent

    save_portal_config(PortalConfig(portal_id="456", token="t"))
    config = PortalConfig(portal_id="456", token="t")

    for _ in range(10):
        tid = new_trace_id()
        emit_trace("456", "request_received", tid, {})
        emit_trace("456", "tool_call", tid, {"agent": "objects"})
        emit_trace("456", "completion", tid, {})

    detector = AnomalyDetector()
    detector.compute_baselines("456", window_hours=168)

    result = dispatch_agent("objects", "find contacts", portal_config=config, mode="preview")
    assert result.status == "preview"
    assert "Anomaly detected" not in (result.error_message or "")


def test_anomaly_detector_does_not_block_on_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hubspot_agent.config.CONFIG_DIR", tmp_path)

    from hubspot_agent.config import PortalConfig, save_portal_config
    from hubspot_agent.orchestrator import dispatch_agent

    save_portal_config(PortalConfig(portal_id="789", token="t"))
    config = PortalConfig(portal_id="789", token="t")

    base_dir = tmp_path / ".claude" / "hubspot" / "789"
    base_dir.mkdir(parents=True)
    (base_dir / "baselines.json").write_text("not json")

    result = dispatch_agent("objects", "find contacts", portal_config=config, mode="preview")
    assert result.status == "preview"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_compute_baselines_std_calculation(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for i in range(20):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        time.sleep(0.01 * (i % 3))
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector()
    baselines = detector.compute_baselines("123", window_hours=168)
    baseline = baselines["objects"]
    assert baseline.duration_std > 0.0
    assert baseline.failure_rate_std == 0.0


def test_load_corrupt_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    base_dir = tmp_path / ".claude" / "hubspot" / "123"
    base_dir.mkdir(parents=True)
    (base_dir / "baselines.json").write_text("not json")

    detector = AnomalyDetector()
    portal = detector._load_portal_baselines("123")
    assert portal.sigma_threshold == 3.0
    assert portal.tools == {}


def test_check_request_duration_no_upward_deviation(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    for _ in range(10):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        time.sleep(0.05)
        emit_trace("123", "completion", tid, {})

    detector = AnomalyDetector()
    detector.compute_baselines("123", window_hours=168)

    for _ in range(3):
        tid = new_trace_id()
        emit_trace("123", "request_received", tid, {})
        emit_trace("123", "tool_call", tid, {"agent": "objects"})
        emit_trace("123", "completion", tid, {})

    result = detector.check_request("123", "objects", "objects")
    assert result.paused is False
