from hubspot_agent.cli import hubspot_command


def test_hubspot_command_empty():
    result = hubspot_command("")
    assert "Usage" in result


def test_hubspot_command_no_portal(tmp_path):
    result = hubspot_command("find contacts", working_dir=str(tmp_path))
    assert "No default portal found" in result


def test_hubspot_command_routing(tmp_path, monkeypatch):
    from pathlib import Path
    portal_file = tmp_path / ".hubspot-portal"
    portal_file.write_text("123\n")

    # mock token via env
    import os
    monkeypatch.setenv("HUBSPOT_TOKEN_123", "test-token")

    result = hubspot_command("find contacts", working_dir=str(tmp_path))
    assert "Portal: 123" in result
    assert "objects" in result


def test_hubspot_command_ambiguous(tmp_path, monkeypatch):
    from pathlib import Path
    portal_file = tmp_path / ".hubspot-portal"
    portal_file.write_text("123\n")
    monkeypatch.setenv("HUBSPOT_TOKEN_123", "test-token")

    result = hubspot_command("hello world", working_dir=str(tmp_path))
    assert "not sure" in result.lower()


def test_hubspot_portal_switch():
    result = hubspot_command("portal switch 456")
    assert "Switched to portal 456" in result


def test_hubspot_refresh_no_portal(tmp_path):
    result = hubspot_command("refresh", working_dir=str(tmp_path))
    assert "No default portal found" in result
