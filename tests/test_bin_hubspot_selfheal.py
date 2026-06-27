"""Bug 1: bin/hubspot stale-venv-on-update self-heal.

A /plugin update swaps the cache (new pyproject.toml + new bin/hubspot) within
a session, but the SessionStart install hook doesn't re-run until next session
— so the venv would keep the old package.  ``bin/hubspot`` self-heals by
comparing the venv's installed ``hubspot-agent`` version against the bundled
``pyproject.toml`` and reinstalling on drift, before exec'ing the router.

These tests exercise the shell ``bin/hubspot`` with a fake venv (no real Python
install) and ``HUBSPOT_NO_EXEC=1`` so the script stops short of exec'ing the
router.  ``grep``/``sed``/``head`` come from the real PATH (macOS/Linux core
utils the self-heal relies on).
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / "hubspot"


def _chmod_x(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _make_fake_venv(venv: Path, installed_ver: str, pip_marker: Path) -> None:
    """Stand up a fake venv whose python reports ``installed_ver`` and whose
    pip records its argv to ``pip_marker``."""
    (venv / "bin").mkdir(parents=True, exist_ok=True)

    py = venv / "bin" / "python"
    py.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ]; then printf "%s" "' + installed_ver + '"; exit 0; fi\n'
        "exit 0\n"
    )
    _chmod_x(py)

    pip = venv / "bin" / "pip"
    pip.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" > "{pip_marker}"\n'
        "exit 0\n"
    )
    _chmod_x(pip)


def _run(plugin_root: Path, data: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "CLAUDE_PLUGIN_ROOT": str(plugin_root),
        "CLAUDE_PLUGIN_DATA": str(data),
        "HOME": str(data),
        "HUBSPOT_NO_EXEC": "1",
        # Keep the real PATH so grep/sed/head are available to the script.
    }
    return subprocess.run(["/bin/sh", str(BIN), "status"], env=env, capture_output=True, text=True)


def _write_pyproject(plugin_root: Path, version: str) -> None:
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "pyproject.toml").write_text(
        f'[project]\nname = "hubspot-agent"\nversion = "{version}"\n'
    )


def test_selfheal_reinstalls_on_version_drift(tmp_path):
    plugin_root = tmp_path / "plugin"
    _write_pyproject(plugin_root, "0.1.2")

    data = tmp_path / "data"
    venv = data / "venv"
    pip_marker = data / "pip.args"
    _make_fake_venv(venv, installed_ver="0.1.1", pip_marker=pip_marker)
    (data / "venv.path").write_text(str(venv) + "\n")

    r = _run(plugin_root, data)
    assert r.returncode == 0, r.stderr
    assert pip_marker.exists(), "self-heal must invoke pip on version drift"
    args = pip_marker.read_text()
    assert "install" in args
    assert str(plugin_root) in args


def test_selfheal_noop_when_versions_match(tmp_path):
    plugin_root = tmp_path / "plugin"
    _write_pyproject(plugin_root, "0.1.2")

    data = tmp_path / "data"
    venv = data / "venv"
    pip_marker = data / "pip.args"
    _make_fake_venv(venv, installed_ver="0.1.2", pip_marker=pip_marker)
    (data / "venv.path").write_text(str(venv) + "\n")

    r = _run(plugin_root, data)
    assert r.returncode == 0, r.stderr
    assert not pip_marker.exists(), "pip must not run when versions already match"


def test_selfheal_no_crash_without_plugin_root(tmp_path):
    # No CLAUDE_PLUGIN_ROOT → no version to compare → skip self-heal, no crash.
    data = tmp_path / "data"
    venv = data / "venv"
    pip_marker = data / "pip.args"
    _make_fake_venv(venv, installed_ver="0.1.2", pip_marker=pip_marker)
    (data / "venv.path").write_text(str(venv) + "\n")

    env = {
        **os.environ,
        "CLAUDE_PLUGIN_DATA": str(data),
        "HOME": str(data),
        "HUBSPOT_NO_EXEC": "1",
    }
    env.pop("CLAUDE_PLUGIN_ROOT", None)
    r = subprocess.run(["/bin/sh", str(BIN), "status"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not pip_marker.exists()