"""Regression: the daemon subprocess import graph populates the tool registry.

The warm-client daemon runs as ``python -m hubspot_agent.daemon`` — a fresh
process that imports ``hubspot_agent.daemon`` → ``handlers`` →
``hubspot_agent.tools`` but never ``agents/*``.  Before ``tools/__init__.py``
self-imported its submodules, the ``@tool`` decorators in the tool submodules
never ran in that process, so the registry was empty and every daemon ``tool``
call returned ``Unknown tool`` (observed on ``hubspot_search_objects`` and
``hubspot_batch_upsert_objects``).

``test_daemon_rpc_tool_roundtrip`` could not catch this: it constructs
``HubSpotDaemon`` in the test process (an asyncio task), sharing the
already-populated registry.  This test reproduces the daemon's *subprocess*
import graph in a clean interpreter and asserts the registry is populated.
"""
from __future__ import annotations

import os
import subprocess
import sys

_PROBE = (
    "import hubspot_agent.daemon  # mirrors `python -m hubspot_agent.daemon`\n"
    "from hubspot_agent.tools import registry\n"
    "names = set(registry)\n"
    "assert 'hubspot_search_objects' in names, sorted(names)[:20]\n"
    "assert 'hubspot_batch_upsert_objects' in names, sorted(names)[:20]\n"
    "print(len(registry))\n"
)


def test_daemon_import_graph_populates_registry(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "CLAUDE_PLUGIN_DATA": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    count = int(result.stdout.strip())
    assert count >= 75, f"expected >=75 registered tools, got {count}"