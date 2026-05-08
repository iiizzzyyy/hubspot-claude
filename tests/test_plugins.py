import logging
from pathlib import Path

import pytest

from hubspot_agent.plugins import (
    PluginLoader,
    PluginModule,
    augment_agent_prompt,
)
from hubspot_agent.tools import ToolDef, registry


def test_plugin_module_model():
    pm = PluginModule(name="test", tools=[], augmentations={})
    assert pm.name == "test"


def test_plugin_loader_extracts_tools_and_augmentations(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    plugin_file = plugin_dir / "my_plugin.py"
    plugin_file.write_text(
        "from hubspot_agent.tools import tool\n"
        "\n"
        "@tool(name='plugin_test_tool', description='A plugin test tool')\n"
        "async def plugin_test_tool(x: int) -> dict:\n"
        "    return {'result': x * 2}\n"
        "\n"
        "AGENT_AUGMENTATIONS = {\n"
        "    'objects': ['- You can also use plugin_test_tool for custom logic.'],\n"
        "}\n"
    )

    loader = PluginLoader()
    plugins = loader.load_plugins(plugin_dir)

    assert len(plugins) == 1
    assert plugins[0].name == "my_plugin"
    assert len(plugins[0].tools) == 1
    assert plugins[0].tools[0].name == "plugin_test_tool"
    assert plugins[0].augmentations == {
        "objects": ["- You can also use plugin_test_tool for custom logic."],
    }

    # Tool was removed from global registry by PluginLoader
    assert "plugin_test_tool" not in registry

    # Registering it back works
    loader.register_tools(registry)
    assert "plugin_test_tool" in registry
    assert registry["plugin_test_tool"].description == "A plugin test tool"

    # Clean up global registry
    registry.pop("plugin_test_tool", None)


def test_plugin_load_failure_is_non_fatal(tmp_path, caplog):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    bad_plugin = plugin_dir / "bad_plugin.py"
    bad_plugin.write_text("this is not valid python!!\n")

    good_plugin = plugin_dir / "good_plugin.py"
    good_plugin.write_text(
        "from hubspot_agent.tools import tool\n"
        "\n"
        "@tool(name='good_tool', description='A good tool')\n"
        "def good_tool() -> dict:\n"
        "    return {'ok': True}\n"
        "\n"
        "AGENT_AUGMENTATIONS = {}\n"
    )

    with caplog.at_level(logging.ERROR, logger="hubspot_agent.plugins"):
        loader = PluginLoader()
        plugins = loader.load_plugins(plugin_dir)

    assert len(plugins) == 1
    assert plugins[0].name == "good_plugin"
    assert "Failed to load plugin: bad_plugin.py" in caplog.text

    # Ensure bad plugin did not pollute global registry
    assert "good_tool" not in registry

    loader.register_tools(registry)
    assert "good_tool" in registry
    registry.pop("good_tool", None)


def test_register_tools_injects_into_registry():
    loader = PluginLoader()
    loader._plugins = [
        PluginModule(
            name="p1",
            tools=[
                ToolDef(
                    name="t1",
                    description="d1",
                    func=lambda: None,
                    is_async=False,
                )
            ],
            augmentations={},
        )
    ]
    test_registry: dict[str, ToolDef] = {}
    loader.register_tools(test_registry)
    assert "t1" in test_registry
    assert test_registry["t1"].description == "d1"


def test_augment_agent_prompt_appends_snippets():
    plugins = [
        PluginModule(
            name="p1",
            tools=[],
            augmentations={
                "objects": ["- Extra instruction 1.", "- Extra instruction 2."],
            },
        ),
        PluginModule(
            name="p2",
            tools=[],
            augmentations={
                "objects": ["- Extra instruction 3."],
                "workflows": ["- Workflow extra."],
            },
        ),
    ]

    base = "You are the Objects Agent."
    augmented = augment_agent_prompt("objects", base, plugins)
    assert "You are the Objects Agent." in augmented
    assert "## Plugin augmentations" in augmented
    assert "- Extra instruction 1." in augmented
    assert "- Extra instruction 2." in augmented
    assert "- Extra instruction 3." in augmented
    assert "- Workflow extra." not in augmented

    # Agent with no augmentations stays unchanged
    unchanged = augment_agent_prompt("lists", "You are the Lists Agent.", plugins)
    assert unchanged == "You are the Lists Agent."


def test_orchestrator_augmented_prompt(tmp_path, monkeypatch):
    import hubspot_agent.orchestrator as orch

    # Reset plugin state so _ensure_plugins re-runs
    orch._PLUGINS_INITIALIZED = False
    orch._PLUGIN_LOADER = None

    # Set up fake plugin dir under tmp_path
    plugin_dir = tmp_path / ".claude" / "hubspot" / "plugins"
    plugin_dir.mkdir(parents=True)

    plugin_file = plugin_dir / "test_plugin.py"
    plugin_file.write_text(
        "from hubspot_agent.tools import tool\n"
        "\n"
        "@tool(name='orch_test_tool', description='An orchestrator test tool')\n"
        "def orch_test_tool() -> dict:\n"
        "    return {'ok': True}\n"
        "\n"
        "AGENT_AUGMENTATIONS = {\n"
        "    'objects': ['- Plugin says hello from objects agent.'],\n"
        "}\n"
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = orch.dispatch_agent("objects", "find contacts")
    assert result.status == "preview"
    assert "Plugin says hello from objects agent." in result.data["system_prompt"]
    assert "Plugin says hello from objects agent." in result.data["full_prompt"]

    # Clean up the injected tool from global registry
    registry.pop("orch_test_tool", None)
