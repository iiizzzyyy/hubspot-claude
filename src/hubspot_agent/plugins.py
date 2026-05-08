from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from hubspot_agent.tools import ToolDef, registry

logger = logging.getLogger(__name__)


class PluginModule(BaseModel):
    name: str
    tools: list[ToolDef]
    augmentations: dict[str, list[str]]


class PluginLoader:
    def __init__(self) -> None:
        self._plugins: list[PluginModule] = []

    def load_plugins(self, plugin_dir: str | Path) -> list[PluginModule]:
        """Scan *plugin_dir* for ``*.py`` files and load each in an isolated namespace.

        Plugin load failures are caught and logged without crashing the orchestrator.
        """
        plugins: list[PluginModule] = []
        path = Path(plugin_dir)
        if not path.exists():
            logger.debug("Plugin directory does not exist: %s", path)
            return plugins

        for file_path in sorted(path.glob("*.py")):
            try:
                plugin = self._load_single_plugin(file_path)
                if plugin is not None:
                    plugins.append(plugin)
            except Exception:
                logger.exception("Failed to load plugin: %s", file_path.name)

        self._plugins = plugins
        return plugins

    def register_tools(self, registry: dict[str, ToolDef]) -> None:
        """Inject loaded plugin tools into *registry*."""
        for plugin in self._plugins:
            for tool_def in plugin.tools:
                registry[tool_def.name] = tool_def

    def _load_single_plugin(self, file_path: Path) -> PluginModule | None:
        """Import a single ``.py`` file in an isolated module namespace.

        Security: plugins are executed with restricted builtins and a
        filtered ``__import__`` that only allows ``hubspot_agent`` packages.
        Path traversal and symlink escapes are rejected.
        """
        name = file_path.stem
        if not name.replace("_", "").isalnum():
            logger.warning("Invalid plugin name (alphanumeric+underscore only): %s", file_path)
            return None

        resolved = file_path.resolve()
        plugin_dir = file_path.parent.resolve()
        if not resolved.is_relative_to(plugin_dir):
            logger.warning("Plugin path escapes plugin directory: %s", file_path)
            return None

        spec = importlib.util.spec_from_file_location(name, resolved)
        if spec is None or spec.loader is None:
            logger.warning("Cannot create module spec for %s", file_path)
            return None

        before = set(registry.keys())

        module = importlib.util.module_from_spec(spec)
        module_name = f"hubspot_agent.plugins.{name}"

        # Restrict builtins to prevent malicious plugins from accessing
        # filesystem, subprocess, or dynamic code execution.
        safe_builtins = {
            k: v
            for k, v in __builtins__.items()
            if k not in ("open", "exec", "eval", "compile")
        }

        def _restricted_import(
            mod_name: str,
            globals_: Any = None,
            locals_: Any = None,
            fromlist: Any = None,
            level: int = 0,
        ) -> Any:
            if not mod_name.startswith("hubspot_agent"):
                raise ImportError(
                    f"Plugins may only import hubspot_agent packages, not {mod_name}"
                )
            return __builtins__["__import__"](mod_name, globals_, locals_, fromlist, level)

        safe_builtins["__import__"] = _restricted_import
        module.__dict__["__builtins__"] = safe_builtins

        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            after = set(registry.keys())
            for n in after - before:
                registry.pop(n, None)
            sys.modules.pop(module_name, None)
            raise
        finally:
            sys.modules.pop(module_name, None)

        after = set(registry.keys())
        new_names = sorted(after - before)
        new_tools = [registry.pop(n) for n in new_names]
        augmentations: dict[str, list[str]] = getattr(module, "AGENT_AUGMENTATIONS", {})

        return PluginModule(
            name=name,
            tools=new_tools,
            augmentations=augmentations,
        )


def augment_agent_prompt(
    agent_name: str,
    base_prompt: str,
    plugins: list[PluginModule],
) -> str:
    """Append plugin prompt snippets for *agent_name* to *base_prompt*."""
    snippets: list[str] = []
    for plugin in plugins:
        snippets.extend(plugin.augmentations.get(agent_name, []))
    if not snippets:
        return base_prompt
    return base_prompt + "\n\n## Plugin augmentations\n\n" + "\n".join(snippets)
