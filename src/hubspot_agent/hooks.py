from __future__ import annotations

import importlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from hubspot_agent.maintenance import _portal_dir

logger = logging.getLogger("hubspot_agent.hooks")


class HookEvent(str, Enum):
    PRE_WRITE = "pre_write"
    POST_WRITE = "post_write"
    PRE_APPROVAL = "pre_approval"
    POST_APPROVAL = "post_approval"


class HookContext(BaseModel):
    portal_id: str | None = None
    agent_name: str | None = None
    action_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    preview_result: dict[str, Any] | None = None
    user_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"extra": "allow"}


class HookResult(BaseModel):
    allowed: bool = True
    modified_payload: dict[str, Any] | None = None
    message: str | None = None


Handler = Callable[[HookContext], Awaitable[HookResult]]


class HookRegistry:
    def __init__(self) -> None:
        self._handlers: dict[HookEvent, list[Handler]] = {
            event: [] for event in HookEvent
        }

    def register(self, event: HookEvent, handler: Handler) -> None:
        self._handlers[event].append(handler)

    async def run(self, event: HookEvent, context: HookContext) -> HookResult:
        handlers = self._handlers.get(event, [])
        current_payload = dict(context.payload) if context.payload else {}
        last_message: str | None = None
        payload_modified = False

        for handler in handlers:
            try:
                result = await handler(context)
            except Exception:
                logger.exception("Hook handler failed for event=%s", event.value)
                continue

            if result.message is not None:
                last_message = result.message

            if not result.allowed:
                return HookResult(
                    allowed=False,
                    modified_payload=result.modified_payload,
                    message=last_message,
                )

            if result.modified_payload is not None:
                current_payload = dict(result.modified_payload)
                context = context.model_copy(update={"payload": current_payload})
                payload_modified = True

        return HookResult(
            allowed=True,
            modified_payload=current_payload if payload_modified else None,
            message=last_message,
        )


_DEFAULT_REGISTRY = HookRegistry()


def get_registry() -> HookRegistry:
    return _DEFAULT_REGISTRY


def reset_registry() -> None:
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = HookRegistry()


def _hooks_path(portal_id: str) -> Path:
    return _portal_dir(portal_id) / "hooks.json"


def load_hooks_config(portal_id: str) -> dict[str, Any]:
    path = _hooks_path(portal_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_handler(module_path: str, func_name: str) -> Handler | None:
    if not module_path.startswith("hubspot_agent."):
        logger.warning(
            "Refusing to load hook handler from non-hubspot_agent module: %s", module_path
        )
        return None
    try:
        module = importlib.import_module(module_path)
        handler = getattr(module, func_name, None)
        if callable(handler):
            return handler
    except Exception:
        logger.exception(
            "Failed to import hook handler %s.%s", module_path, func_name
        )
    return None


def register_hooks_from_config(
    portal_id: str, registry: HookRegistry | None = None
) -> None:
    registry = registry or get_registry()
    config = load_hooks_config(portal_id)
    for event_name, entries in config.items():
        try:
            event = HookEvent(event_name)
        except ValueError:
            logger.warning("Invalid hook event in config: %s", event_name)
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            module = entry.get("module")
            func = entry.get("func")
            if not module or not func:
                continue
            handler = _resolve_handler(module, func)
            if handler is not None:
                registry.register(event, handler)


async def run_hooks(
    event: HookEvent,
    *,
    portal_id: str | None = None,
    agent_name: str | None = None,
    action_id: str | None = None,
    payload: dict[str, Any] | None = None,
    preview_result: dict[str, Any] | None = None,
    user_id: str | None = None,
    registry: HookRegistry | None = None,
) -> HookResult:
    registry = registry or get_registry()
    context = HookContext(
        portal_id=portal_id,
        agent_name=agent_name,
        action_id=action_id,
        payload=payload or {},
        preview_result=preview_result,
        user_id=user_id,
        timestamp=datetime.now(timezone.utc),
    )
    return await registry.run(event, context)
