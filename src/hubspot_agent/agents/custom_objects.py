from __future__ import annotations

import hubspot_agent.tools.objects  # noqa: F401 — registers tools
from hubspot_agent.agents._base import AgentPrompt, build_agent_prompt
from hubspot_agent.cache import SchemaCache
from hubspot_agent.config import PortalConfig
from hubspot_agent.tools import get_tool

_TOOL_NAMES = [
    "hubspot_get_object",
    "hubspot_search_objects",
    "hubspot_create_object",
    "hubspot_update_object",
    "hubspot_delete_object",
    "hubspot_batch_upsert_objects",
]

_DOMAIN = (
    "You manage custom object records in HubSpot. "
    "You retrieve, search, create, update, and delete custom object records "
    "by their object type ID. Always verify the custom object type exists "
    "before attempting operations."
)


def _build_domain(portal_config: PortalConfig | None = None) -> str:
    domain = _DOMAIN
    if portal_config is not None:
        try:
            cache = SchemaCache(portal_config.portal_id)
            custom = cache.list_custom_object_names()
            if custom:
                domain += f" Available custom types: {', '.join(custom)}."
        except Exception:
            pass
    return domain


def get_custom_objects_agent_prompt(portal_config: PortalConfig | None = None) -> AgentPrompt:
    tools = [t for name in _TOOL_NAMES if (t := get_tool(name)) is not None]
    return build_agent_prompt(
        agent_name="Custom Objects Agent",
        domain_description=_build_domain(portal_config),
        available_tools=tools,
        portal_config=portal_config,
    )
