from __future__ import annotations

import hubspot_agent.tools.service  # noqa: F401 — registers tools
from hubspot_agent.agents._base import AgentPrompt, build_agent_prompt
from hubspot_agent.config import PortalConfig
from hubspot_agent.tools import get_tool

_TOOL_NAMES = [
    "hubspot_get_knowledge_base_article",
    "hubspot_list_kb_articles",
    "hubspot_get_ticket_pipeline",
    "hubspot_create_ticket_pipeline",
    "hubspot_list_service_automation",
    "hubspot_get_feedback_survey",
]

_DOMAIN = (
    "You manage HubSpot Service Hub resources. "
    "You retrieve knowledge base articles, ticket pipelines, service automation rules, and customer feedback surveys. "
    "You create ticket pipelines with proper stage definitions."
)


def get_service_agent_prompt(portal_config: PortalConfig | None = None) -> AgentPrompt:
    tools = [t for name in _TOOL_NAMES if (t := get_tool(name)) is not None]
    return build_agent_prompt(
        agent_name="Service Agent",
        domain_description=_DOMAIN,
        available_tools=tools,
        portal_config=portal_config,
    )
