from __future__ import annotations

import hubspot_agent.tools.marketing  # noqa: F401 — registers tools
from hubspot_agent.agents._base import AgentPrompt, build_agent_prompt
from hubspot_agent.config import PortalConfig
from hubspot_agent.tools import get_tool

_TOOL_NAMES = [
    "hubspot_create_email",
    "hubspot_get_email",
    "hubspot_create_campaign",
    "hubspot_list_campaigns",
    "hubspot_create_segment",
    "hubspot_get_segment",
    "hubspot_create_ab_test",
    "hubspot_get_ab_test",
    "hubspot_create_suppression_list",
    "hubspot_list_suppression_lists",
    "hubspot_send_email",
    "hubspot_get_email_performance",
]

_DOMAIN = (
    "You manage HubSpot marketing campaigns, emails, segments, A/B tests, and suppression lists. "
    "You create, retrieve, and send marketing emails; manage campaigns; build segments; "
    "configure A/B tests; and handle suppression lists."
)


def get_marketing_agent_prompt(portal_config: PortalConfig | None = None) -> AgentPrompt:
    tools = [t for name in _TOOL_NAMES if (t := get_tool(name)) is not None]
    return build_agent_prompt(
        agent_name="Marketing Agent",
        domain_description=_DOMAIN,
        available_tools=tools,
        portal_config=portal_config,
    )
