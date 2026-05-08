from __future__ import annotations

import hubspot_agent.tools.cms  # noqa: F401 — registers tools
from hubspot_agent.agents._base import AgentPrompt, build_agent_prompt
from hubspot_agent.config import PortalConfig
from hubspot_agent.tools import get_tool

_TOOL_NAMES = [
    "hubspot_get_page",
    "hubspot_update_page",
    "hubspot_list_files",
    "hubspot_upload_file",
    "hubspot_publish_social_post",
]

_DOMAIN = (
    "You manage HubSpot CMS content, file manager assets, and social media publishing. "
    "You retrieve and update CMS pages (site pages and blog posts), "
    "list and upload files to the file manager, and publish social media posts."
)


def get_cms_agent_prompt(portal_config: PortalConfig | None = None) -> AgentPrompt:
    tools = [t for name in _TOOL_NAMES if (t := get_tool(name)) is not None]
    return build_agent_prompt(
        agent_name="CMS Agent",
        domain_description=_DOMAIN,
        available_tools=tools,
        portal_config=portal_config,
    )
