from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from hubspot_agent.models import RiskLevel


_RISK_ORDER = {
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.DESTRUCTIVE: 4,
}


class RoleConfig(BaseModel):
    user_id: str
    allowed_agents: list[str]
    max_risk_level: RiskLevel
    denied_tools: list[str]


class RoleManager:
    def __init__(self, roles: list[RoleConfig] | None = None) -> None:
        self._roles: dict[str, RoleConfig] = {}
        if roles:
            for role in roles:
                self._roles[role.user_id] = role

    @classmethod
    def load_roles(cls, portal_id: str) -> list[RoleConfig]:
        path = Path.home() / ".claude" / "hubspot" / portal_id / "roles.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return [RoleConfig(**entry) for entry in data]
            return []
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    @classmethod
    def for_portal(cls, portal_id: str) -> "RoleManager":
        return cls(cls.load_roles(portal_id))

    def can_dispatch(
        self,
        user_id: str | None,
        agent_name: str,
        risk_level: RiskLevel,
        tool_name: str | None = None,
    ) -> bool:
        if not self._roles or user_id is None:
            return True
        role = self._roles.get(user_id)
        if role is None:
            return True
        if agent_name not in role.allowed_agents:
            return False
        if _RISK_ORDER[risk_level] > _RISK_ORDER[role.max_risk_level]:
            return False
        if tool_name and tool_name in role.denied_tools:
            return False
        return True
