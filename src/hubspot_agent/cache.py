from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from hubspot_agent.client import HubSpotClient
from hubspot_agent.config import PortalConfig

WARM_DOMAINS = ["contacts", "companies", "deals", "tickets"]


class SchemaCache:
    TTL_SECONDS = 3600  # 1 hour

    def __init__(self, portal_id: str, base_dir: Path | None = None) -> None:
        self.portal_id = portal_id
        self.base_dir = base_dir or (Path.home() / ".claude" / "hubspot" / portal_id)
        self.cache_file = self.base_dir / "schema_cache.json"
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                self._data = json.loads(self.cache_file.read_text())
            except json.JSONDecodeError:
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self._data, indent=2))

    def get(self, domain: str) -> dict[str, Any] | None:
        entry = self._data.get(domain)
        if entry is None:
            return None
        ts = entry.get("_timestamp", 0)
        if time.time() - ts > self.TTL_SECONDS:
            return None
        return entry.get("data")

    def set(self, domain: str, data: dict[str, Any]) -> None:
        self._data[domain] = {"_timestamp": time.time(), "data": data}
        self._save()

    def invalidate(self, domain: str) -> None:
        if domain in self._data:
            del self._data[domain]
            self._save()

    def refresh_all(self) -> None:
        self._data = {}
        if self.cache_file.exists():
            self.cache_file.unlink()

    def refresh_domain(self, domain: str) -> None:
        self.invalidate(domain)


async def warm_standard_schemas(portal_config: PortalConfig) -> SchemaCache:
    cache = SchemaCache(portal_config.portal_id)
    client = HubSpotClient(portal_config)
    try:

        async def _fetch(domain: str) -> tuple[str, dict[str, Any] | None]:
            try:
                resp = await client.get(
                    f"/crm/v3/properties/{domain}",
                    portal_id=portal_config.portal_id,
                )
                return domain, resp.body
            except Exception:
                return domain, None

        results = await asyncio.gather(*(_fetch(d) for d in WARM_DOMAINS))
        for domain, data in results:
            if data is not None:
                cache.set(domain, data)
    finally:
        await client.close()
    return cache
