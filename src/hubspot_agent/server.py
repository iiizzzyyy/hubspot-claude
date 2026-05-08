from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from hubspot_agent.app_credentials import load_app_credentials
from hubspot_agent.config import detect_default_portal, load_portal_config
from hubspot_agent.webhooks import WebhookServer


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _resolve_portal_id(args_portal_id: str | None) -> str | None:
    if args_portal_id:
        return args_portal_id
    portal_id = os.environ.get("HUBSPOT_PORTAL_ID")
    if portal_id:
        return portal_id
    return detect_default_portal(".")


def main(args: list[str] | None = None) -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(description="HubSpot Webhook Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--portal-id", default=None, help="HubSpot portal ID")
    parsed = parser.parse_args(args)

    portal_id = _resolve_portal_id(parsed.portal_id)
    if not portal_id:
        print("Error: No portal ID provided. Use --portal-id or set HUBSPOT_PORTAL_ID.", file=sys.stderr)
        return 1

    portal_config = load_portal_config(portal_id)
    if not portal_config:
        print(f"Error: Portal config not found for {portal_id}.", file=sys.stderr)
        return 1

    creds = load_app_credentials()
    if not creds or not creds.get("client_secret"):
        print("Error: App client_secret not configured. Run save_app_credentials() first.", file=sys.stderr)
        return 1

    client_secret = creds["client_secret"]

    server = WebhookServer(
        client_secret=client_secret,
        portal_id=portal_id,
        host=parsed.host,
        port=parsed.port,
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
