from __future__ import annotations

import json
from difflib import get_close_matches
from typing import Any

from hubspot_agent.cache import SchemaCache


class ValidationError(Exception):
    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.suggestions = suggestions or []


def _extract_property_names(schema_data: dict[str, Any] | None) -> set[str]:
    if schema_data is None:
        return set()
    results = schema_data.get("results", [])
    if not isinstance(results, list):
        return set()
    return {str(n) for p in results if isinstance(p, dict) and (n := p.get("name")) is not None and n != ""}


def _extract_property_type(schema_data: dict[str, Any] | None, prop_name: str) -> str | None:
    if schema_data is None:
        return None
    results = schema_data.get("results", [])
    if not isinstance(results, list):
        return None
    for p in results:
        if isinstance(p, dict) and p.get("name") == prop_name:
            return p.get("type")
    return None


def _type_compatible(value: Any, prop_type: str | None) -> bool:
    if prop_type is None:
        return True
    if value is None:
        return True
    if prop_type == "string":
        return isinstance(value, str)
    if prop_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if prop_type == "bool":
        return isinstance(value, bool)
    if prop_type == "enumeration":
        return isinstance(value, str)
    if prop_type == "date":
        return isinstance(value, str)
    return True


def validate_properties(
    object_type: str,
    properties: dict[str, Any],
    portal_id: str,
    base_dir: Any = None,
    retry_with_refresh: bool = True,
) -> dict[str, Any]:
    """Validate property names and types against cached schema.

    Returns a dict with:
      - 'valid': bool
      - 'errors': list[dict] — each with 'property', 'reason', 'suggestions'
      - 'refreshed': bool — whether cache was refreshed and re-checked
    """
    cache = SchemaCache(portal_id, base_dir=base_dir)
    schema_data = cache.get(object_type)

    # Graceful degradation: if no schema cached, allow the write
    if schema_data is None:
        return {"valid": True, "errors": [], "refreshed": False}

    known = _extract_property_names(schema_data)

    errors: list[dict[str, Any]] = []

    for prop_name, value in properties.items():
        if prop_name not in known:
            suggestions = get_close_matches(prop_name, known, n=3, cutoff=0.6)
            errors.append({
                "property": prop_name,
                "reason": "unknown_property",
                "suggestions": suggestions,
            })
            continue

        prop_type = _extract_property_type(schema_data, prop_name)
        if not _type_compatible(value, prop_type):
            errors.append({
                "property": prop_name,
                "reason": f"type_mismatch (expected {prop_type}, got {type(value).__name__})",
                "suggestions": [],
            })

    if errors and retry_with_refresh:
        # Schema might be stale — invalidate and re-check once
        cache.invalidate(object_type)
        refreshed_schema = cache.get(object_type)
        if refreshed_schema is not None:
            # re-run without retry to avoid infinite loops
            result = validate_properties(
                object_type, properties, portal_id, base_dir=base_dir, retry_with_refresh=False
            )
            result["refreshed"] = True
            return result

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "refreshed": False,
    }
