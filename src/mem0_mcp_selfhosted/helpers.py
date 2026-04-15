"""Shared utilities for the minimal mem0 MCP server."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from mem0_mcp_selfhosted.env import env

logger = logging.getLogger(__name__)


def get_default_user_id() -> str:
    """Get the default user_id from MEM0_USER_ID env var."""
    return env("MEM0_USER_ID", "user")


def _mem0_call(func: Callable, *args: Any, **kwargs: Any) -> str:
    """Wrap a mem0 call with structured error handling."""
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        exc_type = type(exc).__name__
        is_memory_error = any(cls.__name__ == "MemoryError" for cls in type(exc).__mro__)
        if is_memory_error:
            logger.error("Mem0 call failed: %s", exc)
            return json.dumps(
                {
                    "error": str(exc),
                    "error_code": getattr(exc, "error_code", None),
                    "details": getattr(exc, "details", None),
                    "suggestion": getattr(exc, "suggestion", None),
                },
                ensure_ascii=False,
            )
        logger.error("Unexpected error: %s", exc)
        return json.dumps(
            {
                "error": exc_type,
                "detail": str(exc),
            },
            ensure_ascii=False,
        )
    return json.dumps(result, ensure_ascii=False)


def safe_bulk_delete(memory: Any, filters: dict[str, Any]) -> int:
    """Safely delete all memories matching filters without calling delete_all()."""
    result = memory.vector_store.list(filters=filters)
    memories = result[0] if isinstance(result, tuple) else result

    count = 0
    for item in memories:
        memory_id = item.id if hasattr(item, "id") else item.get("id") if isinstance(item, dict) else str(item)
        try:
            memory.delete(memory_id)
            count += 1
        except Exception as exc:
            logger.warning("Failed to delete memory %s: %s", memory_id, exc)

    return count


def list_entities_facet(memory: Any) -> dict[str, list[dict]]:
    """List entities using Qdrant Facet API with scroll fallback."""
    client = memory.vector_store.client
    collection = memory.vector_store.collection_name

    result: dict[str, list[dict]] = {"users": [], "agents": [], "runs": []}
    entity_keys = {"users": "user_id", "agents": "agent_id", "runs": "run_id"}

    try:
        for result_key, payload_key in entity_keys.items():
            facet_response = client.facet(
                collection_name=collection,
                key=payload_key,
            )
            hits = getattr(facet_response, "hits", facet_response)
            result[result_key] = [
                {"value": hit.value, "count": hit.count}
                if hasattr(hit, "value") and hasattr(hit, "count")
                else {"value": hit.get("value"), "count": hit.get("count")}
                for hit in hits
                if (hasattr(hit, "value") and hit.value is not None)
                or (isinstance(hit, dict) and hit.get("value") is not None)
            ]
        return result
    except Exception as exc:
        logger.info("Facet API unavailable, falling back to scroll: %s", exc)

    seen: dict[str, dict[str, int]] = {"users": {}, "agents": {}, "runs": {}}
    records, _ = client.scroll(
        collection_name=collection,
        with_payload=True,
        with_vectors=False,
        limit=1000,
    )
    for record in records:
        payload = getattr(record, "payload", {}) or {}
        for result_key, payload_key in entity_keys.items():
            value = payload.get(payload_key)
            if value:
                seen[result_key][value] = seen[result_key].get(value, 0) + 1

    for result_key, values in seen.items():
        result[result_key] = [
            {"value": value, "count": count}
            for value, count in sorted(values.items(), key=lambda item: (-item[1], item[0]))
        ]

    return result
