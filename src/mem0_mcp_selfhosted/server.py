"""FastMCP server for a minimal, safer mem0 MCP deployment."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Annotated, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from mem0_mcp_selfhosted.config import build_config
from mem0_mcp_selfhosted.env import env
from mem0_mcp_selfhosted.helpers import (
    _mem0_call,
    get_default_user_id,
    list_entities_facet,
    safe_bulk_delete,
)

logger = logging.getLogger(__name__)

memory = None
mcp: FastMCP | None = None

_memory_init_lock = threading.Lock()
_last_init_failure: float = 0.0
_INIT_RETRY_COOLDOWN = 30.0


def _update_memory_preserving_metadata(mem: Any, memory_id: str, text: str) -> dict[str, str]:
    """Update memory text while preserving existing metadata."""
    existing = mem.get(memory_id)
    existing_metadata = existing.get("metadata") if isinstance(existing, dict) else None
    mem.update(memory_id, data=text, metadata=existing_metadata)
    return {"message": "Memory updated successfully!"}


def _init_memory() -> Any:
    """Initialize mem0ai Memory from environment config."""
    global memory

    from mem0 import Memory

    memory = Memory.from_config(build_config())
    return memory


def _ensure_memory() -> Any:
    """Lazy-initialize Memory on first tool call."""
    global memory, _last_init_failure

    if memory is not None:
        return memory

    now = time.monotonic()
    if _last_init_failure and (now - _last_init_failure < _INIT_RETRY_COOLDOWN):
        return None

    with _memory_init_lock:
        if memory is not None:
            return memory

        try:
            _init_memory()
            logger.info("mem0ai Memory initialized successfully (lazy)")
        except Exception as exc:
            _last_init_failure = time.monotonic()
            logger.error("Lazy Memory init failed: %s", exc)
            return None

    return memory


def _create_server() -> FastMCP:
    """Create and configure the FastMCP server."""
    global mcp

    host = env("MEM0_HOST", "0.0.0.0")
    port = int(env("MEM0_PORT", "8081"))

    mcp = FastMCP(
        "mem0",
        host=host,
        port=port,
        instructions=(
            "Memory tools for persistent cross-session memory. "
            "Use search_memories to find relevant context before starting work. "
            "Use add_memory to store important facts, preferences, and decisions. "
            "Use get_memories to browse stored memories with filters. "
            "Use get_memory to retrieve a specific memory by ID. "
            "Use update_memory to modify existing memories. "
            "Use list_entities to see who or what has stored memories."
        ),
    )

    _register_tools(mcp)
    _register_prompts(mcp)
    return mcp


def _memory_not_ready() -> str:
    return json.dumps(
        {"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."},
        ensure_ascii=False,
    )


def _register_tools(mcp: FastMCP) -> None:
    """Register the memory management MCP tools."""

    @mcp.tool()
    def add_memory(
        text: Annotated[str, Field(description="Text to store as a memory. Converted to messages format internally.")],
        messages: Annotated[list[dict] | None, Field(description="Structured conversation history (role/content dicts). When provided, takes precedence over text.")] = None,
        user_id: Annotated[str | None, Field(description="User scope identifier. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope identifier.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope identifier.")] = None,
        metadata: Annotated[dict | None, Field(description="Arbitrary metadata JSON to store alongside the memory.")] = None,
        infer: Annotated[bool | None, Field(description="If true (default), LLM extracts key facts. If false, stores raw text.")] = None,
    ) -> str:
        """Store a new memory."""
        uid = user_id or get_default_user_id()
        msgs = messages if messages else [{"role": "user", "content": text}]

        kwargs: dict[str, Any] = {"user_id": uid}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if metadata:
            kwargs["metadata"] = metadata
        if infer is not None:
            kwargs["infer"] = infer

        mem = _ensure_memory()
        if mem is None:
            return _memory_not_ready()
        return _mem0_call(mem.add, msgs, **kwargs)

    @mcp.tool()
    def search_memories(
        query: Annotated[str, Field(description="Natural language description of what to find.")],
        user_id: Annotated[str | None, Field(description="User scope. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope.")] = None,
        filters: Annotated[dict | None, Field(description="Additional structured filter clauses.")] = None,
        limit: Annotated[int | None, Field(description="Maximum number of results.")] = None,
        threshold: Annotated[float | None, Field(description="Minimum relevance score (0.0-1.0).")] = None,
        rerank: Annotated[bool | None, Field(description="Whether to apply reranking.")] = None,
    ) -> str:
        """Semantic search across existing memories."""
        uid = user_id or get_default_user_id()
        kwargs: dict[str, Any] = {"user_id": uid}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if filters:
            kwargs["filters"] = filters
        if limit is not None:
            kwargs["limit"] = limit
        if threshold is not None:
            kwargs["threshold"] = threshold
        if rerank is not None:
            kwargs["rerank"] = rerank

        mem = _ensure_memory()
        if mem is None:
            return _memory_not_ready()
        return _mem0_call(mem.search, query, **kwargs)

    @mcp.tool()
    def get_memories(
        user_id: Annotated[str | None, Field(description="User scope. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope.")] = None,
        limit: Annotated[int | None, Field(description="Maximum number of memories to return.")] = None,
    ) -> str:
        """Page through memories using filters instead of search."""
        uid = user_id or get_default_user_id()

        kwargs: dict[str, Any] = {"user_id": uid}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if limit is not None:
            kwargs["limit"] = limit

        mem = _ensure_memory()
        if mem is None:
            return _memory_not_ready()
        return _mem0_call(mem.get_all, **kwargs)

    @mcp.tool()
    def get_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to fetch.")],
    ) -> str:
        """Fetch a single memory by its ID."""
        mem = _ensure_memory()
        if mem is None:
            return _memory_not_ready()
        return _mem0_call(mem.get, memory_id)

    @mcp.tool()
    def update_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to update.")],
        text: Annotated[str, Field(description="Replacement text for the memory.")],
    ) -> str:
        """Overwrite an existing memory's text."""
        mem = _ensure_memory()
        if mem is None:
            return _memory_not_ready()

        def _do_update() -> dict[str, str]:
            return _update_memory_preserving_metadata(mem, memory_id, text)

        return _mem0_call(_do_update)

    @mcp.tool()
    def delete_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to delete.")],
    ) -> str:
        """Delete a single memory."""
        mem = _ensure_memory()
        if mem is None:
            return _memory_not_ready()

        def _do_delete() -> dict[str, str]:
            mem.delete(memory_id)
            return {"message": "Memory deleted successfully!"}

        return _mem0_call(_do_delete)

    @mcp.tool()
    def delete_all_memories(
        user_id: Annotated[str | None, Field(description="User scope to delete.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope to delete.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope to delete.")] = None,
    ) -> str:
        """Bulk-delete all memories in the given scope."""
        uid = user_id or get_default_user_id()

        filters: dict[str, Any] = {}
        if uid:
            filters["user_id"] = uid
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        mem = _ensure_memory()
        if mem is None:
            return _memory_not_ready()

        def _do_bulk_delete() -> dict[str, Any]:
            count = safe_bulk_delete(mem, filters)
            return {"message": f"Deleted {count} memories.", "count": count}

        return _mem0_call(_do_bulk_delete)

    @mcp.tool()
    def list_entities() -> str:
        """List which users, agents, or runs currently hold memories."""
        mem = _ensure_memory()
        if mem is None:
            return _memory_not_ready()
        return _mem0_call(list_entities_facet, mem)

    @mcp.tool()
    def delete_entities(
        user_id: Annotated[str | None, Field(description="User entity to delete (cascades to all memories).")] = None,
        agent_id: Annotated[str | None, Field(description="Agent entity to delete.")] = None,
        run_id: Annotated[str | None, Field(description="Run entity to delete.")] = None,
    ) -> str:
        """Delete an entity and cascade-delete all its memories."""
        if not any([user_id, agent_id, run_id]):
            return json.dumps(
                {"error": "At least one scope (user_id, agent_id, or run_id) is required."},
                ensure_ascii=False,
            )

        filters: dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        mem = _ensure_memory()
        if mem is None:
            return _memory_not_ready()

        def _do_delete_entity() -> dict[str, Any]:
            count = safe_bulk_delete(mem, filters)
            return {"message": f"Entity deleted. Removed {count} memories.", "count": count}

        return _mem0_call(_do_delete_entity)


def _register_prompts(mcp: FastMCP) -> None:
    """Register MCP prompts."""

    @mcp.prompt()
    def memory_assistant() -> str:
        """Quick-start guide for using the mem0 memory server."""
        return (
            "You are using the mem0 MCP server for long-term memory management.\n\n"
            "Quick Start:\n"
            "1. Store memories: Use add_memory to save facts, preferences, or conversations\n"
            "2. Search memories: Use search_memories for semantic queries\n"
            "3. Browse memories: Use get_memories for filtered listing\n"
            "4. Update/Delete: Use update_memory and delete_memory for modifications\n"
            "5. Use list_entities when you need a quick inventory of stored scopes\n\n"
            "Tips:\n"
            "- user_id is automatically injected from MEM0_USER_ID by default\n"
            "- Use infer=false to store raw text without LLM extraction\n"
            "- Use threshold on search_memories to filter by relevance score\n"
            "- Use filters for structured queries: {\"key\": {\"eq\": \"value\"}}\n"
        )


def run_server() -> None:
    """Entry point: create server and run over stdio only."""
    log_level = env("MEM0_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s | %(message)s",
    )

    load_dotenv()

    transport = env("MEM0_TRANSPORT", "stdio").lower()
    if transport != "stdio":
        raise ValueError(
            f"Unsupported MEM0_TRANSPORT={transport!r}. "
            "This safe fork only supports stdio."
        )

    server = _create_server()
    server.run(transport="stdio")
