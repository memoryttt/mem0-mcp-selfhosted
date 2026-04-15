"""MCP protocol smoke tests for the minimal server."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import mem0_mcp_selfhosted.server as server_mod

EXPECTED_TOOLS = {
    "add_memory",
    "search_memories",
    "get_memories",
    "get_memory",
    "update_memory",
    "delete_memory",
    "delete_all_memories",
    "list_entities",
    "delete_entities",
}

REQUIRED_PARAMS = {
    "add_memory": {"text"},
    "search_memories": {"query"},
    "get_memories": set(),
    "get_memory": {"memory_id"},
    "update_memory": {"memory_id", "text"},
    "delete_memory": {"memory_id"},
    "delete_all_memories": set(),
    "list_entities": set(),
    "delete_entities": set(),
}


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("MEM0_USER_ID", "test-user")


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.add.return_value = {"results": [{"id": "mem-1", "memory": "test fact"}]}
    mem.search.return_value = {"results": [{"id": "mem-1", "score": 0.95}]}
    mem.get_all.return_value = {"results": []}
    mem.get.return_value = {"id": "mem-1", "memory": "test fact"}
    mem.update.return_value = None
    mem.delete.return_value = None
    mem.vector_store.client.facet.return_value = []
    mem.vector_store.collection_name = "test"
    return mem


@pytest.fixture
def mcp_server(mock_memory):
    original_memory = server_mod.memory
    server_mod.memory = mock_memory
    srv = server_mod._create_server()
    yield srv
    server_mod.memory = original_memory


class TestToolDiscovery:
    @pytest.mark.asyncio
    async def test_list_tools_returns_expected_set(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {tool.name for tool in tools}
        assert tool_names == EXPECTED_TOOLS
        assert len(tools) == 9

    @pytest.mark.asyncio
    async def test_tool_schemas_have_required_params(self, mcp_server):
        tools = await mcp_server.list_tools()
        for tool in tools:
            schema = tool.inputSchema
            assert schema["type"] == "object"
            assert set(schema.get("required", [])) == REQUIRED_PARAMS[tool.name]


class TestCallToolRoundTrip:
    @pytest.mark.asyncio
    async def test_add_memory(self, mcp_server, mock_memory):
        content_blocks, _ = await mcp_server.call_tool("add_memory", {"text": "I prefer Python"})
        parsed = json.loads(content_blocks[0].text)
        assert "results" in parsed
        mock_memory.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_memories(self, mcp_server, mock_memory):
        content_blocks, _ = await mcp_server.call_tool("search_memories", {"query": "Python preferences"})
        parsed = json.loads(content_blocks[0].text)
        assert "results" in parsed
        mock_memory.search.assert_called_once_with("Python preferences", user_id="test-user")


class TestPromptDiscovery:
    @pytest.mark.asyncio
    async def test_list_prompts_contains_memory_assistant(self, mcp_server):
        prompts = await mcp_server.list_prompts()
        assert "memory_assistant" in {prompt.name for prompt in prompts}


class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_tool_exception_returns_json_error(self, mcp_server, mock_memory):
        mock_memory.get.side_effect = RuntimeError("connection lost")
        content_blocks, _ = await mcp_server.call_tool("get_memory", {"memory_id": "uuid-123"})
        parsed = json.loads(content_blocks[0].text)
        assert parsed["error"] == "RuntimeError"
        assert "connection lost" in parsed["detail"]
