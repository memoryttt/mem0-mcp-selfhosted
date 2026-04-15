# mem0-mcp-selfhosted

A safer fork of `elvismdev/mem0-mcp-selfhosted` focused on a minimal local deployment:

- `mem0` built-in `openai` provider for the main LLM
- `mem0` built-in `openai` provider for embeddings
- `Qdrant` as the only vector store
- `stdio` as the only MCP transport
- no Claude credential scraping
- no Anthropic OAuth/OAT refresh flow
- no hooks that patch local editor/client config
- no Neo4j graph memory
- no `sse` or `streamable-http`

This fork is intended for setups like:

- main LLM through an OpenAI-compatible proxy such as `http://100.105.112.4:8317/v1`
- embedding through SiliconFlow `BAAI/bge-m3`
- local Qdrant on `127.0.0.1:6333`

## What Changed

Compared with upstream, this fork removes the highest-risk features:

- deleted `auth.py` and all Claude `~/.claude/.credentials.json` reads
- deleted Anthropic-specific client code and token refresh logic
- deleted hook installers and transcript ingestion hooks
- deleted graph/Neo4j integrations and graph tools
- restricted the server to `stdio` transport only
- replaced custom provider routing with a minimal config built on mem0's built-in providers

## Prerequisites

- Python 3.10+
- `uv`
- Qdrant reachable from this host
- one OpenAI-compatible LLM endpoint
- one OpenAI-compatible embedding endpoint

## Configuration

All configuration is via environment variables.

Required:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `EMBEDDING_API_KEY` or `SILICONFLOW_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`

Common optional:

- `MEM0_EMBED_DIMS` default `1024`
- `MEM0_QDRANT_URL` default `http://127.0.0.1:6333`
- `MEM0_COLLECTION` default `mem0_mcp_selfhosted`
- `MEM0_QDRANT_API_KEY`
- `MEM0_QDRANT_ON_DISK` default `false`
- `MEM0_QDRANT_TIMEOUT`
- `MEM0_USER_ID` default `user`
- `MEM0_TRANSPORT` must be `stdio`

Removed on purpose:

- `MEM0_ENABLE_GRAPH`
- any Neo4j-related variables
- any Anthropic or Claude credential variables

## Example

```bash
OPENAI_API_KEY=dummy \
  OPENAI_BASE_URL=http://100.105.112.4:8317/v1 \
  OPENAI_MODEL=gpt-4o-mini \
  EMBEDDING_API_KEY=your-siliconflow-key \
  EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1 \
  EMBEDDING_MODEL=BAAI/bge-m3 \
  MEM0_QDRANT_URL=http://127.0.0.1:6333 \
  MEM0_TRANSPORT=stdio \
  uv run mem0-mcp-selfhosted
```

For Codex CLI or other MCP clients, register the server with `stdio` and pass the same environment variables through the client config.

## Tools

This fork exposes these MCP tools:

- `add_memory`
- `search_memories`
- `get_memories`
- `get_memory`
- `update_memory`
- `delete_memory`
- `delete_all_memories`
- `list_entities`
- `delete_entities`

## Security Notes

This fork reduces attack surface, but it is still code you should review before production use.

What this fork does better than upstream:

- avoids runtime install from a moving Git branch in the default docs
- does not read local Claude auth state
- does not refresh third-party OAuth tokens
- does not mutate `.claude/settings.json`
- does not auto-ingest session transcripts
- does not expose HTTP MCP transports by default

What you still need to own:

- pin dependencies in your deployment environment
- protect your embedding and LLM API keys
- restrict which local users can modify the repo or launch the server
- audit future upstream merges before rebasing this fork
