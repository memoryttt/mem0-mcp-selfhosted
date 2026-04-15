"""Environment-driven configuration for a minimal mem0 MCP server."""

from __future__ import annotations

from typing import Any

from mem0_mcp_selfhosted.env import bool_env, env, opt_env


def _build_openai_config(
    *,
    api_key_env: str,
    base_url_env: str,
    model_env: str,
    default_base_url: str,
    default_model: str,
) -> dict[str, Any]:
    return {
        "api_key": env(api_key_env),
        "openai_base_url": env(base_url_env, default_base_url),
        "model": env(model_env, default_model),
    }


def build_config() -> dict[str, Any]:
    """Build a minimal mem0 config using built-in openai + qdrant providers."""
    if bool_env("MEM0_ENABLE_GRAPH"):
        raise ValueError(
            "MEM0_ENABLE_GRAPH is not supported in this safe fork. "
            "Neo4j and graph memory were removed intentionally."
        )

    embed_dims = int(env("MEM0_EMBED_DIMS", "1024"))
    qdrant_url = env("MEM0_QDRANT_URL", "http://127.0.0.1:6333")
    collection = env("MEM0_COLLECTION", "mem0_mcp_selfhosted")
    qdrant_api_key = opt_env("MEM0_QDRANT_API_KEY")
    history_db_path = opt_env("MEM0_HISTORY_DB_PATH")

    vector_config: dict[str, Any] = {
        "collection_name": collection,
        "url": qdrant_url,
        "embedding_model_dims": embed_dims,
    }
    if qdrant_api_key:
        vector_config["api_key"] = qdrant_api_key
    if bool_env("MEM0_QDRANT_ON_DISK"):
        vector_config["on_disk"] = True

    qdrant_timeout = opt_env("MEM0_QDRANT_TIMEOUT")
    if qdrant_timeout:
        from qdrant_client import QdrantClient

        client_kwargs: dict[str, Any] = {
            "url": qdrant_url,
            "timeout": int(qdrant_timeout),
        }
        if qdrant_api_key:
            client_kwargs["api_key"] = qdrant_api_key
        vector_config["client"] = QdrantClient(**client_kwargs)

    config_dict: dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": _build_openai_config(
                api_key_env="OPENAI_API_KEY",
                base_url_env="OPENAI_BASE_URL",
                model_env="OPENAI_MODEL",
                default_base_url="https://api.openai.com/v1",
                default_model="gpt-4o-mini",
            ),
        },
        "embedder": {
            "provider": "openai",
            "config": _build_openai_config(
                api_key_env="EMBEDDING_API_KEY",
                base_url_env="EMBEDDING_BASE_URL",
                model_env="EMBEDDING_MODEL",
                default_base_url="https://api.siliconflow.cn/v1",
                default_model="BAAI/bge-m3",
            ),
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vector_config,
        },
        "version": "v1.1",
    }

    if history_db_path:
        config_dict["history_db_path"] = history_db_path

    return config_dict
