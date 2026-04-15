"""Tests for the minimal environment-driven config builder."""

from __future__ import annotations

import pytest

from mem0_mcp_selfhosted.config import build_config


@pytest.fixture(autouse=True)
def _base_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
    monkeypatch.delenv("MEM0_ENABLE_GRAPH", raising=False)


def test_build_config_uses_openai_compatible_providers(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://100.105.112.4:8317/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("MEM0_QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("MEM0_COLLECTION", "shared_memory")

    config = build_config()

    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["config"]["openai_base_url"] == "http://100.105.112.4:8317/v1"
    assert config["embedder"]["provider"] == "openai"
    assert config["embedder"]["config"]["model"] == "BAAI/bge-m3"
    assert config["vector_store"]["provider"] == "qdrant"
    assert config["vector_store"]["config"]["collection_name"] == "shared_memory"
    assert config["vector_store"]["config"]["embedding_model_dims"] == 1024


def test_build_config_accepts_siliconflow_api_key_alias(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sf-key")

    config = build_config()

    assert config["embedder"]["config"]["api_key"] == "sf-key"


def test_build_config_rejects_graph_mode(monkeypatch):
    monkeypatch.setenv("MEM0_ENABLE_GRAPH", "true")

    with pytest.raises(ValueError, match="not supported"):
        build_config()
