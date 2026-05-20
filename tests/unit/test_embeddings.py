"""Unit tests for embedding adapters (stub only; no live API)."""

import pytest

from src.rag.embeddings import OpenAIEmbeddingAdapter, StubEmbeddingAdapter, embedding_from_env


def test_stub_embedding_deterministic():
    adapter = StubEmbeddingAdapter(dimension=16)
    v1 = adapter.embed_query("hello")
    v2 = adapter.embed_query("hello")
    v3 = adapter.embed_query("world")
    assert v1 == v2
    assert v1 != v3
    assert len(v1) == 16


def test_stub_embed_documents_batch():
    adapter = StubEmbeddingAdapter()
    vecs = adapter.embed_documents(["a", "b"])
    assert len(vecs) == 2


def test_embedding_from_env_stub(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODE", "mock")
    monkeypatch.setenv("EMBEDDING_MODEL", "deterministic-stub")
    adapter = embedding_from_env()
    assert isinstance(adapter, StubEmbeddingAdapter)


def test_openai_adapter_requires_key():
    adapter = OpenAIEmbeddingAdapter(api_key="")
    with pytest.raises(RuntimeError):
        adapter.embed_query("test")
