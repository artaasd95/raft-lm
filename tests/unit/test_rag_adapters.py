"""Unit tests for RAG adapter selection (mocks/fakers only; no paid APIs)."""

from __future__ import annotations

import pytest

from src.rag.embeddings import (
    AzureOpenAIEmbeddingAdapter,
    OpenAICompatibleEmbeddingAdapter,
    OpenAIEmbeddingAdapter,
    StubEmbeddingAdapter,
    embedding_from_env,
)
from src.rag.ingestion import ingest_corpus, resolve_corpus_dir
from src.rag.retrievers import ChunkRetriever, retriever_from_env
from src.rag.vector_stores import InMemoryVectorStore, vector_store_from_env


@pytest.fixture
def corpus_dir():
    return resolve_corpus_dir()


def test_embedding_from_env_mock_default(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODE", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "deterministic-stub")
    adapter = embedding_from_env()
    assert isinstance(adapter, StubEmbeddingAdapter)


def test_embedding_from_env_live_openai_model(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODE", "live")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = embedding_from_env()
    assert isinstance(adapter, OpenAIEmbeddingAdapter)
    assert adapter.model_name == "text-embedding-3-small"


def test_embedding_from_env_azure_prefix(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODE", "live")
    monkeypatch.setenv("EMBEDDING_MODEL", "azure:my-embedding-deploy")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    adapter = embedding_from_env()
    assert isinstance(adapter, AzureOpenAIEmbeddingAdapter)
    assert adapter.model_name == "my-embedding-deploy"


def test_embedding_from_env_compatible_prefix(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODE", "live")
    monkeypatch.setenv("EMBEDDING_MODEL", "compatible:local-embed")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8080/v1")
    adapter = embedding_from_env()
    assert isinstance(adapter, OpenAICompatibleEmbeddingAdapter)
    assert adapter.model_name == "local-embed"


def test_openai_adapter_requires_key():
    adapter = OpenAIEmbeddingAdapter(api_key="")
    with pytest.raises(RuntimeError):
        adapter.embed_query("test")


def test_azure_adapter_requires_credentials():
    adapter = AzureOpenAIEmbeddingAdapter(api_key="", endpoint="")
    with pytest.raises(RuntimeError):
        adapter.embed_query("test")


def test_compatible_adapter_requires_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatibleEmbeddingAdapter(model_name="m")


def test_vector_store_from_env_in_memory(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "in_memory")
    store = vector_store_from_env(dimension=16)
    assert store.store_name == "in_memory"


def test_vector_store_from_env_faiss_optional(monkeypatch):
    pytest.importorskip("faiss")
    monkeypatch.setenv("VECTOR_STORE", "faiss")
    store = vector_store_from_env(dimension=8)
    assert store.store_name == "faiss"


def test_vector_store_from_env_qdrant_optional(monkeypatch):
    pytest.importorskip("qdrant_client")
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    store = vector_store_from_env(dimension=8)
    assert store.store_name == "qdrant"


def test_retriever_adapter_injection_no_env_change(corpus_dir):
    chunks = ingest_corpus(corpus_dir)
    embed = StubEmbeddingAdapter(dimension=16)
    store = InMemoryVectorStore()
    retriever = ChunkRetriever(chunks, embedding=embed, store=store)
    hits = retriever.retrieve("CET1 capital", top_k=2)
    assert len(hits) == 2
    assert retriever.embedding_model == "deterministic-stub"
    assert retriever.vector_store_name == "in_memory"


def test_retriever_from_env_uses_mock_adapters(monkeypatch, corpus_dir):
    monkeypatch.setenv("EMBEDDING_MODE", "mock")
    monkeypatch.setenv("VECTOR_STORE", "in_memory")
    retriever = retriever_from_env(corpus_dir)
    assert retriever.embedding_model == "deterministic-stub"
    assert retriever.vector_store_name == "in_memory"
