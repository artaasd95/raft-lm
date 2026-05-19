"""Unit tests for vector store adapters."""

import pytest

from src.rag.embeddings import StubEmbeddingAdapter
from src.rag.vector_stores import InMemoryVectorStore


def test_in_memory_search_ordering():
    store = InMemoryVectorStore()
    embed = StubEmbeddingAdapter(dimension=8)
    texts = ["capital ratio CET1", "office cafeteria supplies", "liquidity coverage"]
    vectors = embed.embed_documents(texts)
    store.upsert(["c0", "c1", "c2"], vectors, [{"t": t} for t in texts])
    q = embed.embed_query("CET1 capital")
    hits = store.search(q, top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_faiss_store_optional():
    pytest.importorskip("faiss")
    from src.rag.vector_stores import FaissVectorStore

    store = FaissVectorStore()
    embed = StubEmbeddingAdapter(dimension=8)
    vectors = embed.embed_documents(["a", "b"])
    store.upsert(["a", "b"], vectors, [{}, {}])
    hits = store.search(embed.embed_query("a"), top_k=1)
    assert len(hits) == 1
