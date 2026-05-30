"""Retrievers, benchmark budget, and vector-store wiring for Standard RAG vs RAFT-LM."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.embeddings import EmbeddingAdapter, embedding_from_env
from src.rag.ingestion import ChunkRecord, ingest_corpus
from src.rag.vector_stores import VectorStoreAdapter, vector_store_from_env

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkBudget:
    max_retrieval_depth: int = 4
    max_context_chars: int = 4096
    model_provider: str = "stub"
    run_count: int = 1
    min_evidence_count: int = 1
    evidence_confidence_threshold: float = 0.15
    distractor_penalty: float = 0.35
    embedding_model: str = "deterministic-stub"
    vector_store: str = "in_memory"
    generation_model: str = "deterministic-stub"
    seed: Optional[int] = None


@dataclass
class RetrievalLog:
    """Per-query retrieval budget fields for report schema parity."""

    query: str
    top_k: int
    chunk_ids: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    embedding_model: str = ""
    vector_store: str = ""
    context_chars_used: int = 0


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    distractor_keywords: List[str]
    source_path: str = ""


class ChunkRetriever:
    """Index ingested chunks with pluggable embeddings and vector stores."""

    def __init__(
        self,
        chunks: List[ChunkRecord],
        *,
        embedding: Optional[EmbeddingAdapter] = None,
        store: Optional[VectorStoreAdapter] = None,
    ) -> None:
        self._chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._embedding = embedding or embedding_from_env()
        self._store = store or vector_store_from_env(self._embedding.dimension)
        texts = [c.text for c in chunks]
        vectors = self._embedding.embed_documents(texts)
        payloads = [
            {
                "doc_id": c.doc_id,
                "text": c.text,
                "source_path": c.source_path,
                "distractor_keywords": c.distractor_keywords,
            }
            for c in chunks
        ]
        self._store.upsert(
            [c.chunk_id for c in chunks],
            vectors,
            payloads,
        )

    @property
    def embedding_model(self) -> str:
        return self._embedding.model_name

    @property
    def vector_store_name(self) -> str:
        return self._store.store_name

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        log: Optional[RetrievalLog] = None,
    ) -> List[RetrievedChunk]:
        q_vec = self._embedding.embed_query(query)
        hits = self._store.search(q_vec, top_k)
        results: List[RetrievedChunk] = []
        for hit in hits:
            chunk = self._by_id[hit.id]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=hit.score,
                    distractor_keywords=list(chunk.distractor_keywords),
                    source_path=chunk.source_path,
                )
            )
        logger.info(
            "retrieve top_k=%s store=%s embedding=%s hits=%s",
            top_k,
            self.vector_store_name,
            self.embedding_model,
            len(results),
        )
        if log is not None:
            log.top_k = top_k
            log.chunk_ids = [r.chunk_id for r in results]
            log.scores = [r.score for r in results]
            log.embedding_model = self.embedding_model
            log.vector_store = self.vector_store_name
        return results

    def apply_distractor_penalty(
        self,
        chunks: List[RetrievedChunk],
        penalty: float,
    ) -> List[RetrievedChunk]:
        from src.rag.raft_policy import apply_distractor_penalty

        return apply_distractor_penalty(chunks, penalty)

    def filter_by_evidence_policy(
        self,
        chunks: List[RetrievedChunk],
        min_count: int,
        threshold: float,
    ) -> List[RetrievedChunk]:
        from src.rag.raft_policy import filter_by_evidence_policy

        return filter_by_evidence_policy(
            chunks, min_count=min_count, threshold=threshold
        )


# Backward-compatible alias
VectorRetriever = ChunkRetriever


def build_retriever(
    corpus_dir,
    *,
    embedding: Optional[EmbeddingAdapter] = None,
    store: Optional[VectorStoreAdapter] = None,
) -> ChunkRetriever:
    chunks = ingest_corpus(corpus_dir)
    return ChunkRetriever(chunks, embedding=embedding, store=store)


def retriever_from_env(corpus_dir) -> ChunkRetriever:
    """Build retriever using embedding and vector-store adapters from environment."""
    embed = embedding_from_env()
    store = vector_store_from_env(embed.dimension)
    return build_retriever(corpus_dir, embedding=embed, store=store)


def budget_from_env() -> BenchmarkBudget:
    return BenchmarkBudget(
        max_retrieval_depth=int(os.getenv("MAX_RETRIEVAL_DEPTH", "4")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "4096")),
        model_provider=os.getenv("MODEL_PROVIDER", "stub"),
        run_count=int(os.getenv("BENCHMARK_RUN_COUNT", "1")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "deterministic-stub"),
        vector_store=os.getenv("VECTOR_STORE", "in_memory"),
        generation_model=os.getenv("GENERATION_MODEL", "deterministic-stub"),
        seed=int(os.getenv("BENCHMARK_SEED", "0")) or None,
    )
