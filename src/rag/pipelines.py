"""
RAG pipelines implemented with LangGraph (LangChain ecosystem).

Orchestration decision: docs/adr/0002-rag-orchestration-framework.md (LangGraph).

Standard RAG: retrieve -> generate
RAFT-LM v1: retrieve -> distractor filter -> evidence policy -> generate

Ingestion hooks: resolve_corpus_dir / ingest_corpus (src/rag/ingestion.py)
Retriever wiring: build_retriever with embedding_from_env / vector_store_from_env
Citation emission: _chunks_to_citations -> CitationRecord in PipelineResult
Benchmark integration: StandardRAGPipeline.run / RaftLMPipeline.run -> benchmark_runner
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.evals.benchmark_schema import CitationRecord
from src.rag.ingestion import ChunkRecord, load_questions, resolve_corpus_dir
from src.rag.retrievers import (
    BenchmarkBudget,
    ChunkRetriever,
    RetrievalLog,
    RetrievedChunk,
    build_retriever,
    budget_from_env,
)


class RAGGraphState(TypedDict, total=False):
    query: str
    retrieved: List[RetrievedChunk]
    filtered: List[RetrievedChunk]
    answer: str
    citations: List[CitationRecord]
    pipeline_name: str
    budget: BenchmarkBudget
    retrieval_log: RetrievalLog


@dataclass
class PipelineResult:
    query: str
    answer: str
    citations: List[CitationRecord]
    retrieved_chunks: List[RetrievedChunk]
    pipeline_name: str
    budget: BenchmarkBudget
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_log: Optional[RetrievalLog] = None


SYSTEM_PROMPT = (
    "You answer questions using only the provided policy excerpts. "
    "Cite chunk IDs in brackets when stating facts."
)
USER_PROMPT_TEMPLATE = (
    "Question: {query}\n\nContext:\n{context}\n\n"
    "Answer concisely and include citations like [chunk_id]."
)


def _build_context(chunks: List[RetrievedChunk], max_chars: int) -> str:
    parts: List[str] = []
    total = 0
    for ch in chunks:
        block = f"[{ch.chunk_id}] {ch.text.strip()}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _stub_generate(query: str, chunks: List[RetrievedChunk], max_chars: int) -> str:
    """Offline generator: extract best-matching sentence from context."""
    context = _build_context(chunks, max_chars)
    if not chunks:
        return "Insufficient evidence to answer. [none]"
    best = chunks[0]
    sentences = re.split(r"(?<=[.!?])\s+", best.text)
    answer_body = sentences[0].strip() if sentences else best.text.strip()
    return f"{answer_body} [{best.chunk_id}]"


def _chunks_to_citations(chunks: List[RetrievedChunk]) -> List[CitationRecord]:
    return [
        CitationRecord(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            excerpt=c.text[:200],
            score=c.score,
        )
        for c in chunks
    ]


def _node_retrieve(state: RAGGraphState, retriever: ChunkRetriever) -> RAGGraphState:
    budget = state["budget"]
    log = RetrievalLog(query=state["query"], top_k=budget.max_retrieval_depth)
    retrieved = retriever.retrieve(
        state["query"], budget.max_retrieval_depth, log=log
    )
    return {**state, "retrieved": retrieved, "retrieval_log": log}


def _node_generate(state: RAGGraphState) -> RAGGraphState:
    budget = state["budget"]
    chunks = state.get("filtered") or state.get("retrieved") or []
    answer = _stub_generate(state["query"], chunks, budget.max_context_chars)
    citations = _chunks_to_citations(chunks)
    return {**state, "answer": answer, "citations": citations}


def _build_standard_graph(retriever: ChunkRetriever) -> Any:
    graph = StateGraph(RAGGraphState)

    def retrieve(state: RAGGraphState) -> RAGGraphState:
        return _node_retrieve(state, retriever)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", _node_generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def _build_raft_graph(retriever: ChunkRetriever) -> Any:
    graph = StateGraph(RAGGraphState)

    def retrieve(state: RAGGraphState) -> RAGGraphState:
        return _node_retrieve(state, retriever)

    def distractor_filter(state: RAGGraphState) -> RAGGraphState:
        budget = state["budget"]
        retrieved = state.get("retrieved") or []
        filtered = retriever.apply_distractor_penalty(
            retrieved, budget.distractor_penalty
        )
        return {**state, "filtered": filtered}

    def evidence_policy(state: RAGGraphState) -> RAGGraphState:
        budget = state["budget"]
        filtered = state.get("filtered") or []
        kept = retriever.filter_by_evidence_policy(
            filtered,
            min_count=budget.min_evidence_count,
            threshold=budget.evidence_confidence_threshold,
        )
        return {**state, "filtered": kept}

    graph.add_node("retrieve", retrieve)
    graph.add_node("distractor_filter", distractor_filter)
    graph.add_node("evidence_policy", evidence_policy)
    graph.add_node("generate", _node_generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "distractor_filter")
    graph.add_edge("distractor_filter", "evidence_policy")
    graph.add_edge("evidence_policy", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


class StandardRAGPipeline:
    """Standard RAG baseline (LangGraph)."""

    def __init__(
        self,
        corpus_dir: Path,
        budget: Optional[BenchmarkBudget] = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir or resolve_corpus_dir())
        self.budget = budget or budget_from_env()
        self._retriever = build_retriever(self.corpus_dir)
        self._graph = _build_standard_graph(self._retriever)

    def run(self, query: str) -> PipelineResult:
        state: RAGGraphState = {
            "query": query,
            "budget": self.budget,
            "pipeline_name": "standard_rag",
        }
        out = self._graph.invoke(state)
        retrieved = out.get("retrieved") or []
        rlog = out.get("retrieval_log")
        return PipelineResult(
            query=query,
            answer=out.get("answer", ""),
            citations=out.get("citations") or [],
            retrieved_chunks=retrieved,
            pipeline_name="standard_rag",
            budget=self.budget,
            metadata={
                "model_provider": self.budget.model_provider,
                "embedding_model": self._retriever.embedding_model,
                "vector_store": self._retriever.vector_store_name,
                "generation_model": self.budget.generation_model,
            },
            retrieval_log=rlog,
        )


class RaftLMPipeline:
    """RAFT-LM v1: distractor-aware retrieval + evidence policy (LangGraph)."""

    def __init__(
        self,
        corpus_dir: Path,
        budget: Optional[BenchmarkBudget] = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir or resolve_corpus_dir())
        self.budget = budget or budget_from_env()
        self._retriever = build_retriever(self.corpus_dir)
        self._graph = _build_raft_graph(self._retriever)

    def run(self, query: str) -> PipelineResult:
        state: RAGGraphState = {
            "query": query,
            "budget": self.budget,
            "pipeline_name": "raft_lm",
        }
        out = self._graph.invoke(state)
        retrieved = out.get("filtered") or out.get("retrieved") or []
        rlog = out.get("retrieval_log")
        return PipelineResult(
            query=query,
            answer=out.get("answer", ""),
            citations=out.get("citations") or [],
            retrieved_chunks=retrieved,
            pipeline_name="raft_lm",
            budget=self.budget,
            metadata={
                "model_provider": self.budget.model_provider,
                "embedding_model": self._retriever.embedding_model,
                "vector_store": self._retriever.vector_store_name,
                "generation_model": self.budget.generation_model,
                "evidence_policy": True,
                "distractor_aware": True,
            },
            retrieval_log=rlog,
        )


class RaftDataBuilder:
    """RAFT-style Q/A pair builder hook (no training loop wired)."""

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = Path(corpus_dir)
        from src.rag.ingestion import ingest_corpus

        self._chunks: List[ChunkRecord] = ingest_corpus(self.corpus_dir)

    def build_pairs(self, max_pairs: int = 10) -> List[Dict[str, str]]:
        questions = load_questions(self.corpus_dir)
        pairs: List[Dict[str, str]] = []
        for row in questions[:max_pairs]:
            pairs.append(
                {
                    "question": row["question"],
                    "ground_truth": row.get("ground_truth", ""),
                    "risk_domain": row.get("risk_domain", "operational"),
                }
            )
        return pairs


