"""Unit tests for LangGraph RAG pipelines."""

from pathlib import Path

import pytest

langgraph = pytest.importorskip("langgraph")

from src.rag.pipelines import RaftDataBuilder, RaftLMPipeline, StandardRAGPipeline
from src.rag.retrievers import BenchmarkBudget


@pytest.fixture
def corpus_dir():
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "benchmark_corpus" / "financial_policy"


@pytest.fixture
def shared_budget():
    return BenchmarkBudget(
        max_retrieval_depth=4,
        max_context_chars=4096,
        model_provider="stub",
        run_count=1,
    )


def test_standard_pipeline_returns_citations(corpus_dir, shared_budget):
    pipeline = StandardRAGPipeline(corpus_dir, budget=shared_budget)
    result = pipeline.run("What is the minimum CET1 ratio?")
    assert result.answer
    assert result.pipeline_name == "standard_rag"
    assert len(result.citations) >= 1
    assert result.budget.max_retrieval_depth == shared_budget.max_retrieval_depth


def test_raft_pipeline_same_budget(corpus_dir, shared_budget):
    standard = StandardRAGPipeline(corpus_dir, budget=shared_budget)
    raft = RaftLMPipeline(corpus_dir, budget=shared_budget)
    q = "What liquidity coverage ratio must be maintained?"
    std = standard.run(q)
    raft_res = raft.run(q)
    assert std.budget.max_retrieval_depth == raft_res.budget.max_retrieval_depth
    assert raft_res.metadata.get("distractor_aware") is True


def test_raft_data_builder_pairs(corpus_dir):
    builder = RaftDataBuilder(corpus_dir)
    pairs = builder.build_pairs(max_pairs=2)
    assert len(pairs) == 2
    assert "question" in pairs[0]
