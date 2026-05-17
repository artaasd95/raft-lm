"""Unit tests for in-memory retriever."""

from pathlib import Path

import pytest

from src.rag.corpus import load_corpus_chunks
from src.rag.retrievers import BenchmarkBudget, VectorRetriever


@pytest.fixture
def corpus_dir():
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "benchmark_corpus" / "financial_policy"


def test_retriever_returns_top_k(corpus_dir):
    chunks = load_corpus_chunks(corpus_dir)
    retriever = VectorRetriever(chunks)
    results = retriever.retrieve("CET1 capital ratio", top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_distractor_penalty_lowers_score(corpus_dir):
    chunks = load_corpus_chunks(corpus_dir)
    retriever = VectorRetriever(chunks)
    results = retriever.retrieve("policy", top_k=4)
    penalized = retriever.apply_distractor_penalty(results, penalty=0.5)
    assert penalized[0].score <= results[0].score + 0.01


def test_budget_defaults():
    budget = BenchmarkBudget()
    assert budget.max_retrieval_depth == 4
    assert budget.max_context_chars == 4096
