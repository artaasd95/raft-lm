"""CI-friendly smoke: ingestion -> retrieve -> generate -> report (mock/stub only)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.evals.benchmark_runner import run_standard_rag_benchmark
from src.evals.benchmark_schema import REQUIRED_REPORT_FIELDS, load_comparison_report
from src.evals.report_writer import validate_report_schema
from src.rag.embeddings import StubEmbeddingAdapter
from src.rag.ingestion import ingest_corpus, resolve_corpus_dir
from src.rag.pipelines import StandardRAGPipeline
from src.rag.retrievers import BenchmarkBudget, ChunkRetriever
from src.rag.vector_stores import InMemoryVectorStore


@pytest.fixture
def corpus_dir():
    return resolve_corpus_dir()


@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_MODE", "smoke")
    monkeypatch.setenv("EMBEDDING_MODEL", "deterministic-stub")
    monkeypatch.setenv("MODEL_PROVIDER", "stub")
    monkeypatch.setenv("VECTOR_STORE", "in_memory")
    monkeypatch.setenv("BENCHMARK_RESULTS_DIR", str(tmp_path))
    return tmp_path


def test_ingestion_stable_chunk_ids(corpus_dir):
    chunks = ingest_corpus(corpus_dir)
    assert len(chunks) >= 4
    ids = {c.chunk_id for c in chunks}
    assert len(ids) == len(chunks)
    sample = chunks[0]
    assert sample.metadata["corpus_id"] == "financial_policy_v1"
    assert sample.source_path.endswith(".md")
    assert "::chunk_" in sample.chunk_id


def test_retriever_mock_embedding(corpus_dir):
    chunks = ingest_corpus(corpus_dir)
    embed = StubEmbeddingAdapter()
    store = InMemoryVectorStore()
    retriever = ChunkRetriever(chunks, embedding=embed, store=store)
    hits = retriever.retrieve("CET1 capital ratio", top_k=2)
    assert len(hits) == 2
    assert retriever.embedding_model == "deterministic-stub"


def test_standard_pipeline_smoke(corpus_dir, mock_env):
    pytest.importorskip("langgraph")
    budget = BenchmarkBudget(
        max_retrieval_depth=2,
        model_provider="stub",
        embedding_model="deterministic-stub",
        vector_store="in_memory",
    )
    pipeline = StandardRAGPipeline(corpus_dir, budget=budget)
    result = pipeline.run("What is the minimum CET1 ratio?")
    assert result.answer
    assert result.citations
    assert result.retrieval_log is not None
    assert len(result.retrieval_log.chunk_ids) <= 2


def test_smoke_benchmark_writes_schema_artifacts(corpus_dir, mock_env):
    pytest.importorskip("langgraph")
    report = run_standard_rag_benchmark(
        corpus_dir=corpus_dir,
        out_dir=mock_env,
        questions_limit=1,
    )
    run_dir = mock_env / report.run_id
    report_path = run_dir / "report.json"
    assert report_path.exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "comparison_chart.json").exists()

    loaded = load_comparison_report(report_path)
    missing = validate_report_schema(loaded)
    assert missing == [], f"missing fields: {missing}"

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    for field in REQUIRED_REPORT_FIELDS:
        assert field in raw

    assert loaded.environment.benchmark_mode == "smoke"
    assert loaded.runs[0].retrieval is not None
    assert loaded.runs[0].retrieval.chunk_ids
