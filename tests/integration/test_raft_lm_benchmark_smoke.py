"""Integration smoke test for RAFT-LM benchmark path (mock providers only)."""

import json
from pathlib import Path

import pytest

from src.evals.benchmark_runner import run_benchmark_comparison, run_raft_lm_benchmark
from src.evals.benchmark_schema import load_comparison_report
from src.evals.compare_runs import compare_run_directories, write_comparison_delta
from src.evals.ragas_runner import score_saved_artifacts, validate_ragas_fields
from src.rag.raft_policy import RAFT_POLICY_VERSION


@pytest.fixture
def corpus_dir():
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "benchmark_corpus" / "financial_policy"


@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_MODE", "stub")
    monkeypatch.setenv("EMBEDDING_MODE", "mock")
    monkeypatch.setenv("MODEL_PROVIDER", "stub")
    monkeypatch.setenv("VECTOR_STORE", "in_memory")
    monkeypatch.setenv("BENCHMARK_RESULTS_DIR", str(tmp_path))


def test_raft_lm_benchmark_smoke(corpus_dir, mock_env, tmp_path):
    pytest.importorskip("langgraph")

    report = run_raft_lm_benchmark(
        corpus_dir=corpus_dir,
        out_dir=tmp_path,
        questions_limit=2,
    )
    run_dir = tmp_path / report.run_id

    assert (run_dir / "report.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "summary.md").exists()

    loaded = load_comparison_report(run_dir / "report.json")
    assert loaded.config.pipeline == "raft_lm"
    assert loaded.config.policy_version == RAFT_POLICY_VERSION
    assert all(r.pipeline_name == "raft_lm" for r in loaded.runs)
    assert loaded.raft_lm.ragas.context_precision >= 0.0


def test_full_comparison_with_delta_and_ragas_rescore(corpus_dir, mock_env, tmp_path):
    pytest.importorskip("langgraph")

    report = run_benchmark_comparison(corpus_dir=corpus_dir, out_dir=tmp_path)
    run_dir = tmp_path / report.run_id

    delta = compare_run_directories(run_dir, out_dir=tmp_path)
    assert delta.comparison_run_id == report.run_id
    assert len(delta.ragas_deltas) == 2

    delta_paths = write_comparison_delta(delta, tmp_path, run_id=report.run_id)
    assert delta_paths["comparison_delta_json"].exists()

    scored = score_saved_artifacts(run_dir)
    assert validate_ragas_fields(scored) == []
    assert scored.config.policy_version == RAFT_POLICY_VERSION

    chart = json.loads((run_dir / "comparison_chart.json").read_text(encoding="utf-8"))
    assert "standard_rag" in chart
    assert "raft_lm" in chart
