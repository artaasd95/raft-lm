"""Integration test for benchmark comparison and artifact loading."""

from pathlib import Path

import pytest

from src.demo.streamlit_app import load_artifacts
from src.evals.benchmark_runner import run_benchmark_comparison
from src.evals.benchmark_schema import load_comparison_report


@pytest.fixture
def corpus_dir():
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "benchmark_corpus" / "financial_policy"


def test_run_benchmark_comparison_writes_artifacts(corpus_dir, tmp_path):
    pytest.importorskip("langgraph")
    report = run_benchmark_comparison(corpus_dir=corpus_dir, out_dir=tmp_path)
    run_dir = tmp_path / report.run_id
    assert (run_dir / "report.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "comparison_chart.json").exists()

    loaded = load_comparison_report(run_dir / "report.json")
    assert loaded.standard.ragas.context_precision >= 0.0
    assert loaded.raft_lm.ragas.faithfulness >= 0.0
    assert report.standard.artifact_path

    artifacts = load_artifacts(run_dir)
    assert "error" not in artifacts
    assert artifacts["report"].corpus_id == "financial_policy_v1"
