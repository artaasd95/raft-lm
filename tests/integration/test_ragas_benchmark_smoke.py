"""Integration smoke: ingestion -> retrieve -> generate -> Ragas -> save (mock only)."""

from __future__ import annotations

import json

import pytest

from src.evals.benchmark_runner import run_benchmark_comparison
from src.evals.benchmark_schema import REQUIRED_REPORT_FIELDS, load_comparison_report
from src.evals.ragas_runner import RAGAS_METRICS_REQUIRED, score_saved_artifacts, validate_ragas_fields
from src.evals.report_writer import validate_report_schema


@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_MODE", "mock")
    monkeypatch.setenv("EMBEDDING_MODE", "mock")
    monkeypatch.setenv("EMBEDDING_MODEL", "deterministic-stub")
    monkeypatch.setenv("MODEL_PROVIDER", "stub")
    monkeypatch.setenv("VECTOR_STORE", "in_memory")
    monkeypatch.setenv("BENCHMARK_RESULTS_DIR", str(tmp_path))
    return tmp_path


def test_ragas_benchmark_smoke_end_to_end(mock_env):
    pytest.importorskip("langgraph")

    report = run_benchmark_comparison(out_dir=mock_env)
    run_dir = mock_env / report.run_id

    for name in ("report.json", "metrics.csv", "summary.md", "comparison_chart.json"):
        assert (run_dir / name).exists(), f"missing artifact: {name}"

    loaded = load_comparison_report(run_dir / "report.json")
    missing_schema = validate_report_schema(loaded)
    assert missing_schema == [], f"missing schema fields: {missing_schema}"

    raw = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    for field in REQUIRED_REPORT_FIELDS:
        assert field in raw

    for metric in RAGAS_METRICS_REQUIRED:
        assert hasattr(loaded.standard.ragas, metric)
        assert getattr(loaded.standard.ragas, metric) is not None
        assert hasattr(loaded.raft_lm.ragas, metric)
        assert getattr(loaded.raft_lm.ragas, metric) is not None

    missing_ragas = validate_ragas_fields(loaded)
    assert missing_ragas == [], f"missing Ragas fields: {missing_ragas}"

    assert loaded.runs
    first = loaded.runs[0]
    assert first.severity in ("none", "low", "medium", "high", "critical")
    assert first.severity_bucket in ("legal", "financial", "compliance", "operational")
    assert first.ragas_context_precision is not None
    assert first.ragas_faithfulness is not None


def test_score_saved_artifacts_backfills_ragas(mock_env):
    pytest.importorskip("langgraph")

    report = run_benchmark_comparison(out_dir=mock_env)
    run_dir = mock_env / report.run_id

    report_path = run_dir / "report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data["standard"]["ragas"] = {"context_precision": 0.0, "faithfulness": 0.0}
    data["raft_lm"]["ragas"] = {"context_precision": 0.0, "faithfulness": 0.0}
    report_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    rescored = score_saved_artifacts(run_dir)
    assert rescored.standard.ragas.context_precision > 0.0 or rescored.standard.ragas.faithfulness > 0.0
    assert validate_ragas_fields(rescored) == []


def test_metrics_csv_includes_severity_buckets(mock_env):
    pytest.importorskip("langgraph")

    report = run_benchmark_comparison(out_dir=mock_env)
    csv_path = mock_env / report.run_id / "metrics.csv"
    text = csv_path.read_text(encoding="utf-8")
    for col in (
        "severity_legal",
        "severity_financial",
        "severity_compliance",
        "severity_operational",
        "context_precision",
        "faithfulness",
    ):
        assert col in text
