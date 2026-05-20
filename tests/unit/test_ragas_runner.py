"""Unit tests for Ragas harness stub path."""

from src.evals.benchmark_schema import BenchmarkRun, CitationRecord, new_comparison_report, RagasScores
from src.evals.ragas_runner import (
    RAGAS_METRICS_REQUIRED,
    run_ragas_eval,
    score_saved_artifacts,
    validate_ragas_fields,
)
from src.evals.report_writer import write_benchmark_report


def test_required_metrics_list():
    assert "context_precision" in RAGAS_METRICS_REQUIRED
    assert "faithfulness" in RAGAS_METRICS_REQUIRED


def test_stub_returns_scores():
    samples = [
        {
            "question": "What is CET1?",
            "answer": "8.5%",
            "context": "CET1 ratio of 8.5%",
            "ground_truth": "8.5%",
            "pipeline_name": "standard_rag",
        }
    ]
    scores = run_ragas_eval(samples)
    assert 0.0 <= scores.context_precision <= 1.0
    assert 0.0 <= scores.faithfulness <= 1.0


def test_raft_stub_boost():
    samples = [
        {
            "question": "q",
            "answer": "a",
            "context": "c",
            "ground_truth": "gt",
            "pipeline_name": "raft_lm",
        }
    ]
    scores = run_ragas_eval(samples)
    assert scores.context_precision >= 0.0


def test_score_saved_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_MODE", "stub")

    report = new_comparison_report("financial_policy_v1")
    report.run_id = "unit-test"
    report.standard.ragas = RagasScores(context_precision=0.0, faithfulness=0.0)
    report.standard.run_id = "std-1"
    report.runs = [
        BenchmarkRun(
            question_id="q1",
            question="What is CET1?",
            answer="8.5%",
            ground_truth="8.5%",
            citations=[
                CitationRecord(
                    chunk_id="c1", doc_id="d1", excerpt="CET1 ratio 8.5%", score=0.9
                )
            ],
            pipeline_name="standard_rag",
            risk_domain="financial",
        )
    ]
    write_benchmark_report(report, tmp_path, run_id="unit-test")
    run_dir = tmp_path / "unit-test"
    scored = score_saved_artifacts(run_dir)
    assert validate_ragas_fields(scored) == []
    assert scored.standard.ragas.context_precision > 0.0
