"""Unit tests for Ragas harness stub path."""

from src.evals.ragas_runner import RAGAS_METRICS_REQUIRED, run_ragas_eval


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
