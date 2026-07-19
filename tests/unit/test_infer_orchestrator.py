"""Inference orchestrator tests."""

from src.application.infer import run_inference


def test_infer_mock_path():
    result = run_inference("What is CVaR?", rag_pipeline="standard")
    assert "query" in result
    assert "answer" in result
