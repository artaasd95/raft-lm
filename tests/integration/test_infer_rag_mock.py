"""Integration smoke for infer CLI path."""

from src.application.infer import run_inference


def test_infer_with_mock_config():
    result = run_inference(
        "Explain CVaR",
        llm_config_path="configs/llm_mock.yaml",
        rag_pipeline="standard",
    )
    assert "answer" in result
    assert result.get("llm_provider") == "mock"
