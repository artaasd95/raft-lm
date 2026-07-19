"""Unit tests for generation mock backend."""

from src.generation import build_generator
from src.generation.mock import MockGenerator


def test_mock_generator_returns_json(mock_generator):
    completion, logprob = mock_generator.generate("Assess tail risk")
    assert "risk" in completion
    assert logprob < 0


def test_build_generator_mock(tiny_mlp_config):
    tiny_mlp_config["generation"] = {"backend": "mock", "risk_level": "high"}
    gen = build_generator(tiny_mlp_config)
    assert isinstance(gen, MockGenerator)
    completion, _ = gen.generate("prompt")
    assert "high" in completion


def test_build_generator_unsupported(tiny_mlp_config):
    tiny_mlp_config["generation"] = {"backend": "vllm"}
    try:
        build_generator(tiny_mlp_config)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "mock" in str(exc)
