"""Regression: removed inference/RAG packages must not be importable."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src.rag",
        "src.llm_integration",
        "src.evals",
        "src.application.infer",
        "src.demo",
    ],
)
def test_deleted_modules_not_importable(module_name: str):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
