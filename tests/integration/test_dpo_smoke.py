"""DPO backend smoke."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.training.backends.alignment_backend import DPOBackend


def test_dpo_smoke_stub():
    config = {
        "method": "dpo",
        "model": {"model_id": "stub"},
        "algorithm": {"beta": 0.1},
        "training": {"seed": 42},
    }
    with tempfile.TemporaryDirectory() as tmp:
        metrics = DPOBackend().run(config, Path(tmp))
        assert "dpo_loss" in metrics
        assert metrics["num_pairs"] >= 1
