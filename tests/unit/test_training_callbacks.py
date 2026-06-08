"""Unit tests for training callbacks."""

import pytest

pytest.importorskip("torch", reason="PyTorch not available", exc_type=ImportError)

import torch

from src.logging.experiment_logger import LocalExperimentLogger
from src.training.callbacks import (
    EngineLabelAlignmentCallback,
    LossDecompositionCallback,
    build_callbacks,
)


def test_loss_decomposition_callback_logs(tmp_path):
    logger = LocalExperimentLogger(tmp_path / "run", experiment_name="test")
    cb = LossDecompositionCallback(logger, alpha=0.9)
    losses = torch.tensor([1.0, 2.0, 10.0, 3.0])
    cb.on_epoch_losses(0, losses)
    lines = (tmp_path / "run" / "logged_metrics.jsonl").read_text(encoding="utf-8").strip()
    assert "loss_cvar" in lines


def test_engine_label_alignment_callback(tmp_path):
    logger = LocalExperimentLogger(tmp_path / "run2", experiment_name="test")
    cb = EngineLabelAlignmentCallback(logger)
    preds = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    targets = torch.tensor([0, 1])
    cb.on_epoch_end(0, preds, targets, engine_label_buckets=[0, 2])
    lines = (tmp_path / "run2" / "logged_metrics.jsonl").read_text(encoding="utf-8")
    assert "engine_label_alignment_error" in lines


def test_build_callbacks_disabled():
    config = {"logging": {}}
    assert build_callbacks(config, LocalExperimentLogger("/tmp/x")) == []


def test_build_callbacks_enabled(tmp_path):
    config = {"logging": {"callbacks": True}, "training": {"loss": {"alpha": 0.95}}}
    cbs = build_callbacks(config, LocalExperimentLogger(tmp_path / "r"))
    assert len(cbs) == 2
