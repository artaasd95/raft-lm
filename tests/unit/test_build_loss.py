"""Unit tests for loss factory / policy alias selection."""

import pytest

pytest.importorskip("torch", reason="PyTorch not available", exc_type=ImportError)

from src.losses.base_losses import CrossEntropyLoss
from src.losses.risk_losses import CVaRLoss, TailAwareLoss
from src.training.loss_factory import build_loss, normalize_loss_type


def _config(loss_type: str, **loss_kwargs) -> dict:
    return {
        "training": {
            "loss": {"type": loss_type, **loss_kwargs},
        }
    }


class TestBuildLoss:
    def test_ce_alias(self):
        loss = build_loss(_config("ce"))
        assert isinstance(loss, CrossEntropyLoss)

    def test_cvar_penalized_alias(self):
        loss = build_loss(_config("cvar_penalized", alpha=0.9))
        assert isinstance(loss, CVaRLoss)
        assert loss.alpha == pytest.approx(0.9)

    def test_tail_aware_alias(self):
        loss = build_loss(_config("tail_aware", alpha=0.85, tail_weight=0.3))
        assert isinstance(loss, TailAwareLoss)
        assert loss.alpha == pytest.approx(0.85)
        assert loss.tail_weight == pytest.approx(0.3)

    def test_legacy_class_names(self):
        assert isinstance(build_loss(_config("CrossEntropyLoss")), CrossEntropyLoss)
        assert isinstance(build_loss(_config("CVaRLoss")), CVaRLoss)

    def test_normalize_loss_type_policy_field(self):
        assert normalize_loss_type({"loss": "cvar_penalized"}) == "CVaRLoss"
