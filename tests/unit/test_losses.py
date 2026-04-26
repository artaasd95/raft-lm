"""Unit tests for loss functions."""

import pytest

try:
    import torch

    _ = torch.tensor([1.0])
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

requires_torch = pytest.mark.skipif(not _TORCH_OK, reason="PyTorch not available")


@requires_torch
class TestBaseLosses:
    def test_mse_loss(self):
        import torch

        from src.losses.base_losses import MSELoss

        loss = MSELoss()
        pred = torch.tensor([1.0, 2.0], requires_grad=True)
        tgt = torch.tensor([1.0, 3.0])
        out = loss(pred, tgt)
        assert out.item() == pytest.approx(0.5)
        out.backward()
        assert pred.grad is not None

    def test_cross_entropy_loss(self):
        import torch

        from src.losses.base_losses import CrossEntropyLoss

        loss = CrossEntropyLoss()
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        tgt = torch.tensor([0, 1])
        out = loss(logits, tgt)
        out.backward()
        assert logits.grad is not None


@requires_torch
class TestRiskLosses:
    def test_cvar_loss_grad(self):
        import torch

        from src.losses.risk_losses import CVaRLoss

        loss_fn = CVaRLoss(alpha=0.5, base_loss=torch.nn.MSELoss(reduction="none"))
        pred = torch.tensor([0.0, 2.0, 4.0], requires_grad=True)
        tgt = torch.tensor([0.0, 0.0, 0.0])
        out = loss_fn(pred, tgt)
        out.backward()
        assert pred.grad is not None

    def test_tail_aware_loss_positive(self):
        import torch

        from src.losses.risk_losses import TailAwareLoss

        loss_fn = TailAwareLoss(alpha=0.75, tail_weight=0.5)
        pred = torch.randn(8, 1, requires_grad=True)
        tgt = torch.randn(8, 1)
        out = loss_fn(pred, tgt)
        assert out.item() >= 0
        out.backward()
