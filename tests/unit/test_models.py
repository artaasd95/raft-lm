"""Unit tests for model architectures."""

import pytest

pytest.importorskip("torch", reason="PyTorch not available", exc_type=ImportError)

import torch

from src.models.base_models import SimpleMLP


class TestSimpleMLP:
    def test_forward_output_shape(self):
        model = SimpleMLP(input_dim=8, hidden_dim=16, output_dim=3, num_layers=2)
        x = torch.randn(5, 8)
        out = model(x)
        assert out.shape == (5, 3)

    def test_parameter_count_positive(self):
        model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2, num_layers=1)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_gradient_flow(self):
        model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2, num_layers=1)
        x = torch.randn(3, 4, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert any(p.grad is not None for p in model.parameters())

    def test_device_transfer_cpu(self):
        model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2, num_layers=1)
        model_cpu = model.to("cpu")
        x = torch.randn(2, 4, device="cpu")
        out = model_cpu(x)
        assert out.device.type == "cpu"
