"""Unit tests for unified model loaders."""

import pytest
import torch

from src.models.base_models import SimpleMLP
from src.models.loaders.unified import load_from_hub_or_local, load_pytorch_checkpoint


def test_load_pytorch_checkpoint(tmp_path):
    model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2, dropout=0.0)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    fresh = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2, dropout=0.0)
    loaded = load_pytorch_checkpoint(ckpt, fresh)
    model.eval()
    loaded.module.eval()
    x = torch.randn(2, 4)
    assert torch.allclose(model(x), loaded.module(x))


def test_load_from_hub_or_local_directory(tmp_path):
    model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2)
    ckpt = tmp_path / "weights.pt"
    torch.save(model.state_dict(), ckpt)

    fresh = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2)
    loaded = load_from_hub_or_local(str(tmp_path), model=fresh)
    assert loaded.source == "pytorch_checkpoint"


def test_missing_local_path_raises(tmp_path):
    missing = tmp_path / "no_such_dir"
    with pytest.raises(FileNotFoundError):
        load_from_hub_or_local(str(missing), model=SimpleMLP(4, 8, 2))
