"""
Unit tests for data loading and preprocessing.

Tests dataset classes and dataloader functionality.
"""

import pytest
import numpy as np
torch = pytest.importorskip("torch", reason="PyTorch not available", exc_type=ImportError)
from src.data.datasets import BaseRiskDataset, SyntheticRiskDataset
from src.data.dataloaders import create_dataloader
from src.data.adapters import (
    build_aligned_panel,
    compute_f2_liquidity_features,
    compute_f3_dependence_features,
)


class TestDatasets:
    """Test suite for dataset classes."""
    
    def test_base_risk_dataset(self):
        """Test BaseRiskDataset functionality."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([0, 1])
        ds = BaseRiskDataset(x, y, metadata={"split": "train"})
        assert len(ds) == 2
        xi, yi = ds[1]
        assert torch.equal(xi, x[1])
        assert yi.item() == 1
        assert ds.metadata["split"] == "train"
    
    def test_synthetic_risk_dataset(self):
        """Test SyntheticRiskDataset."""
        x = torch.randn(4, 3)
        y = torch.randn(4)
        ds = SyntheticRiskDataset(
            x, y, tail_index=1.7, scenario_params={"name": "stress"}
        )
        assert len(ds) == 4
        assert ds.metadata["tail_index"] == pytest.approx(1.7)
        assert ds.metadata["scenario_params"]["name"] == "stress"


class TestDataLoaders:
    """Test suite for dataloader utilities."""
    
    def test_create_dataloader(self):
        """Test dataloader creation."""
        x = torch.randn(10, 2)
        y = torch.randint(0, 2, (10,))
        ds = BaseRiskDataset(x, y)
        dl = create_dataloader(ds, batch_size=4, shuffle=False)
        batches = list(dl)
        assert len(batches) == 3
        xb, yb = batches[0]
        assert xb.shape == (4, 2)
        assert yb.shape == (4,)
    
    def test_dataloader_iteration(self):
        """Test iterating through dataloader."""
        x = torch.randn(7, 2)
        y = torch.arange(7)
        ds = BaseRiskDataset(x, y)
        dl = create_dataloader(ds, batch_size=3, shuffle=False)
        collected = []
        for _, yb in dl:
            collected.extend(yb.tolist())
        assert collected == list(range(7))


class TestDataAdapters:
    def test_build_aligned_panel(self):
        returns = np.array([[0.01, 0.02], [-0.01, 0.01], [0.03, -0.02]])
        vol = np.array([[1e6, 2e6], [1.1e6, 1.8e6], [0.9e6, 2.1e6]])
        panel = build_aligned_panel(returns, dollar_volume=vol, asset_names=["A", "B"])
        assert panel["returns"].shape == (3, 2)
        assert panel["dollar_volume"].shape == (3, 2)
        assert panel["asset_names"] == ["A", "B"]

    def test_compute_f2_features(self):
        returns = np.array(
            [[0.01, -0.005], [0.005, 0.01], [-0.02, 0.003], [0.015, -0.004]]
        )
        vol = np.array(
            [[1e6, 8e5], [1.2e6, 7.5e5], [9e5, 9e5], [1.1e6, 8.2e5]]
        )
        prices = np.array(
            [[100.0, 50.0], [100.5, 49.8], [99.8, 50.1], [100.2, 49.9]]
        )
        feats = compute_f2_liquidity_features(returns, vol, prices=prices, volume_lookback=3)
        assert "amihud_cross_section_mean" in feats
        assert "roll_spread_mean" in feats
        assert np.isfinite(feats["amihud_cross_section_mean"])

    def test_compute_f3_features(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, size=(120, 3))
        factor = returns[:, 0] + 0.2 * rng.normal(0, 0.01, size=120)
        feats = compute_f3_dependence_features(
            returns, factor_returns=factor, rolling_window=30
        )
        assert "diversification_ratio" in feats
        assert "rolling_beta_last_mean" in feats
        assert feats["diversification_ratio"] >= 0.0


# Placeholder for future tests
# TODO: Add tests for data preprocessing
# TODO: Add tests for data augmentation
# TODO: Add tests for data splitting

