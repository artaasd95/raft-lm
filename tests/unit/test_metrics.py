"""
Unit tests for metrics.

Tests correctness of task and risk metrics.
"""

import math

import numpy as np
import pytest

from src.metrics.task_metrics import accuracy, mse, mae, f1_score


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401

        _ = __import__("torch").tensor([1.0])
        return True
    except Exception:
        return False


requires_torch = pytest.mark.skipif(not _torch_ok(), reason="PyTorch not available")


class TestTaskMetrics:
    def test_accuracy(self):
        pred = np.array([[0.9, 0.1], [0.2, 0.8]])
        tgt = np.array([0, 1])
        assert accuracy(pred, tgt) == 1.0

    def test_mse(self):
        pred = np.array([1.0, 2.0])
        tgt = np.array([1.0, 3.0])
        assert mse(pred, tgt) == pytest.approx(0.5)

    def test_mae(self):
        pred = np.array([0.0, 2.0])
        tgt = np.array([1.0, 2.0])
        assert mae(pred, tgt) == pytest.approx(0.5)

    def test_f1_score(self):
        pred = np.array([1, 1, 0, 0])
        tgt = np.array([1, 0, 0, 1])
        assert f1_score(pred, tgt) == pytest.approx(0.5)


@requires_torch
class TestRiskMetrics:
    def test_var_cvar_losses_monotone(self):
        from src.metrics.risk_metrics import compute_cvar, compute_var

        L = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        assert compute_var(L, alpha=0.8) <= compute_cvar(L, alpha=0.8)

    def test_var_historical_returns_sign(self):
        from src.metrics.risk_metrics import var_historical_returns

        r = np.array([-0.10, -0.05, 0.02, 0.01])
        assert var_historical_returns(r, alpha=0.95) > 0

    def test_cvar_exceeds_var_returns(self):
        from src.metrics.risk_metrics import cvar_historical_returns, var_historical_returns

        r = np.random.default_rng(42).normal(-0.001, 0.02, 500)
        assert cvar_historical_returns(r, alpha=0.95) >= var_historical_returns(r, alpha=0.95) - 1e-9

    def test_losses_from_simple_returns_nonnegative(self):
        from src.metrics.risk_metrics import losses_from_simple_returns

        r = np.array([-0.1, 0.05])
        L = losses_from_simple_returns(r)
        assert np.all(L >= 0)
        assert L[0] == pytest.approx(0.1)
        assert L[1] == 0.0

    def test_max_drawdown_wealth(self):
        from src.metrics.risk_metrics import max_drawdown_wealth

        w = np.array([1.0, 1.2, 1.0, 1.15])
        assert max_drawdown_wealth(w) == pytest.approx((1.2 - 1.0) / 1.2)

    def test_max_drawdown_from_returns(self):
        from src.metrics.risk_metrics import max_drawdown_from_returns, max_drawdown_wealth, wealth_from_simple_returns

        r = np.array([0.0, 0.25, -0.20])
        w = wealth_from_simple_returns(r, 1.0)
        assert max_drawdown_from_returns(r) == max_drawdown_wealth(w)

    def test_sharpe_zero_vol(self):
        from src.metrics.risk_metrics import sharpe_ratio

        r = np.ones(10) * 0.01
        assert sharpe_ratio(r) == 0.0

    def test_sortino_vs_sharpe(self):
        from src.metrics.risk_metrics import sortino_ratio

        r = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        assert sortino_ratio(r, mar=0.0) >= 0.0

    def test_downside_deviation(self):
        from src.metrics.risk_metrics import downside_deviation

        r = np.array([0.1, -0.2, -0.1])
        assert downside_deviation(r, mar=0.0) > 0

    def test_realized_vol(self):
        from src.metrics.risk_metrics import realized_volatility

        r = np.array([0.01, -0.01, 0.02])
        assert realized_volatility(r) > 0

    def test_portfolio_vol_and_var(self):
        from src.metrics.risk_metrics import portfolio_var_gaussian, portfolio_volatility

        w = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.0], [0.0, 0.04]])
        sig = portfolio_volatility(w, cov)
        assert sig == pytest.approx(0.14142135, rel=1e-3)
        pvar = portfolio_var_gaussian(w, cov, alpha=0.95)
        assert pvar == pytest.approx(sig * math.sqrt(2) * math.erfinv(2 * 0.95 - 1), rel=1e-5)

    def test_beta(self):
        from src.metrics.risk_metrics import beta_vs_benchmark

        b = np.array([0.01, 0.02, -0.01])
        a = np.array([0.02, 0.04, -0.02])
        assert beta_vs_benchmark(a, b) == pytest.approx(2.0, rel=1e-5)

    def test_exposure(self):
        from src.metrics.risk_metrics import gross_exposure, net_exposure

        w = np.array([0.3, -0.2, 0.5])
        assert gross_exposure(w) == pytest.approx(1.0)
        assert net_exposure(w) == pytest.approx(0.6)

    def test_consecutive_loss_probability(self):
        from src.metrics.risk_metrics import probability_consecutive_losses

        assert probability_consecutive_losses(0.5, 3) == pytest.approx(0.125)

    def test_gambler_ruin_symmetric(self):
        from src.metrics.risk_metrics import gambler_ruin_symmetric

        assert gambler_ruin_symmetric(3, 7) == pytest.approx(0.7)

    def test_constraint_violation_rate(self):
        from src.metrics.risk_metrics import constraint_violation_rate

        v = np.array([0.1, 0.5, 0.2])
        assert constraint_violation_rate(v, 0.15) == pytest.approx(1.0 / 3.0)

    def test_batch_cvar_from_losses(self):
        import torch

        from src.metrics.risk_metrics import batch_cvar_from_losses

        losses = torch.tensor([1.0, 2.0, 10.0, 3.0], requires_grad=True)
        L = batch_cvar_from_losses(losses, alpha=0.75)
        assert L.item() == pytest.approx(10.0)
        L.backward()
        assert losses.grad is not None
