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


class TestRiskMetrics:
    """NumPy-only risk metrics (no PyTorch import required)."""

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
        from statistics import NormalDist

        pvar = portfolio_var_gaussian(w, cov, alpha=0.95)
        z = NormalDist().inv_cdf(0.95)
        assert pvar == pytest.approx(sig * z, rel=1e-5)

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
        assert constraint_violation_rate(v, 0.15) == pytest.approx(2.0 / 3.0)


@requires_torch
class TestBatchCvarTorch:
    def test_batch_cvar_from_losses(self):
        import torch

        from src.metrics.risk_metrics import batch_cvar_from_losses

        losses = torch.tensor([1.0, 2.0, 10.0, 3.0], requires_grad=True)
        L = batch_cvar_from_losses(losses, alpha=0.75)
        assert L.item() == pytest.approx(10.0)
        L.backward()
        assert losses.grad is not None


class TestRoadmapF1Metrics:
    """Phase F1 from docs/RISK-METRICS-ROADMAP.md."""

    def test_omega_ratio(self):
        from src.metrics.risk_metrics import omega_ratio

        r = np.array([-0.01, 0.02, -0.005, 0.03])
        o = omega_ratio(r, threshold=0.0)
        gains = 0.02 + 0.03
        losses = 0.01 + 0.005
        assert o == pytest.approx(gains / losses)

    def test_ulcer_nonnegative(self):
        from src.metrics.risk_metrics import ulcer_index_from_returns

        r = np.array([0.1, -0.05, 0.02])
        assert ulcer_index_from_returns(r) >= 0.0

    def test_calmar_positive_on_trend(self):
        from src.metrics.risk_metrics import calmar_ratio

        rng = np.random.default_rng(3)
        r = np.concatenate([rng.normal(0.002, 0.01, 200), rng.normal(-0.001, 0.015, 52)])
        assert calmar_ratio(r, periods_per_year=252.0) > 0.0

    def test_information_ratio(self):
        from src.metrics.risk_metrics import information_ratio

        b = np.zeros(20)
        a = np.full(20, 0.001)
        assert information_ratio(a, b) > 0.0

    def test_portfolio_var_historical_matches_manual(self):
        from src.metrics.risk_metrics import portfolio_var_historical, var_historical_returns

        rng = np.random.default_rng(0)
        R = rng.normal(0, 0.01, (500, 3))
        w = np.array([0.2, 0.3, 0.5])
        v1 = portfolio_var_historical(R, w, alpha=0.95)
        v2 = var_historical_returns(R @ w, alpha=0.95)
        assert v1 == pytest.approx(v2)

    def test_component_var_sums_to_portfolio_var(self):
        from src.metrics.risk_metrics import (
            component_var_gaussian,
            portfolio_var_gaussian,
        )

        w = np.array([0.3, 0.7])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        alpha = 0.95
        comp = component_var_gaussian(w, cov, alpha)
        total = portfolio_var_gaussian(w, cov, alpha)
        assert comp.sum() == pytest.approx(total, rel=1e-5)

    def test_tail_ratio_at_least_one(self):
        from src.metrics.risk_metrics import tail_ratio_returns

        rng = np.random.default_rng(1)
        r = rng.standard_t(3, size=2000) * 0.01
        assert tail_ratio_returns(r, alpha=0.95) >= 1.0 - 1e-6

    def test_skew_and_kurtosis_finite(self):
        from src.metrics.risk_metrics import excess_kurtosis, skewness

        r = np.random.default_rng(2).normal(0, 1, 100)
        assert math.isfinite(skewness(r))
        assert math.isfinite(excess_kurtosis(r))


class TestRoadmapF2Metrics:
    """Phase F2 from docs/RISK-METRICS-ROADMAP.md."""

    def test_amihud_illiquidity_matches_definition(self):
        from src.metrics.risk_metrics import amihud_illiquidity

        r = np.array([0.01, -0.02, 0.015])
        dv = np.array([1_000_000.0, 2_000_000.0, 500_000.0])
        expected = np.mean(np.abs(r) / dv)
        assert amihud_illiquidity(r, dv) == pytest.approx(expected)

    def test_roll_spread_estimator_nonnegative(self):
        from src.metrics.risk_metrics import roll_spread_estimator

        prices = np.array([100.0, 100.02, 99.99, 100.03, 99.98, 100.01, 99.97, 100.0])
        spread = roll_spread_estimator(prices)
        assert spread >= 0.0

    def test_volume_zscore_rolling_behavior(self):
        from src.metrics.risk_metrics import volume_zscore

        volume = np.array([100.0, 110.0, 120.0, 130.0, 200.0])
        z = volume_zscore(volume, lookback=3)
        assert np.isnan(z[0])
        assert np.isnan(z[1])
        assert z.shape == volume.shape
        assert math.isfinite(float(z[-1]))


class TestRoadmapF3Metrics:
    """Phase F3 from docs/RISK-METRICS-ROADMAP.md."""

    def test_rolling_correlation_shape_and_bounds(self):
        from src.metrics.risk_metrics import rolling_correlation

        x = np.array([0.01, 0.02, -0.01, 0.03, 0.01, -0.02])
        y = np.array([0.005, 0.01, -0.005, 0.02, 0.007, -0.01])
        rc = rolling_correlation(x, y, window=3)
        assert rc.shape == x.shape
        assert np.isnan(rc[0]) and np.isnan(rc[1])
        assert np.all(np.abs(rc[2:]) <= 1.0 + 1e-12)

    def test_rolling_beta_positive_for_scaled_series(self):
        from src.metrics.risk_metrics import rolling_beta

        f = np.array([0.01, -0.005, 0.015, 0.0, 0.02, -0.01])
        a = 1.5 * f
        rb = rolling_beta(a, f, window=4)
        assert np.isnan(rb[0]) and np.isnan(rb[1]) and np.isnan(rb[2])
        assert rb[-1] == pytest.approx(1.5, rel=1e-6)

    def test_sample_copula_tail_dependence_returns_unit_interval(self):
        from src.metrics.risk_metrics import sample_copula_tail_dependence

        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, 300)
        y = 0.6 * x + 0.4 * rng.normal(0, 1, 300)
        lam_l, lam_u = sample_copula_tail_dependence(x, y, quantile=0.95)
        assert 0.0 <= lam_l <= 1.0
        assert 0.0 <= lam_u <= 1.0

    def test_diversification_ratio_above_one_for_imperfect_correlation(self):
        from src.metrics.risk_metrics import diversification_ratio

        w = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        dr = diversification_ratio(w, cov)
        assert dr > 1.0
