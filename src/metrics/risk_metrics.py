"""
Risk-specific metrics.

Implements Tier A–D methods from docs/RISK-METHODS-REQUIREMENTS.md.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch

from .conventions import Array, to_numpy

# ---------------------------------------------------------------------------
# Loss samples (larger = worse), e.g. dollar loss magnitudes or absolute errors
# ---------------------------------------------------------------------------


def compute_var(losses: Union[torch.Tensor, np.ndarray], alpha: float = 0.95) -> float:
    """
    Historical VaR on a sample of **losses** (larger = worse).

    VaR_α is the α-quantile of the loss distribution (e.g. α=0.95 → 95th percentile).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    L = to_numpy(losses)
    if L.size == 0:
        raise ValueError("losses must be non-empty")
    return float(np.quantile(L, alpha))


def compute_cvar(losses: Union[torch.Tensor, np.ndarray], alpha: float = 0.95) -> float:
    """
    Historical CVaR / ES on **losses** (larger = worse).

    Mean of the worst (1-α) fraction of losses (empirical tail mean).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    L = to_numpy(losses)
    n = L.size
    if n == 0:
        raise ValueError("losses must be non-empty")
    k = max(1, int(math.ceil((1.0 - alpha) * n)))
    sorted_desc = np.sort(L)[::-1]
    return float(sorted_desc[:k].mean())


# ---------------------------------------------------------------------------
# Simple returns — VaR / CVaR reported as positive loss magnitudes
# ---------------------------------------------------------------------------


def var_historical_returns(returns: Array, alpha: float = 0.95) -> float:
    """
    Historical VaR from **simple returns**; returns positive loss magnitude.

    q = quantile(r, 1-α) (left tail). Reported VaR = -q.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    r = to_numpy(returns)
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    q = float(np.quantile(r, 1.0 - alpha))
    return float(-q)


def cvar_historical_returns(returns: Array, alpha: float = 0.95) -> float:
    """
    Historical CVaR / ES from **simple returns**; positive loss magnitude.

    Mean of returns in the left tail (r ≤ quantile(r, 1-α)), negated.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    r = to_numpy(returns)
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    q = float(np.quantile(r, 1.0 - alpha))
    tail = r[r <= q]
    if tail.size == 0:
        return float(-q)
    return float(-tail.mean())


def losses_from_simple_returns(returns: Array) -> np.ndarray:
    """
    Map simple returns to **nonnegative** loss magnitudes: max(0, -r).

    Up days yield 0 loss in this convention; use :func:`var_historical_returns`
    for VaR on the full return distribution.
    """
    r = to_numpy(returns)
    return np.maximum(0.0, -r)


# ---------------------------------------------------------------------------
# Wealth & drawdown
# ---------------------------------------------------------------------------


def wealth_from_simple_returns(returns: Array, initial_wealth: float = 1.0) -> np.ndarray:
    """Compound simple returns to a wealth path W_t, W_0 = initial_wealth."""
    if initial_wealth <= 0:
        raise ValueError("initial_wealth must be positive")
    r = to_numpy(returns)
    factors = 1.0 + r
    if np.any(factors <= 0):
        raise ValueError("simple returns imply non-positive wealth factor (1+r<=0)")
    return initial_wealth * np.cumprod(factors)


def max_drawdown_wealth(equity_curve: Array) -> float:
    """
    Maximum drawdown on a strictly positive equity / wealth curve.

    DD_t = (peak_t - W_t) / peak_t; return max_t DD_t ∈ [0, 1] typically.
    """
    w = to_numpy(equity_curve)
    if w.size == 0:
        raise ValueError("equity_curve must be non-empty")
    if np.any(w <= 0):
        raise ValueError("equity_curve must be strictly positive")
    peak = np.maximum.accumulate(w)
    dd = (peak - w) / peak
    return float(dd.max())


def max_drawdown_from_returns(returns: Array, initial_wealth: float = 1.0) -> float:
    """Max drawdown from simple return series (compounds to wealth first)."""
    w = wealth_from_simple_returns(returns, initial_wealth)
    return max_drawdown_wealth(w)


def max_drawdown(cumulative_returns: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Backward-compatible entry point: expects a **positive equity curve** (wealth).

    .. note::
        This is **not** cumulative *sum* of returns. Use
        :func:`max_drawdown_from_returns` for simple return series.
    """
    return max_drawdown_wealth(cumulative_returns)


# ---------------------------------------------------------------------------
# Volatility, downside, Sharpe / Sortino
# ---------------------------------------------------------------------------


def realized_volatility(returns: Array, *, ddof: int = 1) -> float:
    """Sample standard deviation of returns (per-step)."""
    r = to_numpy(returns)
    if r.size < 2:
        raise ValueError("need at least two returns")
    return float(np.std(r, ddof=ddof))


def downside_deviation(returns: Array, mar: float = 0.0, *, ddof: int = 0) -> float:
    """Downside deviation around minimum acceptable return MAR (Sortino denominator)."""
    r = to_numpy(returns)
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    below = np.minimum(r - mar, 0.0)
    if below.size < 2 and ddof == 1:
        ddof = 0
    denom = max(len(below) - ddof, 1)
    return float(np.sqrt((below**2).sum() / denom))


def semi_variance(returns: Array, mar: float = 0.0) -> float:
    """Average squared negative deviation from MAR."""
    r = to_numpy(returns)
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    below = np.minimum(r - mar, 0.0)
    return float((below**2).mean())


def sharpe_ratio(
    returns: Union[torch.Tensor, np.ndarray],
    risk_free_rate: float = 0.0,
    *,
    periods_per_year: float | None = None,
    annualize: bool = False,
) -> float:
    """
    Sharpe ratio using sample mean and std of (returns - risk_free_rate).

    If annualize=True, scales mean by periods_per_year and vol by sqrt(periods_per_year).
    """
    r = to_numpy(returns)
    if r.size < 2:
        raise ValueError("need at least two returns")
    xs = r - risk_free_rate
    mu = float(xs.mean())
    sig = float(xs.std(ddof=1))
    if sig == 0.0:
        return 0.0
    if annualize:
        if periods_per_year is None or periods_per_year <= 0:
            raise ValueError("periods_per_year required when annualize=True")
        mu *= periods_per_year
        sig *= math.sqrt(periods_per_year)
    return mu / sig


def sortino_ratio(
    returns: Array,
    mar: float = 0.0,
    risk_free_rate: float = 0.0,
    *,
    periods_per_year: float | None = None,
    annualize: bool = False,
) -> float:
    """Sortino: excess mean over downside deviation (MAR)."""
    r = to_numpy(returns)
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    excess = r - risk_free_rate
    mu = float(excess.mean())
    ddown = downside_deviation(r, mar=mar, ddof=0)
    if ddown == 0.0:
        return 0.0
    if annualize:
        if periods_per_year is None or periods_per_year <= 0:
            raise ValueError("periods_per_year required when annualize=True")
        mu *= periods_per_year
        ddown *= math.sqrt(periods_per_year)
    return mu / ddown


# ---------------------------------------------------------------------------
# Portfolio / position (linear)
# ---------------------------------------------------------------------------


def portfolio_variance(weights: Array, cov_matrix: np.ndarray) -> float:
    """w^T Σ w."""
    w = to_numpy(weights)
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be square")
    if w.shape[0] != cov_matrix.shape[0]:
        raise ValueError("weights length must match cov_matrix dimension")
    return float(w @ cov_matrix @ w)


def portfolio_volatility(weights: Array, cov_matrix: np.ndarray) -> float:
    """sqrt(w^T Σ w)."""
    return float(math.sqrt(max(portfolio_variance(weights, cov_matrix), 0.0)))


def portfolio_var_gaussian(
    weights: Array,
    cov_matrix: np.ndarray,
    alpha: float = 0.95,
) -> float:
    """
    Parametric (Gaussian) portfolio VaR as **positive loss magnitude** at horizon of Σ.

    VaR ≈ z_α · σ_p where σ_p is portfolio vol of simple returns over the period
    and z_α = Φ^{-1}(α) (right-tail Gaussian quantile for loss direction).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    sig_p = portfolio_volatility(weights, cov_matrix)
    z_alpha = math.sqrt(2.0) * math.erfinv(2.0 * alpha - 1.0)
    return float(z_alpha * sig_p)


def gross_exposure(weights: Array) -> float:
    return float(np.abs(to_numpy(weights)).sum())


def net_exposure(weights: Array) -> float:
    return float(to_numpy(weights).sum())


def concentration_herfindahl(weights: Array) -> float:
    """Sum of squared weights (concentration)."""
    w = to_numpy(weights)
    return float((w**2).sum())


def beta_vs_benchmark(asset_returns: Array, benchmark_returns: Array) -> float:
    """Cov(r, rb) / Var(rb)."""
    a = to_numpy(asset_returns)
    b = to_numpy(benchmark_returns)
    if a.shape != b.shape:
        raise ValueError("asset_returns and benchmark_returns must align")
    if a.size < 2:
        raise ValueError("need at least two paired returns")
    vb = float(np.var(b, ddof=1))
    if vb == 0.0:
        return 0.0
    return float(np.cov(a, b, ddof=1)[0, 1] / vb)


# ---------------------------------------------------------------------------
# Ruin / survival (stylized)
# ---------------------------------------------------------------------------


def probability_consecutive_losses(loss_probability: float, streak_length: int) -> float:
    """
    P(at least one run of `streak_length` consecutive losses) is **bounded** by p^k
    for i.i.d. loss events with probability p (stylized; ignores overlap).
    """
    if not 0 <= loss_probability <= 1:
        raise ValueError("loss_probability must be in [0, 1]")
    if streak_length < 1:
        raise ValueError("streak_length must be >= 1")
    return float(loss_probability**streak_length)


def gambler_ruin_symmetric(start_capital: int, opponent_capital: int) -> float:
    """
    Ruin probability for simple symmetric random walk (p=0.5) until one party is broke.

    P(ruin starting with i) = (N-i)/N where N = start + opponent, i = start.
    """
    if start_capital < 0 or opponent_capital < 0:
        raise ValueError("capitals must be non-negative")
    n = start_capital + opponent_capital
    if n == 0:
        return 1.0
    return float((n - start_capital) / n)


def risk_of_ruin_gbm_log_barrier_approx(
    drift: float,
    vol: float,
    *,
    log_distance_to_ruin: float = 1.0,
) -> float:
    """
    Heuristic: exp(-2 * μ * d / σ²) for GBM-like intuition (μ drift, σ vol, d log-distance).

    **Not** a substitute for Monte Carlo or rigorous first-passage treatment.
    """
    if vol <= 0:
        raise ValueError("vol must be positive")
    if log_distance_to_ruin <= 0:
        raise ValueError("log_distance_to_ruin must be positive")
    return float(math.exp(-2.0 * drift * log_distance_to_ruin / (vol * vol)))


# ---------------------------------------------------------------------------
# Constraint & batch CVaR (PyTorch)
# ---------------------------------------------------------------------------


def constraint_violation_rate(
    values: Union[torch.Tensor, np.ndarray],
    threshold: float,
) -> float:
    """Fraction of samples strictly above threshold."""
    v = to_numpy(values)
    if v.size == 0:
        raise ValueError("values must be non-empty")
    return float((v > threshold).mean())


def batch_cvar_from_losses(elementwise_losses: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """
    Differentiable (subgradient) batch CVaR: mean of worst ceil((1-α)N) per-example losses.

    Aligns with :func:`compute_cvar` on the same finite sample.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    losses = elementwise_losses.reshape(-1)
    n = losses.numel()
    if n == 0:
        raise ValueError("empty losses")
    k = max(1, int(math.ceil((1.0 - alpha) * n)))
    topk = torch.topk(losses, k, largest=True).values
    return topk.mean()
