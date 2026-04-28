"""
Risk-specific metrics.

Implements Tier A–D methods from docs/RISK-METHODS-REQUIREMENTS.md.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np

from .conventions import Array, to_numpy


def _normal_inv_cdf(p: float) -> float:
    """Inverse CDF of standard normal (quantile). Uses ``statistics.NormalDist`` for broad Python support."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    return NormalDist().inv_cdf(p)

# ---------------------------------------------------------------------------
# Loss samples (larger = worse), e.g. dollar loss magnitudes or absolute errors
# ---------------------------------------------------------------------------


def compute_var(losses: Array, alpha: float = 0.95) -> float:
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


def compute_cvar(losses: Array, alpha: float = 0.95) -> float:
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


def max_drawdown(cumulative_returns: Array) -> float:
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
    returns: Array,
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
    if sig < 1e-15:
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
    z_alpha = _normal_inv_cdf(alpha)
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
# Drawdown diagnostics, Omega, Calmar, Ulcer, moments, IR (Phase F1 roadmap)
# ---------------------------------------------------------------------------


def drawdown_series_wealth(equity_curve: Array) -> np.ndarray:
    """Per-period drawdown fraction (peak - W) / peak on strictly positive equity."""
    return _drawdown_fractions(to_numpy(equity_curve))


def _drawdown_fractions(w: np.ndarray) -> np.ndarray:
    if w.size == 0:
        raise ValueError("equity_curve must be non-empty")
    if np.any(w <= 0):
        raise ValueError("equity_curve must be strictly positive")
    peak = np.maximum.accumulate(w)
    return (peak - w) / peak


def average_drawdown_wealth(equity_curve: Array) -> float:
    """Mean drawdown fraction over the path (includes zeros at new peaks)."""
    return float(drawdown_series_wealth(equity_curve).mean())


def ulcer_index_wealth(equity_curve: Array) -> float:
    """
    Ulcer index: RMS of drawdown percentages (Peter Martin / Byron McCann).

    UI = sqrt(mean(DD_t^2)) with DD_t the fractional drawdown from running peak.
    """
    dd = drawdown_series_wealth(equity_curve)
    return float(math.sqrt((dd**2).mean()))


def ulcer_index_from_returns(returns: Array, initial_wealth: float = 1.0) -> float:
    """Ulcer index on wealth compounded from simple returns."""
    w = wealth_from_simple_returns(returns, initial_wealth)
    return ulcer_index_wealth(w)


def omega_ratio(returns: Array, threshold: float = 0.0) -> float:
    """
    Omega ratio at threshold L: sum(max(r-L,0)) / sum(max(L-r,0)).

    Returns +inf if denominator is 0 (no below-threshold outcomes).
    """
    r = to_numpy(returns)
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    gains = np.maximum(r - threshold, 0.0).sum()
    losses = np.maximum(threshold - r, 0.0).sum()
    if losses < 1e-16:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def calmar_ratio(returns: Array, periods_per_year: float = 252.0) -> float:
    """
    Calmar ratio: CAGR / max drawdown (both from compounded simple returns).

    CAGR uses (W_T/W_0)^(1/years) - 1 with years = T/periods_per_year.
    """
    r = to_numpy(returns)
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    dd = max_drawdown_from_returns(r)
    if dd < 1e-12:
        return 0.0
    w = wealth_from_simple_returns(r, 1.0)
    total_growth = float(w[-1] / w[0])
    years = len(r) / periods_per_year
    if years <= 0:
        return 0.0
    cagr = total_growth ** (1.0 / years) - 1.0
    return float(cagr / dd)


def sterling_ratio(returns: Array, periods_per_year: float = 252.0) -> float:
    """
    Sterling ratio (variant): CAGR / average drawdown on the wealth path.

    Average drawdown uses the full series including zero-DD periods at peaks.
    """
    r = to_numpy(returns)
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    w = wealth_from_simple_returns(r, 1.0)
    avg_dd = average_drawdown_wealth(w)
    if avg_dd < 1e-12:
        return 0.0
    years = len(r) / periods_per_year
    if years <= 0 or periods_per_year <= 0:
        return 0.0
    total_growth = float(w[-1] / w[0])
    cagr = total_growth ** (1.0 / years) - 1.0
    return float(cagr / avg_dd)


def skewness(returns: Array) -> float:
    """Third standardized moment m₃/s³ (sample mean and std)."""
    x = to_numpy(returns)
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 returns for skewness")
    m = x.mean()
    s = x.std(ddof=1)
    if s < 1e-16:
        return 0.0
    m3 = ((x - m) ** 3).mean()
    # unbiased-ish: use sample std in denominator cubed
    return float(m3 / (s**3))


def excess_kurtosis(returns: Array) -> float:
    """Excess kurtosis (Fisher): 0 for Gaussian."""
    x = to_numpy(returns)
    n = x.size
    if n < 4:
        raise ValueError("need at least 4 returns for kurtosis")
    m = x.mean()
    s = x.std(ddof=1)
    if s < 1e-16:
        return 0.0
    m4 = ((x - m) ** 4).mean()
    return float(m4 / (s**4) - 3.0)


def information_ratio(
    returns: Array,
    benchmark_returns: Array,
    *,
    periods_per_year: float | None = None,
    annualize: bool = False,
) -> float:
    """
    Information ratio: mean(active return) / tracking error (sample std of active).

    Active = asset - benchmark (simple return difference per period).
    """
    a = to_numpy(returns)
    b = to_numpy(benchmark_returns)
    if a.shape != b.shape:
        raise ValueError("returns and benchmark_returns must align")
    if a.size < 2:
        raise ValueError("need at least two paired returns")
    active = a - b
    mu = float(active.mean())
    te = float(active.std(ddof=1))
    if te == 0.0:
        return 0.0
    if annualize:
        if periods_per_year is None or periods_per_year <= 0:
            raise ValueError("periods_per_year required when annualize=True")
        mu *= periods_per_year
        te *= math.sqrt(periods_per_year)
    return mu / te


def portfolio_var_historical(
    asset_returns: np.ndarray,
    weights: Array,
    alpha: float = 0.95,
) -> float:
    """
    Historical simulation portfolio VaR: R w then :func:`var_historical_returns`.

    Args:
        asset_returns: shape (T, n_assets) simple returns, rows = time.
        weights: length n_assets (not required to sum to 1; scales exposure).
    """
    R = np.asarray(asset_returns, dtype=np.float64)
    w = to_numpy(weights)
    if R.ndim != 2:
        raise ValueError("asset_returns must be 2-D (T x n_assets)")
    if R.shape[1] != w.shape[0]:
        raise ValueError("weights length must match number of asset columns")
    rp = R @ w
    return var_historical_returns(rp, alpha)


def portfolio_cvar_historical(
    asset_returns: np.ndarray,
    weights: Array,
    alpha: float = 0.95,
) -> float:
    """Historical portfolio CVaR / ES on implied portfolio simple returns."""
    R = np.asarray(asset_returns, dtype=np.float64)
    w = to_numpy(weights)
    if R.ndim != 2:
        raise ValueError("asset_returns must be 2-D (T x n_assets)")
    if R.shape[1] != w.shape[0]:
        raise ValueError("weights length must match number of asset columns")
    rp = R @ w
    return cvar_historical_returns(rp, alpha)


def marginal_var_gaussian(
    weights: Array,
    cov_matrix: np.ndarray,
    alpha: float = 0.95,
) -> np.ndarray:
    """
    Marginal VaR (linear, Gaussian): ∂VaR/∂w_i ≈ z_α · (Σw)_i / σ_p.

    Returns vector same shape as weights. Portfolio VaR ≈ sum_i w_i * marginal_i
    when VaR is linear homogeneous (EL+VaR decomposition).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    w = to_numpy(weights)
    sig_p = portfolio_volatility(w, cov_matrix)
    if sig_p < 1e-16:
        return np.zeros_like(w)
    z_alpha = _normal_inv_cdf(alpha)
    return z_alpha * (cov_matrix @ w) / sig_p


def component_var_gaussian(
    weights: Array,
    cov_matrix: np.ndarray,
    alpha: float = 0.95,
) -> np.ndarray:
    """Component VaR_i = w_i · marginal_VaR_i; sums to portfolio VaR (Gaussian linear)."""
    w = to_numpy(weights)
    mv = marginal_var_gaussian(w, cov_matrix, alpha)
    return w * mv


def tail_ratio_returns(returns: Array, alpha: float = 0.95) -> float:
    """CVaR / VaR on simple returns (both as positive loss magnitudes). Tails heavier → ratio > 1."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    v = var_historical_returns(returns, alpha)
    if v < 1e-16:
        return 0.0
    return cvar_historical_returns(returns, alpha) / v


# ---------------------------------------------------------------------------
# Liquidity & microstructure (Phase F2 roadmap)
# ---------------------------------------------------------------------------


def amihud_illiquidity(returns: Array, dollar_volume: Array) -> float:
    """
    Amihud illiquidity: mean(|r_t| / dollar_volume_t) over aligned observations.

    Larger values indicate bigger price moves per traded dollar (lower liquidity).
    """
    r = to_numpy(returns)
    dv = to_numpy(dollar_volume)
    if r.shape != dv.shape:
        raise ValueError("returns and dollar_volume must align")
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    if np.any(dv <= 0):
        raise ValueError("dollar_volume must be strictly positive")
    return float(np.mean(np.abs(r) / dv))


def roll_spread_estimator(prices: Array) -> float:
    """
    Roll's implied spread from serial covariance of price changes.

    Uses S = 2 * sqrt(-Cov(Δp_t, Δp_{t-1})) when covariance is negative.
    Returns 0.0 when covariance is non-negative (estimator not identified).
    """
    p = to_numpy(prices)
    if p.size < 3:
        raise ValueError("need at least three prices")
    if np.any(p <= 0):
        raise ValueError("prices must be strictly positive")
    dp = np.diff(p)
    cov = float(np.cov(dp[1:], dp[:-1], ddof=1)[0, 1])
    if cov >= 0.0:
        return 0.0
    return float(2.0 * math.sqrt(-cov))


def volume_zscore(volume: Array, lookback: int = 20) -> np.ndarray:
    """
    Rolling z-score of volume for regime flags in thin / abnormal markets.

    For index t >= lookback-1, z_t = (v_t - mean(window)) / std(window).
    Earlier entries are NaN because the rolling window is not yet available.
    """
    v = to_numpy(volume)
    if v.size == 0:
        raise ValueError("volume must be non-empty")
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if lookback > v.size:
        raise ValueError("lookback cannot exceed series length")
    if np.any(v < 0):
        raise ValueError("volume must be non-negative")

    z = np.full(v.shape, np.nan, dtype=np.float64)
    for t in range(lookback - 1, v.size):
        w = v[t - lookback + 1 : t + 1]
        mu = float(w.mean())
        sig = float(w.std(ddof=1))
        z[t] = 0.0 if sig < 1e-15 else (v[t] - mu) / sig
    return z


# ---------------------------------------------------------------------------
# Constraint & batch CVaR (PyTorch — lazy import)
# ---------------------------------------------------------------------------


def constraint_violation_rate(
    values: Array,
    threshold: float,
) -> float:
    """Fraction of samples strictly above threshold."""
    v = to_numpy(values)
    if v.size == 0:
        raise ValueError("values must be non-empty")
    return float((v > threshold).mean())


def batch_cvar_from_losses(elementwise_losses: Any, alpha: float = 0.95) -> Any:
    """
    Differentiable (subgradient) batch CVaR: mean of worst ceil((1-α)N) per-example losses.

    Aligns with :func:`compute_cvar` on the same finite sample.
    Requires PyTorch.
    """
    import torch

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not isinstance(elementwise_losses, torch.Tensor):
        elementwise_losses = torch.as_tensor(
            to_numpy(elementwise_losses), dtype=torch.float32
        )
    losses = elementwise_losses.reshape(-1)
    n = losses.numel()
    if n == 0:
        raise ValueError("empty losses")
    k = max(1, int(math.ceil((1.0 - alpha) * n)))
    topk = torch.topk(losses, k, largest=True).values
    return topk.mean()
