"""
Volatility surface helpers: BSM pricing, implied vol, smile interpolation.

Requirements: docs/RISK-METHODS-REQUIREMENTS.md (Tier E).
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np

Array = Union[np.ndarray, Sequence[float]]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_call_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Black–Scholes European call price (scalar).

    Args:
        spot: S
        strike: K
        time_to_expiry: T in years
        risk_free_rate: r (continuously compounded)
        vol: implied volatility σ > 0
        dividend_yield: q (continuously compounded)
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if vol <= 0:
        raise ValueError("vol must be positive")
    if time_to_expiry <= 0:
        return max(spot - strike, 0.0)

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * vol * vol) * time_to_expiry
    ) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc_rf = math.exp(-risk_free_rate * time_to_expiry)
    disc_q = math.exp(-dividend_yield * time_to_expiry)
    return disc_q * spot * _norm_cdf(d1) - disc_rf * strike * _norm_cdf(d2)


def implied_volatility_bisection(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Invert BSM call price for implied σ using bisection."""
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive for implied vol")
    intrinsic = math.exp(-dividend_yield * time_to_expiry) * max(spot - strike, 0.0)
    if market_price < intrinsic - 1e-10:
        raise ValueError("market_price below intrinsic value")

    def price(sig: float) -> float:
        return black_scholes_call_price(
            spot, strike, time_to_expiry, risk_free_rate, sig, dividend_yield
        )

    _p_lo, p_hi = price(lo), price(hi)
    if market_price > p_hi:
        raise ValueError("market_price above BSM upper bound; widen hi")
    a, b = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        p_mid = price(mid)
        if abs(p_mid - market_price) < tol:
            return mid
        if p_mid < market_price:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def atm_strike_index(forward: float, strikes: Array) -> int:
    """Index of strike closest to forward."""
    strikes_arr = np.asarray(strikes, dtype=float)
    if strikes_arr.size == 0:
        raise ValueError("strikes must be non-empty")
    return int(np.argmin(np.abs(strikes_arr - forward)))


def interpolate_iv_1d(
    strikes: Array, implied_vols: Array, strike_eval: float
) -> float:
    """Linear interpolation of IV in strike space."""
    k = np.asarray(strikes, dtype=float)
    iv = np.asarray(implied_vols, dtype=float)
    if k.shape != iv.shape:
        raise ValueError("strikes and implied_vols must have same shape")
    if k.size < 2:
        raise ValueError("need at least two points to interpolate")
    order = np.argsort(k)
    k, iv = k[order], iv[order]
    return float(np.interp(strike_eval, k, iv))


def iv_skew_finite_difference(
    strikes: Array, implied_vols: Array, forward: float, delta_k: float
) -> float:
    """
    Central difference estimate of dIV/dK around ATM (forward).

    Uses interpolated IV at forward ± delta_k.
    """
    if delta_k <= 0:
        raise ValueError("delta_k must be positive")
    iv_up = interpolate_iv_1d(strikes, implied_vols, forward + delta_k)
    iv_dn = interpolate_iv_1d(strikes, implied_vols, forward - delta_k)
    return (iv_up - iv_dn) / (2.0 * delta_k)


def total_implied_variance(time_to_expiry: float, implied_vol: float) -> float:
    """Total variance σ²T for a slice (no variance-of-vol adjustment)."""
    if time_to_expiry < 0:
        raise ValueError("time_to_expiry must be non-negative")
    if implied_vol < 0:
        raise ValueError("implied_vol must be non-negative")
    return float(implied_vol * implied_vol * time_to_expiry)


def svi_total_variance(
    log_moneyness: Array,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """
    Raw SVI slice parameterization:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)).
    """
    if b < 0:
        raise ValueError("b must be non-negative")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not -1 < rho < 1:
        raise ValueError("rho must be in (-1, 1)")
    k = np.asarray(log_moneyness, dtype=float)
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def fit_svi_slice(log_moneyness: Array, total_variance: Array) -> dict[str, float]:
    """
    Lightweight SVI fit without external optimizers.

    Uses a robust heuristic for (m, sigma, rho) and closed-form linear fit for (a, b).
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if k.shape != w.shape:
        raise ValueError("log_moneyness and total_variance must have same shape")
    if k.size < 5:
        raise ValueError("need at least 5 points to fit SVI")
    if np.any(w <= 0):
        raise ValueError("total_variance values must be positive")

    m = float(np.median(k))
    sigma = float(max(np.std(k), 1e-3))
    rho = -0.3

    x = k - m
    basis = rho * x + np.sqrt(x * x + sigma * sigma)
    A = np.column_stack([np.ones_like(basis), basis])
    a_hat, b_hat = np.linalg.lstsq(A, w, rcond=None)[0]
    b_hat = float(max(float(b_hat), 1e-10))
    a_hat = float(max(float(a_hat), 1e-10))

    return {"a": a_hat, "b": b_hat, "rho": rho, "m": m, "sigma": sigma}


def fit_ssvi_slice(
    log_moneyness: Array,
    total_variance: Array,
    *,
    rho: float = -0.3,
) -> dict[str, float]:
    """
    Heuristic SSVI-style slice summary parameters.

    Returns theta (ATM total variance), rho, and eta inferred from slope scale.
    """
    if not -1 < rho < 1:
        raise ValueError("rho must be in (-1, 1)")
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if k.shape != w.shape:
        raise ValueError("log_moneyness and total_variance must have same shape")
    if k.size < 5:
        raise ValueError("need at least 5 points to fit SSVI")
    if np.any(w <= 0):
        raise ValueError("total_variance values must be positive")

    atm_idx = int(np.argmin(np.abs(k)))
    theta = float(w[atm_idx])
    # Approximate eta from local slope scale around ATM
    order = np.argsort(np.abs(k))
    i0, i1 = sorted(order[:2])
    if i1 == i0:
        i1 = min(i0 + 1, len(k) - 1)
    dk = float(abs(k[i1] - k[i0]))
    dw = float(abs(w[i1] - w[i0]))
    eta = float(max(dw / max(dk * math.sqrt(max(theta, 1e-12)), 1e-12), 1e-8))
    return {"theta": theta, "rho": rho, "eta": eta}


def butterfly_no_arb_check(strikes: Array, call_prices: Array, tol: float = 1e-10) -> bool:
    """
    Check static no-arbitrage in strike for a single maturity:
    call prices are non-increasing and convex in strike.
    """
    k = np.asarray(strikes, dtype=float)
    c = np.asarray(call_prices, dtype=float)
    if k.shape != c.shape:
        raise ValueError("strikes and call_prices must have same shape")
    if k.size < 3:
        raise ValueError("need at least 3 strikes")
    order = np.argsort(k)
    k = k[order]
    c = c[order]
    if np.any(np.diff(k) <= 0):
        raise ValueError("strikes must be strictly increasing after sorting")
    if np.any(np.diff(c) > tol):
        return False

    slopes = np.diff(c) / np.diff(k)
    # Convexity: slope should be non-decreasing with strike
    return bool(np.all(np.diff(slopes) >= -tol))


def calendar_no_arb_check(
    maturities: Array, total_variance_by_maturity: np.ndarray, tol: float = 1e-10
) -> bool:
    """
    Check calendar no-arbitrage proxy: total variance non-decreasing in maturity.

    Args:
        maturities: shape (n_maturities,)
        total_variance_by_maturity: shape (n_maturities, n_strikes_like_grid)
    """
    t = np.asarray(maturities, dtype=float)
    w = np.asarray(total_variance_by_maturity, dtype=float)
    if t.ndim != 1:
        raise ValueError("maturities must be 1-D")
    if w.ndim != 2:
        raise ValueError("total_variance_by_maturity must be 2-D")
    if w.shape[0] != t.shape[0]:
        raise ValueError("first dimension of total_variance_by_maturity must match maturities")
    if np.any(t <= 0):
        raise ValueError("maturities must be positive")
    order = np.argsort(t)
    w = w[order]
    return bool(np.all(np.diff(w, axis=0) >= -tol))


def dupire_local_vol(
    strikes: Array,
    maturities: Array,
    call_prices: np.ndarray,
    *,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """
    Dupire local volatility surface estimate from call price grid.

    Uses central finite differences on interior points and returns NaN on boundaries.
    """
    k = np.asarray(strikes, dtype=float)
    t = np.asarray(maturities, dtype=float)
    c = np.asarray(call_prices, dtype=float)
    if k.ndim != 1 or t.ndim != 1:
        raise ValueError("strikes and maturities must be 1-D")
    if c.shape != (t.size, k.size):
        raise ValueError("call_prices must have shape (len(maturities), len(strikes))")
    if t.size < 3 or k.size < 3:
        raise ValueError("need at least 3 maturities and 3 strikes")
    if np.any(k <= 0) or np.any(t <= 0):
        raise ValueError("strikes and maturities must be positive")

    out = np.full_like(c, np.nan, dtype=float)
    for i in range(1, t.size - 1):
        dt = t[i + 1] - t[i - 1]
        if dt <= 0:
            continue
        for j in range(1, k.size - 1):
            dk_l = k[j] - k[j - 1]
            dk_r = k[j + 1] - k[j]
            if dk_l <= 0 or dk_r <= 0:
                continue
            dC_dT = (c[i + 1, j] - c[i - 1, j]) / dt
            dC_dK = (c[i, j + 1] - c[i, j - 1]) / (k[j + 1] - k[j - 1])
            d2C_dK2 = 2.0 * (
                (c[i, j + 1] - c[i, j]) / dk_r - (c[i, j] - c[i, j - 1]) / dk_l
            ) / (dk_l + dk_r)

            denom = 0.5 * k[j] * k[j] * d2C_dK2
            numer = dC_dT + (risk_free_rate - dividend_yield) * k[j] * dC_dK + dividend_yield * c[i, j]
            if denom <= 1e-15 or numer <= 0:
                continue
            lv2 = numer / denom
            out[i, j] = math.sqrt(max(lv2, 0.0))
    return out
