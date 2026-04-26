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

    p_lo, p_hi = price(lo), price(hi)
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
