"""Tests for volatility surface helpers."""

import math

import numpy as np
import pytest

from src.metrics.vol_surface import (
    atm_strike_index,
    black_scholes_call_price,
    implied_volatility_bisection,
    interpolate_iv_1d,
    iv_skew_finite_difference,
    total_implied_variance,
)


def test_black_scholes_atm_call_positive():
    c = black_scholes_call_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        vol=0.2,
        dividend_yield=0.0,
    )
    assert c > 0


def test_implied_vol_roundtrip():
    sigma = 0.25
    c = black_scholes_call_price(
        spot=100.0,
        strike=95.0,
        time_to_expiry=0.5,
        risk_free_rate=0.01,
        vol=sigma,
        dividend_yield=0.0,
    )
    iv = implied_volatility_bisection(
        c, 100.0, 95.0, 0.5, 0.01, 0.0, tol=1e-8
    )
    assert iv == pytest.approx(sigma, rel=1e-4)


def test_interpolate_iv_1d():
    k = np.array([90.0, 100.0, 110.0])
    iv = np.array([0.3, 0.25, 0.28])
    assert interpolate_iv_1d(k, iv, 100.0) == pytest.approx(0.25)


def test_atm_strike_index():
    strikes = np.array([90.0, 100.5, 110.0])
    assert atm_strike_index(100.0, strikes) == 1


def test_iv_skew_finite_difference():
    k = np.linspace(80.0, 120.0, 9)
    iv = 0.2 + 0.001 * (k - 100.0)
    skew = iv_skew_finite_difference(k, iv, forward=100.0, delta_k=5.0)
    assert skew == pytest.approx(0.001, rel=1e-2)


def test_total_implied_variance():
    assert total_implied_variance(0.25, 0.2) == pytest.approx(0.25 * 0.04)
