"""Tests for volatility surface helpers."""


import numpy as np
import pytest

from src.metrics.vol_surface import (
    atm_strike_index,
    black_scholes_call_price,
    butterfly_no_arb_check,
    calendar_no_arb_check,
    dupire_local_vol,
    fit_ssvi_slice,
    fit_svi_slice,
    implied_volatility_bisection,
    interpolate_iv_1d,
    iv_skew_finite_difference,
    svi_total_variance,
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


def test_svi_fit_and_reconstruct_positive():
    k = np.linspace(-0.2, 0.2, 9)
    w = 0.04 + 0.2 * (-0.3 * k + np.sqrt(k * k + 0.1 * 0.1))
    params = fit_svi_slice(k, w)
    w_hat = svi_total_variance(k, **params)
    assert np.all(w_hat > 0)
    assert np.mean(np.abs(w_hat - w)) < 0.05


def test_ssvi_fit_returns_expected_keys():
    k = np.linspace(-0.15, 0.15, 11)
    w = 0.03 + 0.01 * (k * k)
    params = fit_ssvi_slice(k, w)
    assert set(params.keys()) == {"theta", "rho", "eta"}
    assert params["theta"] > 0


def test_butterfly_and_calendar_no_arb_checks():
    strikes = np.array([90.0, 100.0, 110.0, 120.0])
    calls = np.array([12.0, 7.5, 4.0, 2.0])
    assert butterfly_no_arb_check(strikes, calls)

    maturities = np.array([0.25, 0.5, 1.0])
    total_var_surface = np.array(
        [
            [0.02, 0.022, 0.024],
            [0.03, 0.032, 0.034],
            [0.05, 0.052, 0.054],
        ]
    )
    assert calendar_no_arb_check(maturities, total_var_surface)


def test_dupire_local_vol_has_finite_interior():
    strikes = np.array([90.0, 100.0, 110.0, 120.0])
    maturities = np.array([0.25, 0.5, 1.0, 1.5])
    # Smooth synthetic surface with increasing maturity and convex strike profile.
    call_prices = np.array(
        [
            [12.5, 8.0, 4.7, 2.5],
            [13.5, 9.2, 5.9, 3.4],
            [15.2, 11.0, 7.8, 5.0],
            [16.8, 12.6, 9.5, 6.5],
        ]
    )
    lv = dupire_local_vol(strikes, maturities, call_prices)
    interior = lv[1:-1, 1:-1]
    assert interior.size > 0
    assert np.isfinite(interior).any()
