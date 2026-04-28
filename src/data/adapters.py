"""
Data adapters for liquidity/dependence metric integration.

These helpers standardize panel shapes and compute feature blocks used by
Phase F2/F3 roadmap items.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.metrics.risk_metrics import (
    amihud_illiquidity,
    diversification_ratio,
    rolling_beta,
    rolling_correlation,
    roll_spread_estimator,
    sample_copula_tail_dependence,
    volume_zscore,
)


def build_aligned_panel(
    returns: np.ndarray,
    *,
    dollar_volume: np.ndarray | None = None,
    prices: np.ndarray | None = None,
    factor_returns: np.ndarray | None = None,
    asset_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate and return a time-aligned multi-asset panel dictionary.

    Expected shapes:
      - returns: (T, N)
      - dollar_volume/prices: (T, N) when provided
      - factor_returns: (T,) or (T, K) when provided
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns must be 2-D (T, N)")
    t, n = r.shape
    if t < 2 or n < 1:
        raise ValueError("returns panel must have at least 2 rows and 1 asset")

    out: dict[str, Any] = {"returns": r}
    if asset_names is not None:
        if len(asset_names) != n:
            raise ValueError("asset_names length must match number of assets")
        out["asset_names"] = list(asset_names)

    if dollar_volume is not None:
        dv = np.asarray(dollar_volume, dtype=float)
        if dv.shape != (t, n):
            raise ValueError("dollar_volume must match returns shape (T, N)")
        if np.any(dv <= 0):
            raise ValueError("dollar_volume must be strictly positive")
        out["dollar_volume"] = dv

    if prices is not None:
        p = np.asarray(prices, dtype=float)
        if p.shape != (t, n):
            raise ValueError("prices must match returns shape (T, N)")
        if np.any(p <= 0):
            raise ValueError("prices must be strictly positive")
        out["prices"] = p

    if factor_returns is not None:
        f = np.asarray(factor_returns, dtype=float)
        if f.ndim == 1:
            if f.shape[0] != t:
                raise ValueError("factor_returns length must match T")
        elif f.ndim == 2:
            if f.shape[0] != t:
                raise ValueError("factor_returns first dimension must match T")
        else:
            raise ValueError("factor_returns must be 1-D or 2-D")
        out["factor_returns"] = f

    return out


def compute_f2_liquidity_features(
    returns: np.ndarray,
    dollar_volume: np.ndarray,
    *,
    prices: np.ndarray | None = None,
    volume_lookback: int = 20,
) -> dict[str, Any]:
    """Compute F2 liquidity/microstructure features for each asset and cross-section."""
    r = np.asarray(returns, dtype=float)
    dv = np.asarray(dollar_volume, dtype=float)
    if r.ndim != 2 or dv.ndim != 2:
        raise ValueError("returns and dollar_volume must both be 2-D")
    if r.shape != dv.shape:
        raise ValueError("returns and dollar_volume must have same shape")
    t, n = r.shape
    if t < 2:
        raise ValueError("need at least 2 rows")

    amihud = np.array([amihud_illiquidity(r[:, i], dv[:, i]) for i in range(n)], dtype=float)
    vol_z = np.vstack([volume_zscore(dv[:, i], lookback=min(volume_lookback, t)) for i in range(n)]).T

    out: dict[str, Any] = {
        "amihud_by_asset": amihud,
        "amihud_cross_section_mean": float(amihud.mean()),
        "volume_zscore_last_by_asset": vol_z[-1, :],
        "volume_zscore_last_mean": float(np.nanmean(vol_z[-1, :])),
    }

    if prices is not None:
        p = np.asarray(prices, dtype=float)
        if p.shape != r.shape:
            raise ValueError("prices must match returns shape")
        spreads = np.array([roll_spread_estimator(p[:, i]) for i in range(n)], dtype=float)
        out["roll_spread_by_asset"] = spreads
        out["roll_spread_mean"] = float(spreads.mean())
    return out


def compute_f3_dependence_features(
    returns: np.ndarray,
    *,
    factor_returns: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    rolling_window: int = 60,
    tail_quantile: float = 0.95,
) -> dict[str, Any]:
    """Compute F3 dependence/systemic features on aligned panels."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns must be 2-D (T, N)")
    t, n = r.shape
    if t < 2 or n < 1:
        raise ValueError("returns panel must be non-empty")

    w = np.asarray(weights, dtype=float) if weights is not None else np.ones(n, dtype=float) / n
    if w.shape != (n,):
        raise ValueError("weights must be shape (N,)")

    cov = np.cov(r, rowvar=False, ddof=1)
    div_ratio = diversification_ratio(w, cov)

    # Pairwise rolling correlation mean on latest window value.
    rw = min(rolling_window, t)
    pair_vals = []
    tail_vals_l = []
    tail_vals_u = []
    for i in range(n):
        for j in range(i + 1, n):
            rc = rolling_correlation(r[:, i], r[:, j], window=rw)
            if np.isfinite(rc[-1]):
                pair_vals.append(float(rc[-1]))
            lam_l, lam_u = sample_copula_tail_dependence(r[:, i], r[:, j], quantile=tail_quantile)
            tail_vals_l.append(lam_l)
            tail_vals_u.append(lam_u)

    out: dict[str, Any] = {
        "diversification_ratio": float(div_ratio),
        "rolling_correlation_last_mean": float(np.mean(pair_vals)) if pair_vals else 0.0,
        "tail_dependence_lambda_lower_mean": float(np.mean(tail_vals_l)) if tail_vals_l else 0.0,
        "tail_dependence_lambda_upper_mean": float(np.mean(tail_vals_u)) if tail_vals_u else 0.0,
    }

    if factor_returns is not None:
        f = np.asarray(factor_returns, dtype=float)
        if f.ndim == 2:
            f = f[:, 0]
        if f.ndim != 1 or f.shape[0] != t:
            raise ValueError("factor_returns must be shape (T,) or (T, K)")
        betas = np.array([rolling_beta(r[:, i], f, window=rw)[-1] for i in range(n)], dtype=float)
        out["rolling_beta_last_by_asset"] = betas
        out["rolling_beta_last_mean"] = float(np.nanmean(betas))

    return out
