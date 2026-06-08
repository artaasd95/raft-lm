"""Thin wrappers over src/metrics for LLM tool calling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.metrics.risk_metrics import (
    compute_cvar,
    compute_var,
    max_drawdown_from_returns,
    realized_volatility,
)


def _result(value: float, units: str, provenance: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "value": value,
        "units": units,
        "provenance": provenance,
    }
    out.update(extra)
    return out


def _clean_returns(returns: List[float]) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return arr
    if np.any(np.isnan(arr)):
        arr = arr[~np.isnan(arr)]
    return arr


def compute_cvar_tool(returns: List[float], alpha: float = 0.95) -> Dict[str, Any]:
    """CVaR on loss magnitudes derived from simple returns."""
    r = _clean_returns(returns)
    if r.size == 0:
        return _result(0.0, "loss_fraction", "src.metrics.risk_metrics.compute_cvar", alpha=alpha)
    losses = np.maximum(-r, 0.0)
    if np.all(losses == 0):
        return _result(0.0, "loss_fraction", "src.metrics.risk_metrics.compute_cvar", alpha=alpha)
    return _result(
        compute_cvar(losses, alpha=alpha),
        "loss_fraction",
        "src.metrics.risk_metrics.compute_cvar",
        alpha=alpha,
        var=compute_var(losses, alpha=alpha),
    )


def compute_drawdown_tool(returns: List[float]) -> Dict[str, Any]:
    """Maximum drawdown from simple return series."""
    r = _clean_returns(returns)
    if r.size == 0:
        return _result(0.0, "fraction", "src.metrics.risk_metrics.max_drawdown_from_returns")
    return _result(
        max_drawdown_from_returns(r),
        "fraction",
        "src.metrics.risk_metrics.max_drawdown_from_returns",
    )


def compute_volatility_tool(returns: List[float], annualize: bool = False) -> Dict[str, Any]:
    """Realized volatility of return series."""
    r = _clean_returns(returns)
    if r.size == 0:
        return _result(0.0, "per_period", "src.metrics.risk_metrics.realized_volatility")
    vol = realized_volatility(r)
    if annualize:
        vol *= np.sqrt(252.0)
        units = "annualized"
    else:
        units = "per_period"
    return _result(vol, units, "src.metrics.risk_metrics.realized_volatility", annualize=annualize)


def compute_position_size_tool(
    returns: List[float],
    risk_budget: float = 0.02,
    max_leverage: float = 1.0,
) -> Dict[str, Any]:
    """Vol-target position size: risk_budget / realized_vol, capped by max_leverage."""
    r = _clean_returns(returns)
    if r.size == 0 or risk_budget <= 0:
        return _result(0.0, "weight", "src.tools.risk_tools.compute_position_size_tool")
    vol = realized_volatility(r)
    if vol <= 0 or np.isnan(vol):
        return _result(0.0, "weight", "src.tools.risk_tools.compute_position_size_tool")
    size = min(max_leverage, risk_budget / vol)
    return _result(
        float(size),
        "weight",
        "src.tools.risk_tools.compute_position_size_tool",
        risk_budget=risk_budget,
        realized_vol=vol,
        max_leverage=max_leverage,
    )
