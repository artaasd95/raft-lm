"""
Metrics for evaluating risk-aware models.

See docs/RISK-METHODS-REQUIREMENTS.md for definitions and naming conventions.

Torch-dependent risk metrics are loaded lazily so ``import src.metrics.vol_surface``
works in environments where PyTorch is unavailable.
"""

from __future__ import annotations

from typing import Any, FrozenSet

from .conventions import ReturnKind, annualize_volatility, to_numpy
from .task_metrics import accuracy, mae, mse, f1_score
from .vol_surface import (
    atm_strike_index,
    black_scholes_call_price,
    implied_volatility_bisection,
    interpolate_iv_1d,
    iv_skew_finite_difference,
    total_implied_variance,
)

_LAZY_RISK: FrozenSet[str] = frozenset(
    {
        "batch_cvar_from_losses",
        "beta_vs_benchmark",
        "compute_cvar",
        "compute_var",
        "concentration_herfindahl",
        "constraint_violation_rate",
        "cvar_historical_returns",
        "downside_deviation",
        "gambler_ruin_symmetric",
        "gross_exposure",
        "losses_from_simple_returns",
        "max_drawdown",
        "max_drawdown_from_returns",
        "max_drawdown_wealth",
        "net_exposure",
        "portfolio_var_gaussian",
        "portfolio_variance",
        "portfolio_volatility",
        "probability_consecutive_losses",
        "realized_volatility",
        "risk_of_ruin_gbm_log_barrier_approx",
        "semi_variance",
        "sharpe_ratio",
        "sortino_ratio",
        "var_historical_returns",
        "wealth_from_simple_returns",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_RISK:
        from . import risk_metrics

        return getattr(risk_metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ReturnKind",
    "annualize_volatility",
    "to_numpy",
    "accuracy",
    "mae",
    "mse",
    "f1_score",
    "atm_strike_index",
    "black_scholes_call_price",
    "implied_volatility_bisection",
    "interpolate_iv_1d",
    "iv_skew_finite_difference",
    "total_implied_variance",
    *sorted(_LAZY_RISK),
]
