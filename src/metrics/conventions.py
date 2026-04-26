"""
Shared conventions for risk metrics and losses.

See docs/RISK-METHODS-REQUIREMENTS.md for full definitions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Union

import numpy as np

Array = Union[np.ndarray, Any]


class ReturnKind(str, Enum):
    """How a return series should be interpreted."""

    SIMPLE = "simple"
    LOG = "log"


def to_numpy(x: Array) -> np.ndarray:
    """Convert tensor or array to 1D float64 numpy (flattened if needed)."""
    if hasattr(x, "detach") and callable(x.detach):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float64)
    return a.reshape(-1)


def annualize_volatility(
    vol_per_step: float, periods_per_year: float, *, kind: str = "simple"
) -> float:
    """
    Scale per-step volatility to annualized.

    For simple returns, use sqrt scaling; for log returns, same sqrt rule applies
    to per-step log-return std when time steps are uniform.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    _ = kind  # reserved for future log-return nuance
    return float(vol_per_step * np.sqrt(periods_per_year))
