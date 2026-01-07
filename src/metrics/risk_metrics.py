"""
Risk-specific metrics.

Metrics for evaluating tail risk, CVaR, drawdown, and constraint satisfaction.
"""

import torch
import numpy as np
from typing import Optional, Union


def compute_cvar(
    losses: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.95
) -> float:
    """
    Compute Conditional Value at Risk (CVaR).
    
    CVaR is the expected loss in the worst (1-alpha) cases.
    
    Args:
        losses: Loss values (higher = worse)
        alpha: Confidence level (e.g., 0.95 for 95% CVaR)
        
    Returns:
        CVaR value
    """
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy()
    
    # Sort losses in descending order
    sorted_losses = np.sort(losses)[::-1]
    
    # Find VaR (Value at Risk) - the alpha quantile
    n = len(sorted_losses)
    var_index = int(np.ceil((1 - alpha) * n))
    
    # CVaR is the mean of losses beyond VaR
    cvar = sorted_losses[:var_index].mean()
    
    return float(cvar)


def compute_var(
    losses: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.95
) -> float:
    """
    Compute Value at Risk (VaR).
    
    VaR is the alpha-quantile of the loss distribution.
    
    Args:
        losses: Loss values
        alpha: Confidence level
        
    Returns:
        VaR value
    """
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy()
    
    var = np.quantile(losses, 1 - alpha)
    return float(var)


def sharpe_ratio(
    returns: Union[torch.Tensor, np.ndarray],
    risk_free_rate: float = 0.0
) -> float:
    """
    Compute Sharpe ratio.
    
    Measures risk-adjusted return.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate for comparison
        
    Returns:
        Sharpe ratio
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    
    excess_returns = returns - risk_free_rate
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    
    if std_return == 0:
        return 0.0
    
    return float(mean_return / std_return)


def max_drawdown(cumulative_returns: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute maximum drawdown.
    
    Maximum peak-to-trough decline.
    
    Args:
        cumulative_returns: Cumulative returns over time
        
    Returns:
        Maximum drawdown (positive value)
    """
    if isinstance(cumulative_returns, torch.Tensor):
        cumulative_returns = cumulative_returns.detach().cpu().numpy()
    
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_dd = np.abs(drawdown.min())
    
    return float(max_dd)


def constraint_violation_rate(
    values: Union[torch.Tensor, np.ndarray],
    threshold: float
) -> float:
    """
    Compute the rate of constraint violations.
    
    Args:
        values: Values to check (e.g., CVaR, risk levels)
        threshold: Constraint threshold
        
    Returns:
        Fraction of samples violating the constraint
    """
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    
    violations = (values > threshold).sum()
    total = len(values)
    
    return float(violations / total)


# Placeholder for additional risk metrics
# TODO: Add Sortino ratio
# TODO: Add tail event precision/recall
# TODO: Add expected shortfall
# TODO: Add risk-adjusted performance metrics
# TODO: Add calibration metrics for tail events

