"""Descriptive return-risk metrics for research; not trading advice."""

import numpy as np


def maximum_drawdown(equity: np.ndarray) -> float:
    values = np.asarray(equity, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.any(values <= 0):
        raise ValueError("equity must be a positive 1D series with at least two values")
    peaks = np.maximum.accumulate(values)
    return float(np.min(values / peaks - 1.0))


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 2 or periods_per_year < 1:
        raise ValueError("returns need at least two values and a positive period count")
    volatility = values.std(ddof=1)
    if volatility == 0:
        return 0.0
    return float(values.mean() / volatility * np.sqrt(periods_per_year))
