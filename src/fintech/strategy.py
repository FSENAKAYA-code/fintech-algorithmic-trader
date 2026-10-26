"""Small, transparent financial research utilities; no orders are placed."""

import numpy as np
from scipy.optimize import minimize


def moving_average_signal(prices: np.ndarray, *, short_window: int = 10, long_window: int = 30) -> np.ndarray:
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or len(values) < long_window or np.any(values <= 0):
        raise ValueError("prices must be a positive 1D series at least long_window long")
    if short_window < 1 or long_window <= short_window:
        raise ValueError("require 1 <= short_window < long_window")
    short = np.convolve(values, np.ones(short_window) / short_window, mode="valid")
    long = np.convolve(values, np.ones(long_window) / long_window, mode="valid")
    aligned_short = short[long_window - short_window :]
    return (aligned_short > long).astype(int)


def historical_var(returns: np.ndarray, *, confidence: float = 0.95) -> float:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not 0 < confidence < 1:
        raise ValueError("provide at least two returns and confidence in (0, 1)")
    return float(max(0.0, -np.quantile(values, 1 - confidence)))


def minimum_variance_weights(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("returns must contain at least two rows and two assets")
    covariance = np.cov(values, rowvar=False) + np.eye(values.shape[1]) * 1e-9
    n_assets = values.shape[1]
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        np.full(n_assets, 1 / n_assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1)}],
    )
    if not result.success:
        raise RuntimeError(f"portfolio optimization failed: {result.message}")
    return result.x
