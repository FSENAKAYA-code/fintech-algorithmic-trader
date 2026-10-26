import numpy as np
import pytest

from fintech.risk import annualized_sharpe, maximum_drawdown
from fintech.strategy import (
    historical_var,
    minimum_variance_weights,
    moving_average_signal,
)
from fintech.temporal import chronological_indices, evaluate_return_direction


def test_maximum_drawdown():
    assert maximum_drawdown(np.array([100, 120, 90, 110])) == pytest.approx(-0.25)

def test_flat_returns_have_zero_sharpe():
    assert annualized_sharpe(np.ones(5)) == 0


def test_temporal_partitions_are_ordered_and_disjoint():
    train, validation, test = chronological_indices(100)
    assert train[-1] < validation[0] < test[0]
    scores = evaluate_return_direction(
        np.random.default_rng(3).normal(size=(100, 4)), np.arange(100) % 2
    )
    assert {"train_loss", "validation_loss", "test_accuracy"} <= set(scores)


def test_finance_utilities_return_bounded_outputs():
    prices = np.linspace(10, 20, 40)
    assert set(moving_average_signal(prices, short_window=3, long_window=8)) <= {0, 1}
    assert historical_var(np.array([-0.1, 0.02, 0.03, -0.02])) >= 0
    weights = minimum_variance_weights(np.random.default_rng(4).normal(size=(100, 3)))
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1)
