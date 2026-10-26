import numpy as np
import pytest

from fintech.risk import annualized_sharpe, maximum_drawdown
from fintech.temporal import chronological_indices, evaluate_return_direction


def test_maximum_drawdown():
    assert maximum_drawdown(np.array([100, 120, 90, 110])) == pytest.approx(-0.25)

def test_flat_returns_have_zero_sharpe():
    assert annualized_sharpe(np.ones(5)) == 0


def test_temporal_partitions_are_ordered_and_disjoint():
    train, validation, test = chronological_indices(100)
    assert train[-1] < validation[0] < test[0]
    assert len(evaluate_return_direction(np.random.default_rng(3).normal(size=(100, 4)), np.arange(100) % 2)) == 3
