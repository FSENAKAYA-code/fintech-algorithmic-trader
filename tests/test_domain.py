import numpy as np
import pytest

from fintech.risk import annualized_sharpe, maximum_drawdown


def test_maximum_drawdown():
    assert maximum_drawdown(np.array([100, 120, 90, 110])) == pytest.approx(-0.25)

def test_flat_returns_have_zero_sharpe():
    assert annualized_sharpe(np.ones(5)) == 0
