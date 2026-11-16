"""WAP, returns and RV against hand-computed values (METHOD.md section 7, test 1)."""

import numpy as np
import pytest

from lobvol.realized import log_returns, realized_volatility, wap


def test_wap_matches_hand_computed_value() -> None:
    # (0.99 * 300 + 1.01 * 100) / (100 + 300) = (297 + 101) / 400 = 0.995
    value = wap(bid_price=0.99, ask_price=1.01, bid_size=100, ask_size=300)
    assert value == pytest.approx(0.995, abs=1e-12)


def test_wap_weights_each_price_by_the_opposite_size() -> None:
    # A large bid size pulls the WAP towards the ask.
    assert wap(0.99, 1.01, bid_size=900, ask_size=100) == pytest.approx(1.008)


def test_log_returns_match_hand_computed_values() -> None:
    # log(1.01) = 0.00995033...
    returns = log_returns([1.0, 1.01, 1.0])
    assert returns == pytest.approx([0.00995033, -0.00995033], abs=1e-8)


def test_realized_volatility_matches_hand_computed_value() -> None:
    # sqrt(0.003^2 + 0.004^2) = 0.005
    assert realized_volatility([0.003, -0.004]) == pytest.approx(0.005, abs=1e-12)


def test_repeated_prices_leave_rv_unchanged() -> None:
    sparse = np.array([1.0, 1.002, 0.999, 1.001])
    filled = np.repeat(sparse, [3, 1, 4, 2])  # forward-filled copy
    rv_sparse = realized_volatility(log_returns(sparse))
    rv_filled = realized_volatility(log_returns(filled))
    assert rv_filled == pytest.approx(rv_sparse, abs=1e-15)


def test_functions_work_along_the_last_axis() -> None:
    prices = np.array([[1.0, 1.01, 1.0], [1.0, 1.0, 1.0]])
    rv = realized_volatility(log_returns(prices))
    assert rv.shape == (2,)
    assert rv[1] == 0.0
