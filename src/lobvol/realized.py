"""WAP, log returns and realized volatility, as defined in METHOD.md section 4.

Every function takes arrays whose last axis is time within one bucket. Returns
are never computed across buckets.

NaN returns (seconds before the first observation of a bucket) are ignored.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def wap(
    bid_price: ArrayLike,
    ask_price: ArrayLike,
    bid_size: ArrayLike,
    ask_size: ArrayLike,
) -> np.ndarray:
    """Return the weighted average price of one book level."""
    bid_price = np.asarray(bid_price, dtype=np.float64)
    ask_price = np.asarray(ask_price, dtype=np.float64)
    bid_size = np.asarray(bid_size, dtype=np.float64)
    ask_size = np.asarray(ask_size, dtype=np.float64)
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)


def log_returns(prices: ArrayLike) -> np.ndarray:
    """Return log(p_t) - log(p_{t-1}) along the last axis, within one bucket."""
    return np.diff(np.log(np.asarray(prices, dtype=np.float64)), axis=-1)


def realized_volatility(returns: ArrayLike) -> np.ndarray:
    """Return sqrt(sum of squared returns) along the last axis."""
    returns = np.asarray(returns, dtype=np.float64)
    return np.sqrt(np.nansum(returns**2, axis=-1))


def realized_quarticity(returns: ArrayLike) -> np.ndarray:
    """Return (N / 3) * sum of fourth powers of returns along the last axis."""
    returns = np.asarray(returns, dtype=np.float64)
    count = np.sum(~np.isnan(returns), axis=-1)
    return count / 3.0 * np.nansum(returns**4, axis=-1)


def semivariance(returns: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return the upside and downside realized semi-volatility along the last axis."""
    returns = np.nan_to_num(np.asarray(returns, dtype=np.float64))
    up = np.sqrt(np.sum(np.where(returns > 0, returns, 0.0) ** 2, axis=-1))
    down = np.sqrt(np.sum(np.where(returns < 0, returns, 0.0) ** 2, axis=-1))
    return up, down
