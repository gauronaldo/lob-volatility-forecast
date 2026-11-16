"""RMSPE, paired moving-block bootstrap and the Diebold-Mariano test.

Loss alignment (METHOD.md section 4), with y > 0 the target and c a forecast:

1. RMSPE^2 = mean_i ((y_i - c_i) / y_i)^2 = mean_i (1 / y_i^2) (y_i - c_i)^2.
   Minimising RMSPE is therefore weighted least squares with weights 1 / y^2.

2. Ratio target. Let q = y / x with x > 0 known at forecast time, and let the
   forecast be c = qhat * x. Then (y - c) / y = (q*x - qhat*x) / (q*x)
   = (q - qhat) / q. The percentage error of y equals that of q, so a model
   of q is trained with weights 1 / q^2.

3. Log target. Minimising E[(1 - c / y)^2] over a constant c gives
   c* = E[1 / y] / E[1 / y^2]. If log y ~ N(mu, s^2), then
   E[y^-1] = exp(-mu + s^2 / 2) and E[y^-2] = exp(-2*mu + 2*s^2), hence
   c* = exp(mu - 1.5 * s^2). The result is exact under Gaussian residuals
   with constant variance and an approximation otherwise; s^2 is estimated
   on training rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def rmspe(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the root mean squared percentage error of positive targets."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.all(y_true > 0):
        raise ValueError("RMSPE is undefined for non-positive targets")
    return float(np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)))


def squared_pct_error(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Return the squared percentage error of every row."""
    y_true = np.asarray(y_true, dtype=np.float64)
    return ((y_true - np.asarray(y_pred, dtype=np.float64)) / y_true) ** 2


def per_time(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Sum squared percentage errors per time_id, ordered by position.

    `frame` has time_id, position, target and one prediction column per
    model. The result has one row per time_id: `n` and one sum per model.
    """
    errors = pd.DataFrame(
        {c: squared_pct_error(frame["target"], frame[c]) for c in columns}
    )
    errors["position"] = frame["position"].to_numpy()
    errors["n"] = 1.0
    return errors.groupby("position").sum().sort_index()


def block_indices(n: int, block_length: int, n_resamples: int, seed: int) -> np.ndarray:
    """Draw moving-block bootstrap resamples of n ordered items.

    Each resample concatenates blocks of consecutive items that start at
    random places and is cut to n items.
    """
    rng = np.random.default_rng(seed)
    length = min(block_length, n)
    n_blocks = int(np.ceil(n / length))
    starts = rng.integers(0, n - length + 1, size=(n_resamples, n_blocks))
    index = starts[:, :, None] + np.arange(length)[None, None, :]
    return index.reshape(n_resamples, -1)[:, :n]


def bootstrap_rmspe(
    sums: pd.DataFrame, columns: list[str], index: np.ndarray
) -> np.ndarray:
    """Return pooled RMSPE per resample and model, all stocks of a time_id together."""
    count = sums["n"].to_numpy()[index].sum(axis=1)
    out = np.empty((len(index), len(columns)))
    for j, column in enumerate(columns):
        out[:, j] = np.sqrt(sums[column].to_numpy()[index].sum(axis=1) / count)
    return out


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Return the percentile interval of bootstrap samples."""
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def diebold_mariano(
    loss_a: ArrayLike, loss_b: ArrayLike, lag: int | None = None
) -> dict[str, float]:
    """Diebold-Mariano test of equal accuracy on per-time_id losses.

    The loss differential is d_t = loss_a_t - loss_b_t. Its long-run variance
    is the Newey-West (Bartlett) estimate with `lag` autocovariances, by
    default floor(T^(1/3)). The statistic carries the Harvey-Leybourne-
    Newbold small-sample correction with h = lag + 1 and is compared with a
    Student t distribution with T - 1 degrees of freedom. A negative
    statistic means model a has the lower loss.
    """
    from scipy import stats

    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = len(d)
    if lag is None:
        lag = int(np.floor(n ** (1.0 / 3.0)))
    centred = d - d.mean()
    variance = centred @ centred / n
    for k in range(1, lag + 1):
        gamma = centred[k:] @ centred[:-k] / n
        variance += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    h = lag + 1
    stat = d.mean() / np.sqrt(max(variance, 1e-300) / n)
    stat *= np.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1e-12) / n)
    p_value = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "mean_difference": float(d.mean()),
        "lag": float(lag),
        "n": float(n),
    }
