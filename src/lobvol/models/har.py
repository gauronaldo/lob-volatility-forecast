"""HAR-style Ridge on log y with stock fixed effects (Corsi 2009).

The point forecast is exp(mu - 1.5 * s^2), the RMSPE-optimal back-transform
of a log-normal target (derived in `lobvol.evaluation`); s^2 is the residual
variance on the training rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from lobvol.cv import inner_split
from lobvol.evaluation import rmspe

INPUTS = ["rv1_600", "rv1_300", "rv1_150", "mkt_rv_mean"]


def _design(frame: pd.DataFrame, state: dict[str, Any]) -> sparse.csr_matrix:
    """Standardised log inputs next to one-hot stock indicators."""
    logs = np.log(np.maximum(frame[INPUTS].to_numpy(dtype=np.float64), state["floor"]))
    logs = np.nan_to_num((logs - state["mean"]) / state["std"])
    codes = frame["stock_id"].map(state["stock_code"]).to_numpy()
    known = ~np.isnan(codes)
    rows = np.flatnonzero(known)
    onehot = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, codes[known].astype(int))),
        shape=(len(frame), len(state["stock_code"])),
    )
    return sparse.hstack([sparse.csr_matrix(logs), onehot], format="csr")


def _prepare(train: pd.DataFrame) -> dict[str, Any]:
    """Fit floors, scaling and the stock coding on training rows."""
    values = train[INPUTS].to_numpy(dtype=np.float64)
    floor = np.array([np.percentile(c[c > 0], 1) for c in values.T])
    logs = np.log(np.maximum(values, floor))
    stocks = np.sort(train["stock_id"].unique())
    return {
        "floor": floor,
        "mean": logs.mean(axis=0),
        "std": logs.std(axis=0) + 1e-12,
        "stock_code": pd.Series(np.arange(len(stocks)), index=stocks),
    }


def _fit(train: pd.DataFrame, alpha: float) -> dict[str, Any]:
    """Fit the Ridge and the residual variance on training rows."""
    state = _prepare(train)
    x, y = _design(train, state), np.log(train["target"].to_numpy())
    model = Ridge(alpha=alpha).fit(x, y)
    state.update(model=model, s2=float(np.var(y - model.predict(x))))
    return state


def _predict(state: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Return exp(mu - 1.5 * s^2)."""
    mu = state["model"].predict(_design(frame, state))
    return np.exp(mu - 1.5 * state["s2"])


def fit_predict(
    train: pd.DataFrame,
    val: pd.DataFrame,
    params: dict[str, Any],
    inner_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose alpha on the inner time split, refit on all training rows, predict."""
    inner = inner_split(train["position"].to_numpy(), inner_fraction)
    scores = {}
    for alpha in params["alphas"]:
        state = _fit(train[~inner], alpha)
        scores[alpha] = rmspe(train.loc[inner, "target"], _predict(state, train[inner]))
    best = min(scores, key=scores.get)
    state = _fit(train, best)
    return _predict(state, val), {"alpha": best, "s2": state["s2"]}
