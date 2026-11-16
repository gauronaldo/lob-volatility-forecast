"""Non-negative blend with weights summing to 1, fitted on out-of-fold predictions."""

from __future__ import annotations

import numpy as np


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.arange(1, len(v) + 1)
    rho = np.nonzero(u - css / k > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


def fit_weights(preds: np.ndarray, y: np.ndarray, n_iter: int = 2000) -> np.ndarray:
    """Minimise the RMSPE of the blend over the simplex by projected gradient.

    Dividing by y turns the problem into least squares: minimise
    ||A w - 1||^2 with A = preds / y, w >= 0 and sum(w) = 1.
    """
    a = preds / y[:, None]
    n_models = a.shape[1]
    gram = a.T @ a / len(a)
    rhs = a.mean(axis=0)
    step = 1.0 / max(np.linalg.eigvalsh(gram).max(), 1e-12)
    w = np.full(n_models, 1.0 / n_models)
    for _ in range(n_iter):
        w = _project_simplex(w - step * (gram @ w - rhs))
    return w


def predict(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the weighted combination of member predictions."""
    return preds @ weights
