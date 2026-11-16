"""Model ladder: naive, har_ridge, forest, gbdt, mlp, blend (METHOD.md 5.4).

Every model takes training and validation frames and returns predictions of
`y` on the original scale. Frames carry the columns `stock_id`, `position`,
`rv1_600` and, for training rows, `target`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rv_floor(train_rv: np.ndarray | pd.Series) -> float:
    """Return the 1st percentile of positive RV_600 in the training rows."""
    values = np.asarray(train_rv, dtype=np.float64)
    return float(np.percentile(values[values > 0], 1))


def denominator(rv: np.ndarray | pd.Series, floor: float) -> np.ndarray:
    """Return x = max(RV_600, floor), the denominator of the ratio target."""
    return np.maximum(np.asarray(rv, dtype=np.float64), floor)
