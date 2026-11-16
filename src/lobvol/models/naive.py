"""Naive persistence: yhat = RV_600."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict(frame: pd.DataFrame) -> np.ndarray:
    """Return the current realized volatility as the forecast."""
    return frame["rv1_600"].to_numpy(dtype=np.float64)
