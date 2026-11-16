"""Random forest on the ratio target with weights 1 / q^2. Optional model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from lobvol.models import denominator, rv_floor


def fit_predict(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    seed: int,
    threads: int = -1,
) -> np.ndarray:
    """Fit a subsampled forest with large leaves and predict y = qhat * x."""
    floor = rv_floor(train["rv1_600"])
    q = train["target"].to_numpy() / denominator(train["rv1_600"], floor)
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        min_samples_leaf=params["min_samples_leaf"],
        max_samples=params["max_samples"],
        max_features=params["max_features"],
        n_jobs=threads or -1,
        random_state=seed,
    )
    model.fit(train[features], q, sample_weight=1.0 / q**2)
    return model.predict(val[features]) * denominator(val["rv1_600"], floor)
