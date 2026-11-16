"""LightGBM on the ratio target q = y / x with weights 1 / q^2.

Weighted squared error on q with weights 1 / q^2 is the squared percentage
error of y (see `lobvol.evaluation`), so the training loss is RMSPE^2.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from lobvol.cv import inner_split
from lobvol.models import denominator, rv_floor

CATEGORICAL = ["stock_id"]
TREE_KEYS = [
    "learning_rate",
    "num_leaves",
    "min_data_in_leaf",
    "feature_fraction",
    "bagging_fraction",
    "lambda_l1",
    "lambda_l2",
    "max_bin",
]


def _lgb_params(params: dict[str, Any], seed: int, threads: int) -> dict[str, Any]:
    """Translate a model config into LightGBM parameters."""
    out = {k: params[k] for k in TREE_KEYS if k in params}
    out.update(
        objective="regression",
        metric="l2",
        bagging_freq=1,
        seed=seed,
        num_threads=threads,
        verbose=-1,
        force_col_wise=True,
    )
    return out


def _dataset(frame: pd.DataFrame, features: list[str], floor: float) -> lgb.Dataset:
    """Build a weighted LightGBM dataset on the ratio target."""
    q = frame["target"].to_numpy() / denominator(frame["rv1_600"], floor)
    cats = [c for c in CATEGORICAL if c in features]
    return lgb.Dataset(
        frame[features],
        label=q,
        weight=1.0 / q**2,
        categorical_feature=cats,
        free_raw_data=False,
    )


def fit(
    train: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    seed: int,
    inner_fraction: float,
    threads: int = 0,
) -> dict[str, Any]:
    """Fit on training rows; return the booster, the RV floor and the tree count.

    With `early_stopping_rounds` the number of trees is chosen on the inner
    split (the last share of training time_ids), never on the validation
    fold, and the model is then refitted on all training rows.
    """
    floor = rv_floor(train["rv1_600"])
    lgb_params = _lgb_params(params, seed, threads)
    if "early_stopping_rounds" in params:
        inner = inner_split(train["position"].to_numpy(), inner_fraction)
        probe = lgb.train(
            lgb_params,
            _dataset(train[~inner], features, floor),
            num_boost_round=params["max_rounds"],
            valid_sets=[_dataset(train[inner], features, floor)],
            callbacks=[
                lgb.early_stopping(params["early_stopping_rounds"], verbose=False)
            ],
        )
        n_trees = max(1, probe.best_iteration)
    else:
        n_trees = params["n_estimators"]
    booster = lgb.train(
        lgb_params, _dataset(train, features, floor), num_boost_round=n_trees
    )
    return {"booster": booster, "floor": floor, "features": features, "trees": n_trees}


def predict(fitted: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Predict y = qhat * x."""
    q_hat = fitted["booster"].predict(frame[fitted["features"]])
    return q_hat * denominator(frame["rv1_600"], fitted["floor"])


def importance(fitted: dict[str, Any]) -> pd.Series:
    """Return the gain importance of every feature."""
    gain = fitted["booster"].feature_importance(importance_type="gain")
    return pd.Series(gain, index=fitted["features"], dtype=np.float64)
