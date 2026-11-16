"""Same-time, stock-level, cluster, neighbour and lag features (S6).

See METHOD.md section 5.3. Three kinds of features live here:

- same-time market features need no fit and use feature windows only;
- stock-level statistics and clusters are fitted on the training rows of a
  split and then applied to every row;
- neighbour and lag features depend on the recovered order. Wide arrays are
  indexed (position, stock): row `p` is the time_id at position `p`.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lobvol.runlog import write_run_record

logger = logging.getLogger(__name__)

KEYS = ["stock_id", "time_id"]
SAME_TIME = ["mkt_rv_mean", "mkt_rv_std", "rv_over_mkt"]
STOCK_LEVEL = ["stock_rv_typ", "stock_spread_typ", "stock_updates_typ", "rv_rel_stock"]
CLUSTER = ["cluster_rv", "rv_over_cluster"]


def _ratio(num: Any, den: Any) -> np.ndarray:
    """Divide, returning NaN where the denominator is not positive."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    """Mean ignoring NaN; NaN where every value is NaN, without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


# --------------------------------------------------------------------------
# Same-time market (no fit)
# --------------------------------------------------------------------------
def same_time_features(bucket: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out mean and std of RV over the other stocks at the same time_id."""
    x = bucket["rv1_600"]
    group = bucket["time_id"]
    n = x.groupby(group).transform("count")
    total = x.groupby(group).transform("sum")
    total_sq = (x**2).groupby(group).transform("sum")
    others = (n - 1).where(n > 1)
    mean = (total - x) / others
    var = ((total_sq - x**2) / others - mean**2).clip(lower=0.0)
    out = bucket[KEYS].copy()
    out["mkt_rv_mean"] = mean
    out["mkt_rv_std"] = np.sqrt(var)
    out["rv_over_mkt"] = _ratio(x, mean)
    return out


# --------------------------------------------------------------------------
# Stock level and clusters (fitted on training rows)
# --------------------------------------------------------------------------
def fit_stock_level(train: pd.DataFrame) -> pd.DataFrame:
    """Return the typical RV, spread and update count of each stock."""
    stats = train.groupby("stock_id")[["rv1_600", "spread_600", "n_updates"]].median()
    stats.columns = STOCK_LEVEL[:3]
    return stats


def apply_stock_level(frame: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Return the stock-level feature columns for every row of a frame."""
    out = stats.reindex(frame["stock_id"]).reset_index(drop=True)
    out.index = frame.index
    out["rv_rel_stock"] = _ratio(frame["rv1_600"], out["stock_rv_typ"])
    return out


def fit_clusters(train: pd.DataFrame, n_clusters: int, seed: int) -> pd.Series:
    """Cluster stocks by KMeans on the stock-by-stock correlation of log RV."""
    from sklearn.cluster import KMeans

    positive = train[train["rv1_600"] > 0]
    wide = positive.assign(log_rv=np.log(positive["rv1_600"])).pivot_table(
        index="time_id", columns="stock_id", values="log_rv"
    )
    corr = wide.corr().fillna(0.0)
    k = max(1, min(n_clusters, len(corr)))
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return pd.Series(model.fit_predict(corr.to_numpy()), index=corr.index)


def apply_clusters(frame: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Return the leave-one-out cluster mean RV at the same time_id."""
    cluster = frame["stock_id"].map(labels)
    x = frame["rv1_600"].where(cluster.notna())
    group = [frame["time_id"], cluster]
    n = x.groupby(group).transform("count")
    total = x.groupby(group).transform("sum")
    mean = (total - x) / (n - 1).where(n > 1)
    out = pd.DataFrame(index=frame.index)
    out["cluster_rv"] = mean
    out["rv_over_cluster"] = _ratio(frame["rv1_600"], mean)
    return out


# --------------------------------------------------------------------------
# Neighbours
# --------------------------------------------------------------------------
def neighbour_index(dist: np.ndarray, k: int, mode: str, gap: int = 1) -> np.ndarray:
    """Return the positions of the k nearest candidates of every position.

    `dist` is ordered by position. `mode` is "past" (candidates at least
    `gap` positions earlier) or "any" (every other time_id). Rows with fewer
    than k candidates are filled with -1.
    """
    n = len(dist)
    d = np.array(dist, dtype=np.float64)
    idx = np.arange(n)
    if mode == "past":
        d[idx[None, :] > idx[:, None] - gap] = np.inf
    elif mode == "any":
        d[idx, idx] = np.inf
    else:
        raise ValueError(f"unknown candidate set: {mode}")
    if k >= n:
        return np.full((n, k), -1, dtype=np.int64)
    nearest = np.argpartition(d, k - 1, axis=1)[:, :k]
    near_d = np.take_along_axis(d, nearest, axis=1)
    order = np.argsort(near_d, axis=1, kind="stable")
    nearest = np.take_along_axis(nearest, order, axis=1)
    nearest[~np.isfinite(near_d).all(axis=1)] = -1
    return nearest


def volatility_distance(rv_wide: np.ndarray) -> np.ndarray:
    """RMS difference of log RV over the stocks observed at both time_ids."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rv = np.where(rv_wide > 0, np.log(rv_wide), np.nan)
    mask = (~np.isnan(log_rv)).astype(np.float64)
    x = np.nan_to_num(log_rv)
    sq = (x**2) @ mask.T + mask @ (x**2).T - 2.0 * x @ x.T
    count = mask @ mask.T
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.sqrt(np.maximum(sq, 0.0) / count)
    dist[count == 0] = np.inf
    return dist


def neighbour_features(
    rv_wide: np.ndarray,
    dists: dict[str, np.ndarray],
    ks: list[int],
    mode: str,
    gap: int = 1,
) -> dict[str, np.ndarray]:
    """Mean of own RV and of market RV over the k nearest time_ids.

    Feature windows only, never targets. Values are NaN where fewer than k
    candidates exist.
    """
    market = _nanmean(rv_wide, axis=1)
    out = {}
    for space, dist in dists.items():
        for k in ks:
            idx = neighbour_index(dist, k, mode, gap)
            valid = idx[:, 0] >= 0
            safe = np.where(idx >= 0, idx, 0)
            own = _nanmean(rv_wide[safe], axis=1)
            mkt = market[safe].mean(axis=1)
            own[~valid] = np.nan
            mkt = np.where(valid, mkt, np.nan)
            mkt_wide = np.repeat(mkt[:, None], rv_wide.shape[1], axis=1)
            name = f"nn_{space}{k}"
            out[f"{name}_own"] = own
            out[f"{name}_mkt"] = mkt_wide
            out[f"{name}_own_r"] = _ratio(own, rv_wide)
            out[f"{name}_mkt_r"] = _ratio(mkt_wide, market[:, None] + 0 * rv_wide)
    return out


# --------------------------------------------------------------------------
# Lags
# --------------------------------------------------------------------------
def lag_index(n: int, gap: int, n_mean: int) -> np.ndarray:
    """Return, per position, the n_mean positions starting `gap` steps earlier."""
    idx = np.arange(n)[:, None] - gap - np.arange(n_mean)[None, :]
    return np.where(idx >= 0, idx, -1)


def lag_features(
    rv_wide: np.ndarray, target_wide: np.ndarray, gap: int, n_mean: int
) -> dict[str, np.ndarray]:
    """RV and realized target of earlier time_ids: the previous one and a mean.

    `target_wide` holds NaN wherever the caller may not see the target.
    """
    n, n_stocks = rv_wide.shape
    idx = lag_index(n, gap, n_mean)
    valid = idx >= 0
    safe = np.where(valid, idx, 0)

    def lagged(wide: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        stack = np.where(valid[:, :, None], wide[safe], np.nan)
        return stack[:, 0], _nanmean(stack, axis=1)

    rv1, rv_mean = lagged(rv_wide)
    y1, y_mean = lagged(target_wide)
    market = _nanmean(target_wide, axis=1)
    m_stack = np.where(valid, market[safe], np.nan)
    ones = np.ones((1, n_stocks))
    return {
        "lag1_rv": rv1,
        "lag1_target": y1,
        "lagm_rv": rv_mean,
        "lagm_target": y_mean,
        "lag1_mkt_target": m_stack[:, :1] * ones,
        "lagm_mkt_target": _nanmean(m_stack, axis=1)[:, None] * ones,
        "lag1_target_r": _ratio(y1, rv_wide),
        "lagm_target_r": _ratio(y_mean, rv_wide),
    }


# --------------------------------------------------------------------------
# Wide <-> long and the S6 command
# --------------------------------------------------------------------------
def wide_layout(bucket: pd.DataFrame, order: pd.DataFrame) -> dict[str, Any]:
    """Map every bucket row to its (position, stock) cell."""
    position = order.set_index("time_id")["position"]
    stocks = np.sort(bucket["stock_id"].unique())
    row = bucket["time_id"].map(position).to_numpy()
    col = np.searchsorted(stocks, bucket["stock_id"].to_numpy())
    return {"row": row, "col": col, "shape": (len(position), len(stocks))}


def to_wide(values: np.ndarray, layout: dict[str, Any]) -> np.ndarray:
    """Scatter a column into a (position, stock) array, NaN where absent."""
    wide = np.full(layout["shape"], np.nan)
    wide[layout["row"], layout["col"]] = values
    return wide


def to_long(features: dict[str, np.ndarray], keys: pd.DataFrame, layout: dict):
    """Gather wide feature arrays back to one row per (stock_id, time_id)."""
    out = keys.reset_index(drop=True).copy()
    for name, wide in features.items():
        out[name] = wide[layout["row"], layout["col"]]
    return out


def price_distance_by_position(config: dict[str, Any], order: pd.DataFrame):
    """Load the price-space distance of S3, reordered by position."""
    dist = np.load(Path(config["paths"]["interim"]) / "dist_price.npy")
    by_time_id = order.sort_values("time_id")["position"].to_numpy()
    perm = np.argsort(by_time_id)
    return dist[perm][:, perm].astype(np.float64)


def temporal_feature_tables(
    bucket: pd.DataFrame,
    order: pd.DataFrame,
    dist_price: np.ndarray,
    targets: pd.DataFrame | None,
    frozen: dict[str, Any],
    setting: str,
) -> dict[str, pd.DataFrame]:
    """Build the neighbour and lag tables of one setting, for every bucket row."""
    layout = wide_layout(bucket, order)
    keys = bucket[KEYS]
    rv_wide = to_wide(bucket["rv1_600"].to_numpy(), layout)
    dists = {"price": dist_price, "vol": volatility_distance(rv_wide)}
    ks, gap = frozen["neighbours_k"], frozen["order_gap"]
    if setting == "competition":
        any_ = neighbour_features(rv_wide, dists, ks, "any")
        return {"nn_any": to_long(any_, keys, layout)}
    tables = {
        "nn_past": neighbour_features(rv_wide, dists, ks, "past", 1),
        "nn_past_gap": neighbour_features(rv_wide, dists, ks, "past", gap),
    }
    if targets is not None:
        merged = keys.merge(targets, on=KEYS, how="left")
        target_wide = to_wide(merged["target"].to_numpy(), layout)
        n_mean = frozen["lag_mean_last"]
        tables["lag"] = lag_features(rv_wide, target_wide, frozen["lag_single"], n_mean)
        tables["lag_gap"] = lag_features(rv_wide, target_wide, gap, n_mean)
    return {name: to_long(feats, keys, layout) for name, feats in tables.items()}


def run_xsection(config: dict[str, Any], setting: str, force: bool = False) -> None:
    """Write the neighbour and lag feature tables of one setting (S6)."""
    from lobvol.cv import load_dev_targets, order_status

    paths = config["paths"]
    out_dir = Path(paths["features"])
    names = ["nn_any"] if setting == "competition" else ["nn_past", "lag"]
    if all((out_dir / f"{n}.parquet").exists() for n in names) and not force:
        logger.info("%s features exist; skipping (use --force to redo)", setting)
        return
    if order_status(config)["status"] != "ok":
        logger.warning("order fallback: temporal features are not built")
        return
    bucket = pd.read_parquet(out_dir / "bucket.parquet", columns=KEYS + ["rv1_600"])
    order = pd.read_parquet(Path(paths["interim"]) / "time_order.parquet")
    dist_price = price_distance_by_position(config, order)
    targets = load_dev_targets(config) if setting == "strict" else None
    tables = temporal_feature_tables(
        bucket, order, dist_price, targets, config["frozen"], setting
    )
    for name, table in tables.items():
        if name.startswith("lag"):
            # Lags use targets: keep development rows only until S8.
            table = table.merge(targets[KEYS], on=KEYS)
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
        logger.info("wrote %s: %d rows, %d columns", name, *table.shape)
    write_run_record(out_dir, f"xsection_{setting}", config, [])
