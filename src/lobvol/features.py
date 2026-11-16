"""In-bucket features per (stock_id, time_id), no fitted statistics (S2).

Each bucket is forward-filled onto a 1-second grid from its first observed
second to second 599 (METHOD.md section 4). Forward-filling adds only zero
returns, so RV is unchanged, while averages of spread, depth and imbalance
become time-weighted. Nothing is filled across buckets or backwards.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lobvol.data import BOOK_DIR, list_stock_ids, read_book, read_trades
from lobvol.realized import (
    log_returns,
    realized_quarticity,
    realized_volatility,
    semivariance,
    wap,
)
from lobvol.runlog import write_run_record

logger = logging.getLogger(__name__)

KEYS = ["stock_id", "time_id"]


def grid_index(bucket: np.ndarray, second: np.ndarray, n_buckets: int, n_seconds: int):
    """Return, per (bucket, second), the row holding the latest observation or -1."""
    index = np.full((n_buckets, n_seconds), -1, dtype=np.int64)
    index[bucket, second] = np.arange(len(bucket))
    return np.maximum.accumulate(index, axis=1)


def to_grid(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Forward-fill one column onto the grid; NaN before the first observation."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.full(index.shape, np.nan)
    dense = values[np.maximum(index, 0)]
    dense[index < 0] = np.nan
    return dense


def _window(returns: np.ndarray, window: int) -> np.ndarray:
    """Return the returns that fall in the last `window` seconds of the bucket."""
    n_seconds = returns.shape[1] + 1
    return returns[:, max(0, n_seconds - window - 1) :]


def _nanmean(values: np.ndarray) -> np.ndarray:
    """Row mean ignoring NaN, NaN for an all-NaN row, without warnings."""
    count = np.sum(~np.isnan(values), axis=1)
    total = np.nansum(values, axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _nanstd(values: np.ndarray) -> np.ndarray:
    """Row standard deviation ignoring NaN, NaN for an all-NaN row."""
    mean = _nanmean(values)
    return np.sqrt(_nanmean((values - mean[:, None]) ** 2))


def stock_bucket_features(
    book: pd.DataFrame,
    trades: pd.DataFrame,
    windows: Sequence[int],
    bucket_seconds: int = 600,
) -> pd.DataFrame:
    """Compute the in-bucket features of one stock, one row per time_id."""
    book = book.sort_values(["time_id", "seconds_in_bucket"], kind="stable")
    time_ids, bucket = np.unique(book["time_id"].to_numpy(), return_inverse=True)
    n = len(time_ids)
    second = book["seconds_in_bucket"].to_numpy()
    index = grid_index(bucket, second, n, bucket_seconds)

    def col(name: str) -> np.ndarray:
        return to_grid(book[name].to_numpy(), index)

    bp1, ap1, bp2, ap2 = (
        col(c) for c in ("bid_price1", "ask_price1", "bid_price2", "ask_price2")
    )
    bs1, as1, bs2, as2 = (
        col(c) for c in ("bid_size1", "ask_size1", "bid_size2", "ask_size2")
    )
    wap1 = wap(bp1, ap1, bs1, as1)
    wap2 = wap(bp2, ap2, bs2, as2)
    r1 = log_returns(wap1)
    r2 = log_returns(wap2)
    spread = (ap1 - bp1) / (0.5 * (ap1 + bp1))
    imb1 = (bs1 - as1) / (bs1 + as1)
    imb2 = (bs2 - as2) / (bs2 + as2)

    # Trades, aligned to the buckets of the book.
    trades = trades[trades["time_id"].isin(time_ids)]
    trades = trades.sort_values(["time_id", "seconds_in_bucket"], kind="stable")
    t_bucket = np.searchsorted(time_ids, trades["time_id"].to_numpy())
    t_second = trades["seconds_in_bucket"].to_numpy()
    t_index = grid_index(t_bucket, t_second, n, bucket_seconds)
    rt = log_returns(to_grid(trades["price"].to_numpy(), t_index))
    t_size = np.zeros((n, bucket_seconds))
    t_size[t_bucket, t_second] = trades["size"].to_numpy()
    t_seen = np.zeros((n, bucket_seconds))
    t_seen[t_bucket, t_second] = 1.0
    t_orders = np.zeros(n)
    np.add.at(t_orders, t_bucket, trades["order_count"].to_numpy())
    no_trade = t_seen.sum(axis=1) == 0

    out: dict[str, Any] = {"time_id": time_ids}
    for w in windows:
        w1, w2, wt = _window(r1, w), _window(r2, w), _window(rt, w)
        up, down = semivariance(w1)
        out[f"rv1_{w}"] = realized_volatility(w1)
        out[f"rv2_{w}"] = realized_volatility(w2)
        out[f"rq1_{w}"] = realized_quarticity(w1)
        out[f"rsv_up_{w}"] = up
        out[f"rsv_dn_{w}"] = down
        out[f"trv_{w}"] = np.where(no_trade, np.nan, realized_volatility(wt))
        out[f"spread_{w}"] = _nanmean(spread[:, bucket_seconds - w :])
        out[f"tvol_{w}"] = t_size[:, bucket_seconds - w :].sum(axis=1)
        out[f"tsec_{w}"] = t_seen[:, bucket_seconds - w :].sum(axis=1)

    for minute in range(bucket_seconds // 60):
        chunk = r1[:, max(0, 60 * minute - 1) : 60 * minute + 59]
        out[f"rv1_m{minute}"] = realized_volatility(chunk)
    half = bucket_seconds // 2
    first_half = realized_volatility(r1[:, : half - 1])
    second_half = realized_volatility(r1[:, half - 1 :])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = second_half / first_half
    out["rv1_half_ratio"] = np.where(first_half > 0, ratio, np.nan)

    out["spread_std"] = _nanstd(spread)
    out["imb1_mean"] = _nanmean(imb1)
    out["imb1_std"] = _nanstd(imb1)
    out["imb2_mean"] = _nanmean(imb2)
    out["depth_mean"] = _nanmean(bs1 + as1 + bs2 + as2)
    out["wap_gap"] = _nanmean(np.abs(wap1 - wap2))
    out["n_updates"] = np.bincount(bucket, minlength=n).astype(np.float64)
    out["first_second"] = np.argmax(index >= 0, axis=1).astype(np.float64)
    out["t_orders"] = t_orders
    out["no_trade"] = no_trade.astype(np.float64)
    return pd.DataFrame(out)


def feature_group(name: str) -> str:
    """Return the feature group of a column, as named in METHOD.md section 5.3."""
    prefixes = [
        (("lag",), "Lags"),
        (("nn_",), "Neighbours"),
        (("cluster", "rv_over_cluster"), "Stock clusters"),
        (("stock_", "rv_rel_stock"), "Stock level"),
        (("mkt_", "rv_over_mkt"), "Same-time market"),
        (("trv_", "tvol_", "tsec_", "t_orders", "no_trade"), "Trades"),
        (("spread", "imb", "depth", "wap_gap", "n_updates", "first_sec"), "Book"),
        (("rv1_m", "rv1_half"), "Multi-window"),
    ]
    for starts, group in prefixes:
        if name.startswith(starts):
            return group
    if name.endswith(("_300", "_150")):
        return "Multi-window"
    if name.startswith(("rv", "rq", "rsv")):
        return "Realized"
    return "Other"


def bucket_feature_names(frame: pd.DataFrame) -> list[str]:
    """Return the in-bucket feature columns of a bucket feature table."""
    return [c for c in frame.columns if c not in KEYS]


def run_features(config: dict[str, Any], force: bool = False) -> Path:
    """Write the in-bucket features of every stock to data/features/bucket.parquet."""
    raw = Path(config["paths"]["raw"])
    part_dir = Path(config["paths"]["interim"]) / "bucket"
    final = Path(config["paths"]["features"]) / "bucket.parquet"
    if final.exists() and not force:
        logger.info("%s exists; skipping (use --force to redo)", final)
        return final
    part_dir.mkdir(parents=True, exist_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    windows = config["frozen"]["windows"]
    parts = []
    for stock_id in list_stock_ids(raw / BOOK_DIR):
        part = part_dir / f"stock_id={stock_id}.parquet"
        if not part.exists() or force:
            book = read_book(raw, stock_id)
            trades = read_trades(raw, stock_id)
            feats = stock_bucket_features(book, trades, windows)
            feats.insert(0, "stock_id", stock_id)
            feats.to_parquet(part, index=False)
            logger.info("stock %d: %d buckets", stock_id, len(feats))
            del book, trades, feats
        parts.append(pd.read_parquet(part))
    table = pd.concat(parts, ignore_index=True).sort_values(KEYS, ignore_index=True)
    table.to_parquet(final, index=False)
    write_run_record(final.parent, "features", config, [])
    logger.info("wrote %s: %d rows, %d columns", final, *table.shape)
    return final
