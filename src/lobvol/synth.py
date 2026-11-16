"""Small synthetic generator in the Optiver file format, with a known time order.

Tests and CI run on this data, never on Kaggle data. The generator reproduces
the properties the pipeline depends on:

- real prices sit on a tick grid and follow one random walk across time_ids,
  so both the price level and the time order can be recovered;
- prices are normalised within each (stock_id, time_id) by the mid price at
  second 0;
- a book row exists only for seconds in which the book changed, and some
  buckets do not start at second 0;
- trades are sparse and one stock is illiquid enough to have empty buckets;
- time_ids come in days: consecutive buckets of a day are nearly contiguous
  and a larger overnight gap separates days;
- volatility is persistent along the true order and shared across stocks;
  it moves between a feature window and its target window, so a target is
  closer in time to the next feature window than to the previous one;
- the target is the RV of WAP1 over the 600 seconds after the feature window;
- time_id labels are shuffled and some (stock_id, time_id) pairs are missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lobvol.data import BOOK_DIR, TRADE_DIR, TRAIN_CSV
from lobvol.realized import log_returns, realized_volatility, wap

logger = logging.getLogger(__name__)

BUCKET_SECONDS = 600
TRUE_ORDER_CSV = "true_order.csv"


@dataclass(frozen=True)
class SynthData:
    """Location of a synthetic dataset, its stocks and its true time order."""

    raw_dir: Path
    stock_ids: list[int]
    true_order: pd.DataFrame
    true_p0: pd.DataFrame


def _hold(
    rng: np.random.Generator, change_prob: float, values: np.ndarray
) -> np.ndarray:
    """Hold each row's value until the next random change point."""
    change = rng.random(values.shape) < change_prob
    change[:, 0] = True
    last = np.where(change, np.arange(values.shape[1]), 0)
    last = np.maximum.accumulate(last, axis=1)
    return np.take_along_axis(values, last, axis=1)


def generate(
    out_dir: str | Path,
    n_stocks: int = 6,
    n_time_ids: int = 40,
    seed: int = 0,
    tick: float = 0.01,
    missing_fraction: float = 0.02,
    day_length: int = 13,
) -> SynthData:
    """Write a synthetic dataset in the Optiver format and return its true order."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    n_sec = 2 * BUCKET_SECONDS  # feature window followed by target window
    shape = (n_time_ids, n_sec)
    seconds = np.arange(BUCKET_SECONDS)

    # Index = true position in time; value = shuffled time_id label.
    time_ids = rng.choice(np.arange(5, 32000), size=n_time_ids, replace=False)
    by_label = np.argsort(time_ids)  # raw files are sorted by time_id
    stock_ids = np.sort(rng.choice(127, size=n_stocks, replace=False))

    # Market state shared by all stocks: persistent log-volatility and returns.
    # One volatility state per window: feature 0, target 0, feature 1, ...
    log_vol = np.zeros(2 * n_time_ids)
    shocks = rng.normal(0.0, 0.18, 2 * n_time_ids)
    for pos in range(1, 2 * n_time_ids):
        log_vol[pos] = 0.95 * log_vol[pos - 1] + shocks[pos]
    sigma = 1e-4 * np.exp(log_vol).reshape(n_time_ids, 2)  # per-second volatility
    sigma = np.repeat(sigma, BUCKET_SECONDS, axis=1)  # (time_id, second)
    market_ret = rng.normal(size=shape) * sigma
    overnight = np.arange(n_time_ids) % day_length == day_length - 1
    gap_sd = np.where(overnight, 0.012, 0.001)  # gap after each bucket
    market_gap = rng.normal(size=n_time_ids) * gap_sd
    missing = rng.random((n_stocks, n_time_ids)) < missing_fraction

    targets = []
    true_p0 = []
    for s, stock_id in enumerate(stock_ids):
        illiquid = s == n_stocks - 1
        base_price = rng.uniform(20.0, 300.0)
        vol_scale = np.exp(rng.normal(0.0, 0.3))
        trade_prob = 0.002 if illiquid else rng.uniform(0.03, 0.2)
        size_change_prob = 0.01 if illiquid else rng.uniform(0.02, 0.3)

        # Log real price: one random walk across all time_ids in true order.
        idio_ret = rng.normal(size=shape) * sigma
        steps = vol_scale * (0.6 * market_ret + 0.8 * idio_ret)
        steps[:, 0] = 0.0
        within = steps.cumsum(axis=1)
        gap = 0.6 * market_gap + 0.8 * rng.normal(size=n_time_ids) * gap_sd
        block = within[:, -1] + gap
        start = np.log(base_price) + np.concatenate([[0.0], np.cumsum(block)[:-1]])
        price = np.exp(start[:, None] + within)

        # Book in integer ticks, then sizes.
        bid1 = np.floor(price / tick).astype(np.int64)
        ask1 = bid1 + _hold(rng, 0.02, rng.integers(1, 3, size=shape))
        bid2 = bid1 - _hold(rng, 0.02, rng.integers(1, 3, size=shape))
        ask2 = ask1 + _hold(rng, 0.02, rng.integers(1, 3, size=shape))
        bid_size1, ask_size1, bid_size2, ask_size2 = (
            _hold(rng, size_change_prob, rng.integers(1, 500, size=shape))
            for _ in range(4)
        )

        # Normalise by the mid price at second 0 of the feature window.
        ref = 0.5 * (bid1[:, 0] + ask1[:, 0])
        true_p0.append(
            pd.DataFrame(
                {"stock_id": int(stock_id), "time_id": time_ids, "p0": ref * tick}
            )
        )
        ref_col = ref[:, None]

        # Target: RV of WAP1 over the target window only.
        wap1 = wap(bid1 / ref_col, ask1 / ref_col, bid_size1, ask_size1)
        target = realized_volatility(log_returns(wap1[:, BUCKET_SECONDS:]))

        # Book rows: only seconds in which something changed.
        levels = [bid1, ask1, bid2, ask2, bid_size1, ask_size1, bid_size2, ask_size2]
        dense = np.stack(levels, axis=-1)[:, :BUCKET_SECONDS]
        changed = np.ones((n_time_ids, BUCKET_SECONDS), dtype=bool)
        changed[:, 1:] = (dense[:, 1:] != dense[:, :-1]).any(axis=-1)
        late = rng.integers(1, 6, n_time_ids)
        first = np.where(rng.random(n_time_ids) < 0.9, 0, late)[:, None]
        present = ~missing[s][:, None] & (seconds[None, :] >= first)
        keep = present & (changed | (seconds[None, :] == first))

        rows, secs = np.nonzero(keep[by_label])
        pos = by_label[rows]
        book = pd.DataFrame(
            {
                "time_id": time_ids[pos].astype(np.int16),
                "seconds_in_bucket": secs.astype(np.int16),
                "bid_price1": (bid1[pos, secs] / ref[pos]).astype(np.float32),
                "ask_price1": (ask1[pos, secs] / ref[pos]).astype(np.float32),
                "bid_price2": (bid2[pos, secs] / ref[pos]).astype(np.float32),
                "ask_price2": (ask2[pos, secs] / ref[pos]).astype(np.float32),
                "bid_size1": bid_size1[pos, secs].astype(np.int32),
                "ask_size1": ask_size1[pos, secs].astype(np.int32),
                "bid_size2": bid_size2[pos, secs].astype(np.int32),
                "ask_size2": ask_size2[pos, secs].astype(np.int32),
            }
        )

        # Trades: sparse, at the best bid or the best ask.
        traded = present & (rng.random((n_time_ids, BUCKET_SECONDS)) < trade_prob)
        at_ask = rng.random((n_time_ids, BUCKET_SECONDS)) < 0.5
        trade_size = rng.integers(1, 300, size=(n_time_ids, BUCKET_SECONDS))
        order_count = rng.integers(1, 6, size=(n_time_ids, BUCKET_SECONDS))
        rows, secs = np.nonzero(traded[by_label])
        pos = by_label[rows]
        trade_ticks = np.where(at_ask[pos, secs], ask1[pos, secs], bid1[pos, secs])
        trades = pd.DataFrame(
            {
                "time_id": time_ids[pos].astype(np.int16),
                "seconds_in_bucket": secs.astype(np.int16),
                "price": (trade_ticks / ref[pos]).astype(np.float32),
                "size": trade_size[pos, secs].astype(np.int32),
                "order_count": order_count[pos, secs].astype(np.int16),
            }
        )

        for name, frame in ((BOOK_DIR, book), (TRADE_DIR, trades)):
            part_dir = out / name / f"stock_id={stock_id}"
            part_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(part_dir / "part-0.parquet", index=False)

        observed = ~missing[s]
        targets.append(
            pd.DataFrame(
                {
                    "stock_id": int(stock_id),
                    "time_id": time_ids[observed],
                    "target": target[observed],
                }
            )
        )

    train = pd.concat(targets, ignore_index=True)
    train = train.sort_values(["stock_id", "time_id"], ignore_index=True)
    train.to_csv(out / TRAIN_CSV, index=False)

    true_order = pd.DataFrame({"time_id": time_ids, "position": np.arange(n_time_ids)})
    true_order.to_csv(out / TRUE_ORDER_CSV, index=False)

    logger.info(
        "Wrote synthetic data to %s: %d stocks, %d time_ids, %d target rows",
        out,
        n_stocks,
        n_time_ids,
        len(train),
    )
    return SynthData(
        raw_dir=out,
        stock_ids=[int(s) for s in stock_ids],
        true_order=true_order,
        true_p0=pd.concat(true_p0, ignore_index=True),
    )
