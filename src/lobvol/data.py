"""Reading the raw Optiver files one stock_id at a time (METHOD.md section 4).

Raw files are read-only. Nothing here loads more than one stock into memory.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

BOOK_DIR = "book_train.parquet"
TRADE_DIR = "trade_train.parquet"
TRAIN_CSV = "train.csv"

BOOK_COLUMNS = [
    "time_id",
    "seconds_in_bucket",
    "bid_price1",
    "ask_price1",
    "bid_price2",
    "ask_price2",
    "bid_size1",
    "ask_size1",
    "bid_size2",
    "ask_size2",
]
TRADE_COLUMNS = ["time_id", "seconds_in_bucket", "price", "size", "order_count"]


def list_stock_ids(dataset_dir: str | Path) -> list[int]:
    """Return the stock ids that have a partition in a book or trade directory."""
    partitions = Path(dataset_dir).glob("stock_id=*")
    return sorted(int(p.name.split("=")[1]) for p in partitions if p.is_dir())


def read_partition(dataset_dir: str | Path, stock_id: int) -> pd.DataFrame:
    """Read every row of one stock from a partitioned parquet directory."""
    return pd.read_parquet(Path(dataset_dir) / f"stock_id={stock_id}")


def read_book(raw_dir: str | Path, stock_id: int) -> pd.DataFrame:
    """Read the order-book updates of one stock."""
    return read_partition(Path(raw_dir) / BOOK_DIR, stock_id)


def read_trades(raw_dir: str | Path, stock_id: int) -> pd.DataFrame:
    """Read the trades of one stock."""
    return read_partition(Path(raw_dir) / TRADE_DIR, stock_id)


def read_targets(raw_dir: str | Path, allowed_time_ids: Iterable[int]) -> pd.DataFrame:
    """Read targets, keeping only the time_ids the caller is allowed to see.

    The explicit `allowed_time_ids` argument is the guard of METHOD.md rule 2:
    no caller can obtain holdout targets without naming their time_ids.
    """
    allowed = set(int(t) for t in allowed_time_ids)
    frame = pd.read_csv(Path(raw_dir) / TRAIN_CSV)
    return frame[frame["time_id"].isin(allowed)].reset_index(drop=True)
