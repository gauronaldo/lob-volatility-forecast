"""Data audit (S1): schema, row counts, missing pairs, liquidity.

The audit never looks at `target` values (METHOD.md rule 2). Their
distribution is added in S4, on development rows only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from lobvol.data import (
    BOOK_COLUMNS,
    BOOK_DIR,
    TRADE_COLUMNS,
    TRADE_DIR,
    TRAIN_CSV,
    list_stock_ids,
    read_book,
    read_trades,
)
from lobvol.runlog import write_run_record

logger = logging.getLogger(__name__)

TRAIN_COLUMNS = ["stock_id", "time_id", "target"]


def _dir_size(path: Path) -> int:
    """Return the total size in bytes of the files under a path."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _partition_rows(dataset_dir: Path, stock_id: int) -> int:
    """Return the number of rows of one stock from the parquet metadata."""
    files = (dataset_dir / f"stock_id={stock_id}").glob("*.parquet")
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def audit_stock(raw: Path, stock_id: int) -> dict[str, Any]:
    """Scan one stock and return its audit statistics."""
    book = read_book(raw, stock_id)
    trades = read_trades(raw, stock_id)
    keys = ["time_id", "seconds_in_bucket"]
    per_bucket = book.groupby("time_id")["seconds_in_bucket"].agg(["min", "size"])
    book_ids = set(per_bucket.index)
    trade_ids = set(trades["time_id"].unique())
    return {
        "stock_id": stock_id,
        "book_rows": len(book),
        "trade_rows": len(trades),
        "book_columns_ok": list(book.columns) == BOOK_COLUMNS,
        "trade_columns_ok": list(trades.columns) == TRADE_COLUMNS,
        "book_dtypes": ", ".join(
            f"{c}:{t}" for c, t in book.dtypes.astype(str).items()
        ),
        "trade_dtypes": ", ".join(
            f"{c}:{t}" for c, t in trades.dtypes.astype(str).items()
        ),
        "buckets": len(per_bucket),
        "buckets_start_at_0": int((per_bucket["min"] == 0).sum()),
        "buckets_no_trades": len(book_ids - trade_ids),
        "duplicate_book_rows": int(book.duplicated(keys).sum()),
        "duplicate_trade_rows": int(trades.duplicated(keys).sum()),
        "median_updates": float(per_bucket["size"].median()),
        "seconds_ok": bool(book["seconds_in_bucket"].between(0, 599).all()),
        "book_time_ids": book_ids,
    }


def run_audit(config: dict[str, Any], force: bool = False) -> bool:
    """Write results/audit.md and return whether the schema gate passed."""
    raw = Path(config["paths"]["raw"])
    results = Path(config["paths"]["results"])
    out = results / "audit.md"
    results.mkdir(parents=True, exist_ok=True)

    header = list(pd.read_csv(raw / TRAIN_CSV, nrows=0).columns)
    pairs = pd.read_csv(raw / TRAIN_CSV, usecols=["stock_id", "time_id"])
    stock_ids = list_stock_ids(raw / BOOK_DIR)
    time_ids = sorted(pairs["time_id"].unique())
    full = pd.MultiIndex.from_product([stock_ids, time_ids], names=pairs.columns)
    present = pd.MultiIndex.from_frame(pairs)
    missing = full.difference(present).to_frame(index=False)
    missing.to_csv(results / "audit_missing_pairs.csv", index=False)

    rows = []
    train_without_book = 0
    book_without_train = 0
    by_stock = {s: set(g) for s, g in pairs.groupby("stock_id")["time_id"]}
    for stock_id in stock_ids:
        stats = audit_stock(raw, stock_id)
        book_ids = stats.pop("book_time_ids")
        train_ids = by_stock.get(stock_id, set())
        train_without_book += len(train_ids - book_ids)
        book_without_train += len(book_ids - train_ids)
        rows.append(stats)
    table = pd.DataFrame(rows)
    table.drop(columns=["book_dtypes", "trade_dtypes"]).to_csv(
        results / "audit_stocks.csv", index=False
    )

    checks = {
        "train.csv columns are stock_id, time_id, target": header == TRAIN_COLUMNS,
        "book columns match section 7 for every stock": bool(
            table["book_columns_ok"].all()
        ),
        "trade columns match section 7 for every stock": bool(
            table["trade_columns_ok"].all()
        ),
        "book and trade partitions cover the same stocks": stock_ids
        == list_stock_ids(raw / TRADE_DIR),
        "train.csv stocks all have a book partition": set(pairs["stock_id"])
        <= set(stock_ids),
        "no duplicate (stock_id, time_id) in train.csv": not pairs.duplicated().any(),
        "train.csv rows <= stocks x time_ids": len(pairs)
        <= len(stock_ids) * len(time_ids),
        "seconds_in_bucket within 0..599": bool(table["seconds_ok"].all()),
        "every train.csv pair has book rows": train_without_book == 0,
    }
    passed = all(checks.values())

    buckets = int(table["buckets"].sum())
    least = table.nsmallest(10, "median_updates")[
        ["stock_id", "median_updates", "book_rows", "buckets_no_trades"]
    ]
    size_gb = sum(_dir_size(raw / n) for n in (BOOK_DIR, TRADE_DIR, TRAIN_CSV)) / 1e9
    meta_rows = sum(_partition_rows(raw / BOOK_DIR, s) for s in stock_ids)
    lines = [
        "# Data audit (S1)",
        "",
        "Generated by `python -m lobvol audit`. Target values are not read here;",
        "their distribution is added by `python -m lobvol splits` (S4), on",
        "development rows only.",
        "",
        "**Gate (schema matches METHOD.md section 4): "
        f"{'PASS' if passed else 'FAIL'}**",
        "",
        "| Check | Result |",
        "| --- | --- |",
        *[f"| {k} | {'ok' if v else 'FAILED'} |" for k, v in checks.items()],
        "",
        "## Size and rows",
        "",
        f"- Size on disk (book, trade, train.csv): {size_gb:.2f} GB",
        f"- Stocks: {len(stock_ids)}; time_ids: {len(time_ids)}",
        f"- train.csv rows: {len(pairs):,} of {len(full):,} possible pairs;"
        f" missing pairs: {len(missing):,} (see `audit_missing_pairs.csv`)",
        f"- Book rows: {int(table['book_rows'].sum()):,} (parquet metadata:"
        f" {meta_rows:,}); per stock min / median / max:"
        f" {table['book_rows'].min():,} / {int(table['book_rows'].median()):,} /"
        f" {table['book_rows'].max():,}",
        f"- Trade rows: {int(table['trade_rows'].sum()):,}; per stock min / median /"
        f" max: {table['trade_rows'].min():,} / {int(table['trade_rows'].median()):,}"
        f" / {table['trade_rows'].max():,}",
        f"- Book buckets: {buckets:,}; train.csv pairs without book rows:"
        f" {train_without_book}; book buckets without a train.csv row:"
        f" {book_without_train}",
        "",
        "## Buckets",
        "",
        f"- Share of buckets whose first row is at second 0:"
        f" {table['buckets_start_at_0'].sum() / buckets:.4f}",
        f"- Buckets with no trades: {int(table['buckets_no_trades'].sum()):,}"
        f" ({table['buckets_no_trades'].sum() / buckets:.4%})",
        f"- Duplicate (time_id, seconds_in_bucket) rows: book"
        f" {int(table['duplicate_book_rows'].sum())}, trade"
        f" {int(table['duplicate_trade_rows'].sum())}",
        "",
        "## Dtypes",
        "",
        f"- Book: {table['book_dtypes'].iloc[0]}"
        f" ({table['book_dtypes'].nunique()} distinct layout(s) across stocks)",
        f"- Trade: {table['trade_dtypes'].iloc[0]}"
        f" ({table['trade_dtypes'].nunique()} distinct layout(s) across stocks)",
        "",
        "## Least liquid stocks (median book updates per bucket)",
        "",
        "| stock_id | median updates | book rows | buckets without trades |",
        "| --- | --- | --- | --- |",
        *[
            f"| {int(r.stock_id)} | {r.median_updates:.0f} | {int(r.book_rows):,} |"
            f" {int(r.buckets_no_trades)} |"
            for r in least.itertuples()
        ],
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    write_run_record(results / "run_records", "audit", config, [])
    logger.info("wrote %s; gate %s", out, "PASS" if passed else "FAIL")
    return passed
