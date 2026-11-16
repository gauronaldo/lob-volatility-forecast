"""In-bucket features and the audit (METHOD.md section 7, test 1)."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lobvol.audit import run_audit
from lobvol.data import BOOK_COLUMNS, TRADE_COLUMNS, TRAIN_CSV, read_book
from lobvol.features import feature_group, run_features, stock_bucket_features
from lobvol.realized import log_returns, realized_volatility, wap
from lobvol.synth import SynthData

WINDOWS = [600, 300, 150]


def _book(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=BOOK_COLUMNS)


def _no_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLUMNS).astype(
        {"time_id": int, "seconds_in_bucket": int, "price": float}
    )


# time_id 5: rows at seconds 0, 10 and 400. Spread is 0.02 / 1.0 until the last
# row, where the ask moves to 1.03.
BOOK = _book(
    [
        (5, 0, 0.99, 1.01, 0.98, 1.02, 100, 300, 50, 50),
        (5, 10, 0.99, 1.01, 0.98, 1.02, 300, 100, 50, 50),
        (5, 400, 0.99, 1.03, 0.98, 1.04, 100, 100, 50, 50),
        (9, 3, 1.00, 1.02, 0.99, 1.03, 10, 10, 20, 20),
    ]
)


def test_rv_equals_rv_of_the_sparse_rows() -> None:
    feats = stock_bucket_features(BOOK, _no_trades(), WINDOWS).set_index("time_id")
    rows = BOOK[BOOK["time_id"] == 5]
    sparse = wap(
        rows["bid_price1"], rows["ask_price1"], rows["bid_size1"], rows["ask_size1"]
    )
    # Hand-computed WAPs: 0.995, 1.005 and 1.01.
    assert sparse == pytest.approx([0.995, 1.005, 1.01])
    expected = realized_volatility(log_returns(sparse))
    assert feats.loc[5, "rv1_600"] == pytest.approx(expected, rel=1e-12)


def test_windows_use_only_the_last_seconds() -> None:
    feats = stock_bucket_features(BOOK, _no_trades(), WINDOWS).set_index("time_id")
    last_move = abs(np.log(1.01) - np.log(1.005))  # the change at second 400
    assert feats.loc[5, "rv1_300"] == pytest.approx(last_move, rel=1e-12)
    assert feats.loc[5, "rv1_150"] == 0.0
    assert feats.loc[5, "rv1_m0"] == pytest.approx(np.log(1.005 / 0.995), rel=1e-12)
    assert feats.loc[5, "rv1_m6"] == pytest.approx(last_move, rel=1e-12)


def test_spread_is_time_weighted() -> None:
    feats = stock_bucket_features(BOOK, _no_trades(), WINDOWS).set_index("time_id")
    # 400 seconds at 0.02 / 1.00 and 200 seconds at 0.04 / 1.01.
    expected = (400 * 0.02 / 1.00 + 200 * 0.04 / 1.01) / 600
    assert feats.loc[5, "spread_600"] == pytest.approx(expected, rel=1e-9)
    assert feats.loc[5, "n_updates"] == 3


def test_grid_starts_at_the_first_observed_second() -> None:
    feats = stock_bucket_features(BOOK, _no_trades(), WINDOWS).set_index("time_id")
    assert feats.loc[9, "first_second"] == 3
    assert feats.loc[9, "rv1_600"] == 0.0
    assert feats.loc[9, "spread_600"] == pytest.approx(0.02 / 1.01, rel=1e-9)


def test_buckets_without_trades_are_flagged() -> None:
    trades = pd.DataFrame(
        [(5, 20, 1.00, 10, 2), (5, 30, 1.01, 5, 1)], columns=TRADE_COLUMNS
    )
    feats = stock_bucket_features(BOOK, trades, WINDOWS).set_index("time_id")
    assert feats.loc[5, "no_trade"] == 0
    assert feats.loc[5, "tsec_600"] == 2
    assert feats.loc[5, "tvol_600"] == 15
    assert feats.loc[5, "t_orders"] == 3
    assert feats.loc[5, "trv_600"] == pytest.approx(np.log(1.01), rel=1e-12)
    assert feats.loc[9, "no_trade"] == 1
    assert np.isnan(feats.loc[9, "trv_600"])


def test_forward_filling_a_synthetic_stock_leaves_rv_unchanged(
    pipeline_synth: SynthData,
) -> None:
    stock_id = pipeline_synth.stock_ids[0]
    book = read_book(pipeline_synth.raw_dir, stock_id)
    feats = stock_bucket_features(book, _no_trades(), WINDOWS).set_index("time_id")
    for time_id, rows in list(book.groupby("time_id"))[:20]:
        sparse = wap(
            rows["bid_price1"], rows["ask_price1"], rows["bid_size1"], rows["ask_size1"]
        )
        expected = realized_volatility(log_returns(sparse))
        assert feats.loc[time_id, "rv1_600"] == pytest.approx(expected, rel=1e-9)


def test_every_feature_belongs_to_a_named_group() -> None:
    feats = stock_bucket_features(BOOK, _no_trades(), WINDOWS)
    groups = {feature_group(c) for c in feats.columns if c != "time_id"}
    assert groups == {"Realized", "Multi-window", "Book", "Trades"}


def test_run_features_and_audit_on_synthetic_data(
    synth_config: dict[str, Any], pipeline_synth: SynthData
) -> None:
    assert run_audit(synth_config)
    table = pd.read_parquet(run_features(synth_config))
    train = pd.read_csv(pipeline_synth.raw_dir / TRAIN_CSV)
    assert len(table) == len(train)
    assert not table.duplicated(["stock_id", "time_id"]).any()
    assert "target" not in table.columns
    assert (table["rv1_600"] > 0).all()


def test_audit_never_reads_target_values(
    synth_config: dict[str, Any], pipeline_synth: SynthData, tmp_path: Path
) -> None:
    import copy
    import shutil

    raw = tmp_path / "raw"
    shutil.copytree(pipeline_synth.raw_dir, raw)
    train = pd.read_csv(raw / TRAIN_CSV)
    train["target"] = "not a number"
    train.to_csv(raw / TRAIN_CSV, index=False)
    config = copy.deepcopy(synth_config)
    config["paths"]["raw"] = str(raw)
    config["paths"]["results"] = str(tmp_path / "results")
    assert run_audit(config)
