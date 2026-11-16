"""The synthetic generator follows the data contract of METHOD.md section 4."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
from lobvol.synth import TRUE_ORDER_CSV, SynthData, generate

N_STOCKS = 5
N_TIME_IDS = 30


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> SynthData:
    out = tmp_path_factory.mktemp("synth")
    return generate(out, n_stocks=N_STOCKS, n_time_ids=N_TIME_IDS, seed=0)


def test_partitions_exist_for_every_stock(synth: SynthData) -> None:
    assert len(synth.stock_ids) == N_STOCKS
    assert list_stock_ids(synth.raw_dir / BOOK_DIR) == synth.stock_ids
    assert list_stock_ids(synth.raw_dir / TRADE_DIR) == synth.stock_ids


def test_book_follows_the_data_contract(synth: SynthData) -> None:
    for stock_id in synth.stock_ids:
        book = read_book(synth.raw_dir, stock_id)
        assert list(book.columns) == BOOK_COLUMNS
        assert book["seconds_in_bucket"].between(0, 599).all()
        assert not book.duplicated(["time_id", "seconds_in_bucket"]).any()
        assert book["time_id"].is_monotonic_increasing
        assert (book["ask_price1"] > book["bid_price1"]).all()
        assert (book["bid_price1"] > book["bid_price2"]).all()
        assert (book["ask_price2"] > book["ask_price1"]).all()
        assert (book.filter(like="size") > 0).all().all()
        # Prices are normalised: the mid price of each bucket starts near 1.
        first = book.groupby("time_id").first()
        mid = 0.5 * (first["bid_price1"] + first["ask_price1"])
        assert np.allclose(mid, 1.0, atol=5e-3)


def test_book_rows_exist_only_when_the_book_changed(synth: SynthData) -> None:
    for stock_id in synth.stock_ids:
        book = read_book(synth.raw_dir, stock_id)
        assert len(book) < 600 * book["time_id"].nunique()
        values = book.drop(columns="seconds_in_bucket")
        same_as_previous = (values == values.shift()).all(axis=1)
        assert not same_as_previous.any()


def test_some_buckets_do_not_start_at_second_zero(synth: SynthData) -> None:
    firsts = pd.concat(
        read_book(synth.raw_dir, s).groupby("time_id")["seconds_in_bucket"].min()
        for s in synth.stock_ids
    )
    assert (firsts == 0).mean() > 0.5
    assert (firsts > 0).any()


def test_trades_are_sparse_and_can_be_absent(synth: SynthData) -> None:
    for stock_id in synth.stock_ids:
        trades = read_trades(synth.raw_dir, stock_id)
        assert list(trades.columns) == TRADE_COLUMNS
        assert trades["seconds_in_bucket"].between(0, 599).all()
        assert (trades[["price", "size", "order_count"]] > 0).all().all()
    illiquid = synth.stock_ids[-1]
    with_book = read_book(synth.raw_dir, illiquid)["time_id"].nunique()
    with_trades = read_trades(synth.raw_dir, illiquid)["time_id"].nunique()
    assert with_trades < with_book


def test_train_csv_matches_the_book(synth: SynthData) -> None:
    train = pd.read_csv(synth.raw_dir / TRAIN_CSV)
    assert list(train.columns) == ["stock_id", "time_id", "target"]
    assert len(train) <= N_STOCKS * N_TIME_IDS
    assert not train.duplicated(["stock_id", "time_id"]).any()
    assert (train["target"] > 0).all()
    assert 1e-4 < train["target"].median() < 1e-1
    for stock_id, rows in train.groupby("stock_id"):
        book_ids = set(read_book(synth.raw_dir, stock_id)["time_id"])
        assert set(rows["time_id"]) == book_ids


def test_time_ids_are_shuffled_and_shared_by_all_stocks(synth: SynthData) -> None:
    order = synth.true_order
    assert sorted(order["position"]) == list(range(N_TIME_IDS))
    assert order["time_id"].is_unique
    assert not order.sort_values("position")["time_id"].is_monotonic_increasing
    train = pd.read_csv(synth.raw_dir / TRAIN_CSV)
    assert set(train["time_id"]) <= set(order["time_id"])
    saved = pd.read_csv(synth.raw_dir / TRUE_ORDER_CSV)
    pd.testing.assert_frame_equal(saved, order, check_dtype=False)


def test_volatility_is_persistent_along_the_true_order(synth: SynthData) -> None:
    train = pd.read_csv(synth.raw_dir / TRAIN_CSV).merge(synth.true_order)
    market = train.groupby("position")["target"].mean().sort_index()
    log_rv = np.log(market.to_numpy())
    assert np.corrcoef(log_rv[:-1], log_rv[1:])[0, 1] > 0.3


def test_same_seed_gives_identical_files(synth: SynthData, tmp_path: Path) -> None:
    again = generate(tmp_path, n_stocks=N_STOCKS, n_time_ids=N_TIME_IDS, seed=0)
    assert again.stock_ids == synth.stock_ids
    pd.testing.assert_frame_equal(
        pd.read_csv(again.raw_dir / TRAIN_CSV),
        pd.read_csv(synth.raw_dir / TRAIN_CSV),
    )
    stock_id = synth.stock_ids[0]
    pd.testing.assert_frame_equal(
        read_book(again.raw_dir, stock_id), read_book(synth.raw_dir, stock_id)
    )


def test_different_seed_gives_different_targets(
    synth: SynthData, tmp_path: Path
) -> None:
    other = generate(tmp_path, n_stocks=N_STOCKS, n_time_ids=N_TIME_IDS, seed=1)
    a = pd.read_csv(synth.raw_dir / TRAIN_CSV)["target"].to_numpy()
    b = pd.read_csv(other.raw_dir / TRAIN_CSV)["target"].to_numpy()
    assert len(a) != len(b) or not np.allclose(a, b)
