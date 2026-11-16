"""Price and order recovery on synthetic data (METHOD.md section 7, test 6)."""

from typing import Any

import pandas as pd
import pytest

from lobvol.data import TRAIN_CSV, read_book, read_trades
from lobvol.features import stock_bucket_features
from lobvol.synth import SynthData, generate
from lobvol.time_order import (
    build_price_table,
    order_accuracy,
    positions,
    recover_order,
    stock_prices,
    target_direction,
)


@pytest.fixture(scope="module")
def order_synth(synth_config: dict[str, Any], tmp_path_factory) -> SynthData:
    """The synthetic dataset on which the S3 gate is defined (config order.synth)."""
    out = tmp_path_factory.mktemp("order") / "raw"
    return generate(out, **synth_config["order"]["synth"])


@pytest.fixture(scope="module")
def recovered(synth_config: dict[str, Any], order_synth: SynthData, tmp_path_factory):
    cache = tmp_path_factory.mktemp("prices")
    tick = synth_config["frozen"]["tick_size"]
    table = build_price_table(order_synth.raw_dir, cache, tick, force=True)
    return recover_order(table, synth_config["order"])


def test_price_level_is_recovered_from_the_tick(order_synth: SynthData) -> None:
    truth = order_synth.true_p0.set_index(["stock_id", "time_id"])["p0"]
    for stock_id in order_synth.stock_ids[:8]:
        book = read_book(order_synth.raw_dir, stock_id)
        prices = stock_prices(book, tick=0.01).set_index("time_id")
        true_p0 = truth.loc[stock_id].reindex(prices.index)
        error = (prices["p0"] / true_p0 - 1.0).abs()
        assert error.notna().mean() > 0.9  # a tick was found for most buckets
        assert (error.dropna() < 1e-4).mean() > 0.95


def test_order_recovery_meets_the_gate(
    recovered: dict[str, Any], synth_config: dict[str, Any], order_synth: SynthData
) -> None:
    gates = synth_config["gates"]
    truth = order_synth.true_order.set_index("time_id")["position"]
    true_position = truth.reindex(recovered["mats"]["time_ids"]).to_numpy()
    seq, forward, backward = recovered["sequences"]["end_start_chain"]
    stats = order_accuracy(seq, true_position)
    assert stats["spearman"] >= gates["order_spearman_min"]  # correct direction too
    assert stats["displacement_p99"] <= gates["order_displacement_p99_max"]
    assert forward < backward  # price cue: end(i) is closer to start(i+1)


def test_target_cue_agrees_with_the_price_cue(
    recovered: dict[str, Any], synth_config: dict[str, Any], order_synth: SynthData
) -> None:
    seq, _, _ = recovered["sequences"]["end_start_chain"]
    time_ids = recovered["mats"]["time_ids"]
    position = pd.Series(positions(seq), index=time_ids)
    windows = synth_config["frozen"]["windows"]
    parts = []
    for stock_id in order_synth.stock_ids:
        book = read_book(order_synth.raw_dir, stock_id)
        feats = stock_bucket_features(
            book, read_trades(order_synth.raw_dir, stock_id), windows
        )
        parts.append(feats[["time_id", "rv1_600"]].assign(stock_id=stock_id))
    rv = pd.concat(parts).rename(columns={"rv1_600": "rv"})
    targets = pd.read_csv(order_synth.raw_dir / TRAIN_CSV)
    forward, backward = target_direction(position, rv=rv, targets=targets)
    assert forward > backward
    reverse = pd.Series(positions(seq[::-1]), index=time_ids)
    forward_r, backward_r = target_direction(reverse, rv=rv, targets=targets)
    assert forward_r < backward_r
