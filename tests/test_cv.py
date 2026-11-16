"""Splits, embargo and holdout (METHOD.md section 7, test 3)."""

import numpy as np
import pandas as pd
import pytest

from lobvol.cv import (
    TRAIN,
    UNUSED,
    VAL,
    cut_holdout,
    inner_split,
    make_split,
    time_folds,
)

N_TIME = 500
EMBARGO = 10
N_FOLDS = 5


@pytest.fixture(scope="module")
def roles() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    position = pd.DataFrame(
        {"time_id": rng.permutation(N_TIME) + 100, "position": np.arange(N_TIME)}
    )
    return cut_holdout(position, fraction=0.2, embargo=EMBARGO)


@pytest.fixture(scope="module")
def rows(roles: pd.DataFrame) -> pd.DataFrame:
    dev = roles[roles["role"] == "development"]
    stocks = pd.DataFrame({"stock_id": range(7)})
    return stocks.merge(dev[["time_id", "position"]], how="cross")


def test_holdout_is_the_last_fifth_behind_an_embargo(roles: pd.DataFrame) -> None:
    counts = roles["role"].value_counts()
    assert counts["holdout"] == 100
    assert counts["embargo"] == EMBARGO
    assert counts["development"] == N_TIME - 100 - EMBARGO
    last_dev = roles.loc[roles["role"] == "development", "position"].max()
    first_holdout = roles.loc[roles["role"] == "holdout", "position"].min()
    assert first_holdout - last_dev - 1 == EMBARGO
    assert roles.loc[roles["role"] == "holdout", "position"].max() == N_TIME - 1


@pytest.mark.parametrize("scheme", ["group_time", "blocked_time", "walk_forward"])
def test_no_time_id_is_shared_between_train_and_validation(
    rows: pd.DataFrame, scheme: str
) -> None:
    split = make_split(rows, scheme, N_FOLDS, EMBARGO, seed=0)
    for k in range(N_FOLDS):
        train = set(split.loc[split[f"fold_{k}"] == TRAIN, "time_id"])
        val = set(split.loc[split[f"fold_{k}"] == VAL, "time_id"])
        assert train and val
        assert not train & val


def test_kfold_rows_partitions_the_rows(rows: pd.DataFrame) -> None:
    split = make_split(rows, "kfold_rows", N_FOLDS, EMBARGO, seed=0)
    folds = split[[f"fold_{k}" for k in range(N_FOLDS)]].to_numpy()
    assert ((folds == VAL).sum(axis=1) == 1).all()
    assert ((folds == TRAIN).sum(axis=1) == N_FOLDS - 1).all()


def test_every_development_row_is_validated_once_in_complete_schemes(
    rows: pd.DataFrame,
) -> None:
    for scheme in ("group_time", "blocked_time"):
        split = make_split(rows, scheme, N_FOLDS, EMBARGO, seed=0)
        folds = split[[f"fold_{k}" for k in range(N_FOLDS)]].to_numpy()
        assert ((folds == VAL).sum(axis=1) == 1).all()


def test_blocked_time_has_an_embargo_on_both_sides() -> None:
    roles = time_folds(400, "blocked_time", N_FOLDS, EMBARGO, seed=0)
    for k in range(N_FOLDS):
        val = np.flatnonzero(roles[:, k] == VAL)
        train = np.flatnonzero(roles[:, k] == TRAIN)
        assert np.all(np.diff(val) == 1)  # contiguous block
        gap = np.abs(train[:, None] - val[None, :]).min()
        assert gap == EMBARGO + 1
        unused = np.flatnonzero(roles[:, k] == UNUSED)
        expected = EMBARGO * (2 if 0 < k < N_FOLDS - 1 else 1)
        assert len(unused) == expected


def test_walk_forward_trains_only_on_the_past_with_an_embargo() -> None:
    roles = time_folds(600, "walk_forward", N_FOLDS, EMBARGO, seed=0)
    sizes = []
    for k in range(N_FOLDS):
        val = np.flatnonzero(roles[:, k] == VAL)
        train = np.flatnonzero(roles[:, k] == TRAIN)
        assert np.all(np.diff(val) == 1)
        assert train.max() < val.min()
        assert val.min() - train.max() - 1 == EMBARGO
        assert np.array_equal(train, np.arange(len(train)))  # blocks 1..i
        sizes.append(len(train))
    assert sizes == sorted(sizes) and len(set(sizes)) == N_FOLDS
    assert (roles[:100] != VAL).all()  # the first block is never validated


def test_holdout_time_ids_appear_in_no_development_fold(
    roles: pd.DataFrame, rows: pd.DataFrame
) -> None:
    locked = set(roles.loc[roles["role"] != "development", "time_id"])
    for scheme in ("kfold_rows", "group_time", "blocked_time", "walk_forward"):
        split = make_split(rows, scheme, N_FOLDS, EMBARGO, seed=0)
        assert not locked & set(split["time_id"])


def test_inner_split_takes_the_last_share_of_training_time_ids() -> None:
    position = np.repeat(np.arange(100), 3)
    inner = inner_split(position, 0.15)
    assert set(position[inner]) == set(range(85, 100))
