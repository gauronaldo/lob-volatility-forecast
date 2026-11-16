"""Leakage tests for fitted and temporal features (METHOD.md section 7, tests 4-5)."""

import numpy as np
import pandas as pd
import pytest

from lobvol.cross_section import (
    apply_clusters,
    apply_stock_level,
    fit_clusters,
    fit_stock_level,
    lag_features,
    lag_index,
    neighbour_features,
    neighbour_index,
    same_time_features,
    volatility_distance,
)

N_TIME, N_STOCKS, GAP = 60, 6, 10


@pytest.fixture(scope="module")
def rv_wide() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.exp(rng.normal(-6.0, 0.5, size=(N_TIME, N_STOCKS)))


@pytest.fixture(scope="module")
def dist() -> np.ndarray:
    rng = np.random.default_rng(1)
    d = rng.uniform(0.1, 1.0, size=(N_TIME, N_TIME))
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


# ---- test 4: strict indices point to the past ------------------------------
@pytest.mark.parametrize("k", [5, 20])
def test_past_neighbours_are_strictly_earlier(dist: np.ndarray, k: int) -> None:
    idx = neighbour_index(dist, k, "past", gap=1)
    for t in range(N_TIME):
        if t < k:
            assert (idx[t] == -1).all()  # fewer than k candidates -> NaN
        else:
            assert (idx[t] >= 0).all() and (idx[t] < t).all()
            assert len(set(idx[t])) == k


def test_gap_neighbours_are_at_least_g_positions_earlier(dist: np.ndarray) -> None:
    idx = neighbour_index(dist, 5, "past", gap=GAP)
    for t in range(N_TIME):
        if t - GAP + 1 < 5:
            assert (idx[t] == -1).all()
        else:
            assert (idx[t] <= t - GAP).all() and (idx[t] >= 0).all()


def test_any_neighbours_exclude_only_the_time_id_itself(dist: np.ndarray) -> None:
    idx = neighbour_index(dist, 5, "any")
    assert (idx >= 0).all()
    assert all(t not in idx[t] for t in range(N_TIME))
    assert any((idx[t] > t).any() for t in range(N_TIME))  # look-ahead by design


def test_neighbours_are_the_nearest_candidates(dist: np.ndarray) -> None:
    idx = neighbour_index(dist, 5, "past", gap=1)
    t = 40
    expected = np.argsort(dist[t, :t])[:5]
    assert list(idx[t]) == list(expected)


def test_lag_indices_point_to_earlier_positions() -> None:
    for gap in (1, GAP):
        idx = lag_index(N_TIME, gap, 6)
        for t in range(N_TIME):
            valid = idx[t][idx[t] >= 0]
            assert (valid <= t - gap).all()
            assert len(valid) == min(6, max(0, t - gap + 1))
    assert list(lag_index(N_TIME, 1, 6)[10]) == [9, 8, 7, 6, 5, 4]


# ---- test 5: causality -------------------------------------------------------
def test_strict_neighbour_features_ignore_later_feature_windows(
    rv_wide: np.ndarray, dist: np.ndarray
) -> None:
    dists = {"price": dist, "vol": volatility_distance(rv_wide)}
    base = neighbour_features(rv_wide, dists, [5], "past", 1)
    t = 30
    changed = rv_wide.copy()
    changed[t + 1 :] *= 3.0  # perturb every later feature window
    dists_changed = {"price": dist, "vol": volatility_distance(changed)}
    after = neighbour_features(changed, dists_changed, [5], "past", 1)
    for name in base:
        np.testing.assert_array_equal(base[name][: t + 1], after[name][: t + 1])
    any_base = neighbour_features(rv_wide, dists, [5], "any")
    any_after = neighbour_features(changed, dists_changed, [5], "any")
    assert not np.array_equal(
        any_base["nn_vol5_own"][: t + 1], any_after["nn_vol5_own"][: t + 1]
    )


@pytest.mark.parametrize("gap", [1, GAP])
def test_lag_features_ignore_targets_at_or_after_t(
    rv_wide: np.ndarray, gap: int
) -> None:
    rng = np.random.default_rng(2)
    targets = np.exp(rng.normal(-6.0, 0.5, size=rv_wide.shape))
    base = lag_features(rv_wide, targets, gap, 6)
    for t in (0, 1, 7, 30, N_TIME - 1):
        shuffled = targets.copy()
        shuffled[t:] = rng.uniform(size=shuffled[t:].shape)  # targets at or after t
        after = lag_features(rv_wide, shuffled, gap, 6)
        for name in base:
            np.testing.assert_array_equal(base[name][t], after[name][t])
    # The features do read earlier targets.
    shuffled = targets.copy()
    shuffled[:20] = rng.uniform(size=shuffled[:20].shape)
    assert (
        not np.array_equal(
            base["lag1_target"][20],
            lag_features(rv_wide, shuffled, gap, 6)["lag1_target"][20],
        )
        or gap > 1
    )


def test_lag_features_match_hand_computed_values() -> None:
    rv = np.arange(1.0, 9.0)[:, None]  # one stock, positions 0..7
    y = 10.0 * np.arange(1.0, 9.0)[:, None]
    feats = lag_features(rv, y, gap=1, n_mean=3)
    assert feats["lag1_target"][4, 0] == 40.0  # target of position 3
    assert feats["lagm_target"][4, 0] == pytest.approx((40 + 30 + 20) / 3)
    assert feats["lag1_rv"][4, 0] == 4.0
    assert feats["lag1_target_r"][4, 0] == pytest.approx(40.0 / 5.0)
    assert np.isnan(feats["lag1_target"][0, 0])
    assert feats["lagm_target"][1, 0] == 10.0  # only one earlier position


# ---- fitted statistics ---------------------------------------------------------
def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = [(s, t) for s in range(8) for t in range(40)]
    frame = pd.DataFrame(rows, columns=["stock_id", "time_id"])
    frame["rv1_600"] = np.exp(rng.normal(-6, 0.4, len(frame))) * (1 + frame["stock_id"])
    frame["spread_600"] = rng.uniform(1e-4, 1e-3, len(frame))
    frame["n_updates"] = rng.integers(50, 500, len(frame)).astype(float)
    return frame


def test_same_time_market_is_leave_one_out() -> None:
    frame = _frame()
    feats = same_time_features(frame)
    row = 5
    others = frame[
        (frame["time_id"] == frame.loc[row, "time_id"]) & (frame.index != row)
    ]["rv1_600"]
    assert feats.loc[row, "mkt_rv_mean"] == pytest.approx(others.mean())
    assert feats.loc[row, "mkt_rv_std"] == pytest.approx(others.std(ddof=0))


def test_fitted_statistics_use_training_rows_only() -> None:
    frame = _frame()
    train = (frame["time_id"] < 30).to_numpy()
    stats = fit_stock_level(frame[train])
    labels = fit_clusters(frame[train], 3, seed=0)
    changed = frame.copy()
    changed.loc[~train, ["rv1_600", "spread_600", "n_updates"]] *= 5.0
    pd.testing.assert_frame_equal(stats, fit_stock_level(changed[train]))
    pd.testing.assert_series_equal(labels, fit_clusters(changed[train], 3, seed=0))
    level = apply_stock_level(frame, stats)
    expected = frame[train].groupby("stock_id")["rv1_600"].median()
    assert level.loc[0, "stock_rv_typ"] == pytest.approx(expected.loc[0])
    cluster = apply_clusters(frame, labels)
    assert list(cluster.columns) == ["cluster_rv", "rv_over_cluster"]
    assert len(cluster) == len(frame)
