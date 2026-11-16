"""End-to-end smoke test of S3..S7 on synthetic data, with leakage checks."""

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from lobvol.cross_section import run_xsection
from lobvol.cv import load_roles, load_split, run_splits
from lobvol.features import run_features
from lobvol.pipeline import (
    feature_names,
    fold_frame,
    load_frame,
    metrics_path,
    oof_path,
    run_matrix,
    run_name,
)
from lobvol.time_order import run_order

REPO_ROOT = Path(__file__).resolve().parents[1]
FAST = {
    "gbdt": {
        "ladder": {"n_estimators": 30, "min_data_in_leaf": 20},
        "tuned": {"max_rounds": 60, "early_stopping_rounds": 10},
        "search": {"min_data_in_leaf": [10, 40, True]},
    },
    "forest": {"n_estimators": 10, "min_samples_leaf": 10},
    "mlp": {"max_epochs": 2, "hidden": [16], "batch_size": 256},
}


@pytest.fixture(scope="module")
def run_config(synth_config: dict[str, Any], tmp_path_factory) -> dict[str, Any]:
    """Run S2..S7 once on synthetic data with small models and open gates."""
    config = copy.deepcopy(synth_config)
    config_dir = tmp_path_factory.mktemp("configs")
    shutil.copytree(REPO_ROOT / "configs" / "models", config_dir / "models")
    for name, changes in FAST.items():
        path = config_dir / "models" / f"{name}.yaml"
        model = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, value in changes.items():
            if isinstance(value, dict):
                model[key].update(value)
            else:
                model[key] = value
        path.write_text(yaml.safe_dump(model), encoding="utf-8")
    config["config_dir"] = str(config_dir)
    config["frozen"]["optuna_trials"] = 2
    # Plumbing test: the gates are tested elsewhere and on the real data.
    config["gates"].update(
        naive_rmspe=[0.0, 10.0], sanity_rmspe_max=10.0, order_displacement_p99_max=1000
    )
    config["gates"]["order_permutations"] = 200
    config["runs"]["shap"]["rows"] = 50
    run_features(config)
    assert run_order(config)
    run_splits(config)
    for setting in ("strict", "competition"):
        run_xsection(config, setting)
    assert run_matrix(config)
    return config


def test_every_run_of_the_matrix_wrote_metrics(run_config: dict[str, Any]) -> None:
    runs = run_config["runs"]
    for scheme in runs["ladder"]["schemes"]:
        assert metrics_path(run_config, run_name("gbdt_fixed", "A", scheme)).exists()
    for model, features in runs["models"]:
        path = metrics_path(run_config, run_name(model, features, "walk_forward"))
        metrics = json.loads(path.read_text(encoding="utf-8"))
        assert np.isfinite(metrics["rmspe_mean"])
        assert metrics["n_rows"] > 0 and metrics["naive_mean"] > 0
    results = Path(run_config["paths"]["results"])
    assert (results / "gbdt_tuned.json").exists()
    assert (results / "shap.csv").exists()
    trials = json.loads((results / "trials.json").read_text(encoding="utf-8"))
    assert sum(t["kind"] == "optuna_trial" for t in trials) == 2


def test_all_models_share_the_same_folds(run_config: dict[str, Any]) -> None:
    split = load_split(run_config, "walk_forward")
    expected = np.full(len(split), -1)
    for k in range(run_config["frozen"]["walk_forward_folds"]):
        expected[split[f"fold_{k}"].to_numpy() == 2] = k
    for model, features in run_config["runs"]["models"]:
        oof = pd.read_parquet(
            oof_path(run_config, run_name(model, features, "walk_forward"))
        )
        assert oof[["stock_id", "time_id"]].equals(split[["stock_id", "time_id"]])
        assert np.array_equal(oof["fold"].to_numpy(), expected)
        assert oof.loc[oof["fold"] >= 0, "pred"].notna().all()


def test_holdout_rows_never_reach_training(run_config: dict[str, Any]) -> None:
    roles = load_roles(run_config)
    locked = set(roles.loc[roles["role"] != "development", "time_id"])
    assert len(locked) > 0
    for features in ("A", "D", "C_comp"):
        frame = load_frame(run_config, features)
        assert not locked & set(frame["time_id"])
    lag = pd.read_parquet(Path(run_config["paths"]["features"]) / "lag.parquet")
    assert not locked & set(lag["time_id"])


@pytest.mark.parametrize("features", ["A", "B", "C_strict", "C_comp"])
def test_target_shuffle_leaves_validation_features_identical(
    run_config: dict[str, Any], features: str
) -> None:
    # METHOD.md section 7, test 5, for feature sets without LAG.
    frame = load_frame(run_config, features)
    split = load_split(run_config, "walk_forward")
    train = split["fold_2"].to_numpy() == 1
    val = split["fold_2"].to_numpy() == 2
    shuffled = frame.copy()
    rng = np.random.default_rng(0)
    shuffled.loc[val, "target"] = rng.uniform(1e-4, 1e-2, int(val.sum()))
    base = fold_frame(frame, features, train, run_config)
    other = fold_frame(shuffled, features, train, run_config)
    cols = feature_names(base)
    pd.testing.assert_frame_equal(base.loc[val, cols], other.loc[val, cols])


def test_lag_features_of_a_row_use_only_earlier_targets(
    run_config: dict[str, Any],
) -> None:
    frame = load_frame(run_config, "D")
    by_key = frame.set_index(["stock_id", "position"])
    row = frame[frame["position"] == frame["position"].median().astype(int)].iloc[0]
    earlier = (row["stock_id"], row["position"] - 1)
    if earlier in by_key.index:
        assert row["lag1_target"] == pytest.approx(by_key.loc[earlier, "target"])
        assert row["lag1_rv"] == pytest.approx(by_key.loc[earlier, "rv1_600"])


def test_evaluate_then_holdout_runs_exactly_once(run_config: dict[str, Any]) -> None:
    from lobvol.holdout import run_holdout
    from lobvol.report import run_evaluate

    results = Path(run_config["paths"]["results"])
    summary = run_evaluate(run_config)
    table = pd.read_csv(results / "model_table.csv")
    n_models = len(run_config["runs"]["models"])
    assert len(table) == n_models + len(run_config["runs"]["ladder"]["schemes"])
    assert (table["ci_low"] <= table["rmspe_pooled"]).all()
    assert (table["rmspe_pooled"] <= table["ci_high"]).all()
    for name in summary["figures"].values():
        assert name is not None and (results / "figures" / name).exists()
    assert len(pd.read_csv(results / "differences.csv")) >= 15
    assert len(pd.read_csv(results / "ablation.csv")) == 9

    assert run_holdout(run_config)
    holdout = pd.read_csv(results / "holdout" / "model_table.csv")
    assert len(holdout) == n_models and np.isfinite(holdout["rmspe"]).all()
    roles = load_roles(run_config)
    preds = pd.read_parquet(
        Path(run_config["paths"]["interim"]) / "holdout_predictions.parquet"
    )
    assert set(preds["time_id"]) <= set(
        roles.loc[roles["role"] == "holdout", "time_id"]
    )
    assert len(pd.read_csv(results / "holdout" / "dm_tests.csv")) == 9
    assert not run_holdout(run_config)  # the holdout is opened once
    run_evaluate(run_config)  # figure 3 now carries the holdout
