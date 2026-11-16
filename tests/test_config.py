"""The base config mirrors the frozen parameters of METHOD.md section 6."""

from pathlib import Path

import pytest

from lobvol.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# Copied by hand from METHOD.md section 6. A failure here means a frozen
# parameter changed; that is a change of protocol, not a test update.
FROZEN = {
    "holdout_fraction": 0.20,
    "embargo": 10,
    "walk_forward_folds": 5,
    "windows": [600, 300, 150],
    "neighbours_k": [5, 20],
    "lag_single": 1,
    "lag_mean_last": 6,
    "inner_split_fraction": 0.15,
    "optuna_trials": 40,
    "optuna_feature_set": "B",
    "optuna_seed": 0,
    "seeds": [0, 1, 2],
    "bootstrap_resamples": 1000,
    "bootstrap_block_length": 20,
    "tick_size": 0.01,
    "order_gap": 10,
}


def test_frozen_parameters_match_method() -> None:
    config = load_config(REPO_ROOT / "configs" / "base.yaml")
    assert config["frozen"] == FROZEN


def test_gate_thresholds_match_method() -> None:
    gates = load_config(REPO_ROOT / "configs" / "base.yaml")["gates"]
    assert gates["naive_rmspe"] == [0.33, 0.35]
    assert gates["sanity_rmspe_max"] == 0.24
    assert gates["order_spearman_min"] == 0.99
    assert gates["order_displacement_p99_max"] == 10
    assert gates["order_jump_ratio_max"] == 0.25
    assert gates["order_p_value_max"] == 0.01


def test_every_model_has_a_config() -> None:
    names = {p.stem for p in (REPO_ROOT / "configs" / "models").glob("*.yaml")}
    assert names == {"naive", "har_ridge", "forest", "gbdt", "mlp", "blend"}
    for name in names:
        model = load_config(REPO_ROOT / "configs" / "models" / f"{name}.yaml")
        assert model["model"] == name


def test_load_config_rejects_a_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
