"""Holdout evaluation (S8): refit on the development set, predict the holdout once.

See METHOD.md section 5.6. This is the only module that reads holdout
targets. Fitted statistics, feature selection, hyperparameters and blend
weights come from the development set; strict features of holdout rows may
use earlier time_ids, including their realized targets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lobvol.cross_section import (
    KEYS,
    price_distance_by_position,
    temporal_feature_tables,
)
from lobvol.cv import load_roles, order_status, primary_scheme
from lobvol.data import read_targets
from lobvol.evaluation import (
    block_indices,
    bootstrap_rmspe,
    diebold_mariano,
    interval,
    per_time,
    rmspe,
    squared_pct_error,
)
from lobvol.models import blend
from lobvol.pipeline import (
    FEATURE_TABLES,
    SEEDED,
    SELECTING,
    SELECTION_BASE,
    assemble_frame,
    feature_names,
    fit_predict,
    fold_frame,
    load_frame,
    model_config,
    noise_selection,
    oof_path,
    resolve_set,
    run_name,
)
from lobvol.report import LABELS, difference_row, md_table
from lobvol.runlog import git_commit, write_run_record

logger = logging.getLogger(__name__)

FEATURE_LADDER = [
    ("naive", "A"),
    ("gbdt", "A"),
    ("gbdt", "B"),
    ("gbdt", "C_strict"),
    ("gbdt", "D"),
]
MODEL_LADDER = [
    ("naive", "A"),
    ("har_ridge", "A"),
    ("forest", "D"),
    ("gbdt", "D"),
    ("mlp", "D"),
    ("blend", "D"),
]
PAIRS = [
    ("RQ3 look-ahead: C_strict minus C_comp", ("gbdt", "C_strict"), ("gbdt", "C_comp")),
    ("RQ1: LightGBM A minus naive", ("gbdt", "A"), ("naive", "A")),
    ("RQ1: B minus A", ("gbdt", "B"), ("gbdt", "A")),
    ("RQ1: C_strict minus B", ("gbdt", "C_strict"), ("gbdt", "B")),
    ("RQ1: D minus C_strict", ("gbdt", "D"), ("gbdt", "C_strict")),
    ("RQ1: D minus B", ("gbdt", "D"), ("gbdt", "B")),
    (
        "Order error: C_strict_gap minus C_strict",
        ("gbdt", "C_strict_gap"),
        ("gbdt", "C_strict"),
    ),
    ("Order error: C_strict_gap minus B", ("gbdt", "C_strict_gap"), ("gbdt", "B")),
    ("Order error: D_gap minus D", ("gbdt", "D_gap"), ("gbdt", "D")),
    ("Order error: D_gap minus B", ("gbdt", "D_gap"), ("gbdt", "B")),
    ("Model: HAR ridge minus naive", ("har_ridge", "A"), ("naive", "A")),
    ("Model: forest minus HAR ridge", ("forest", "D"), ("har_ridge", "A")),
    ("Model: LightGBM D minus forest", ("gbdt", "D"), ("forest", "D")),
    ("RQ4: MLP minus LightGBM D", ("mlp", "D"), ("gbdt", "D")),
    ("Model: blend minus LightGBM D", ("blend", "D"), ("gbdt", "D")),
]


def source_hash(config: dict[str, Any]) -> str:
    """Hash the package sources and configs that define the frozen models."""
    config_dir = Path(config.get("config_dir", "configs"))
    files = sorted(Path(__file__).parent.rglob("*.py")) + sorted(
        config_dir.rglob("*.yaml")
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def column(model: str, features: str) -> str:
    """Return the prediction column name of one model."""
    return f"{model}__{features}"


def run_holdout(config: dict[str, Any]) -> bool:
    """Refit every frozen model on the development set and score the holdout once."""
    paths, frozen = config["paths"], config["frozen"]
    out = Path(paths["results"]) / "holdout"
    if (out / "metrics.json").exists():
        logger.error("The holdout has already been evaluated; it runs exactly once.")
        return False
    out.mkdir(parents=True, exist_ok=True)
    ok = order_status(config)["status"] == "ok"
    models = []
    for model, features in config["runs"]["models"]:
        key = (model, resolve_set(config, features))
        if key not in models:
            models.append(key)
    freeze = {
        "git_commit": git_commit(),
        "source_sha256": source_hash(config),
        "models": [list(m) for m in models],
        "primary_scheme": primary_scheme(config),
        "order_status": order_status(config)["status"],
    }
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    # Rows of every role, with all targets: allowed from S8 on.
    feat_dir = Path(paths["features"])
    roles = load_roles(config)
    bucket = pd.read_parquet(feat_dir / "bucket.parquet")
    targets = read_targets(paths["raw"], roles["time_id"])
    rows = targets.merge(roles[["time_id", "position"]], on="time_id")
    role_of = roles.set_index("time_id")["role"]
    tables: dict[str, pd.DataFrame] = {}
    if ok:
        order = pd.read_parquet(Path(paths["interim"]) / "time_order.parquet")
        strict = temporal_feature_tables(
            bucket[KEYS + ["rv1_600"]],
            order,
            price_distance_by_position(config, order),
            targets,
            frozen,
            "strict",
        )
        tables = {name: strict[name] for name in ("lag", "lag_gap")}
        for name in ("nn_past", "nn_past_gap", "nn_any"):
            tables[name] = pd.read_parquet(feat_dir / f"{name}.parquet")

    frames: dict[str, pd.DataFrame] = {}

    def frame_for(features: str) -> pd.DataFrame:
        if features not in frames:
            parts = [tables[name] for name in FEATURE_TABLES[features]]
            frames[features] = assemble_frame(bucket, rows, parts)
        return frames[features]

    base = frame_for("A")
    role = base["time_id"].map(role_of).to_numpy()
    train_mask, test_mask = role == "development", role == "holdout"
    if ok:
        # Development rows must carry exactly the features they had in S7.
        dev_now = frame_for("D")[train_mask].reset_index(drop=True)
        dev_then = load_frame(config, "D").reset_index(drop=True)
        for name in ("lag1_target", "lagm_target", "nn_price5_own"):
            if not np.allclose(dev_now[name], dev_then[name], equal_nan=True):
                raise RuntimeError(f"development feature {name} changed at S8")

    selections: dict[str, list[str]] = {}

    def selection(features: str) -> list[str]:
        key = SELECTION_BASE.get(features, features)
        if key not in selections:
            full = fold_frame(frame_for(key), key, train_mask, config)
            selections[key] = noise_selection(
                full[train_mask], feature_names(full), config
            )
        return selections[key]

    preds = base.loc[test_mask, KEYS + ["position", "target", "rv1_600"]].copy()
    info: dict[str, dict[str, Any]] = {}
    for model, features in models:
        if model == "blend":
            continue
        full = fold_frame(frame_for(features), features, train_mask, config)
        cols = feature_names(full)
        if model in SELECTING:
            kept = set(selection(features))
            cols = [c for c in cols if c in kept]
        seeds = frozen["seeds"] if model in SEEDED else frozen["seeds"][:1]
        train, test = full[train_mask], full[test_mask]
        start = time.perf_counter()
        runs = []
        for seed in seeds:
            pred, _ = fit_predict(model, train, test, cols, seed, config)
            runs.append(pred)
        preds[column(model, features)] = np.mean(runs, axis=0)
        info[column(model, features)] = {
            "seeds": seeds,
            "n_features": len(cols),
            "train_seconds": round(time.perf_counter() - start, 1),
            "n_train_rows": int(train_mask.sum()),
        }
        logger.info("holdout: %s %s fitted", model, features)

    blends = [m for m in models if m[0] == "blend"]
    if blends:
        # Weights from the development out-of-fold predictions only.
        scheme = primary_scheme(config)
        members = [
            (m, resolve_set(config, f))
            for m, f in model_config(config, "blend")["members"]
        ]
        dev = load_frame(config, "A")[KEYS + ["target"]].copy()
        for m, f in members:
            oof = pd.read_parquet(oof_path(config, run_name(m, f, scheme)))
            dev[column(m, f)] = oof["pred"].where(oof["fold"] >= 0).to_numpy()
        cols = [column(m, f) for m, f in members]
        dev = dev.dropna(subset=cols)
        weights = blend.fit_weights(dev[cols].to_numpy(), dev["target"].to_numpy())
        name = column(*blends[0])
        preds[name] = blend.predict(preds[cols].to_numpy(), weights)
        info[name] = {
            "seeds": [],
            "n_features": len(cols),
            "train_seconds": 0.0,
            "n_train_rows": len(dev),
            "weights": dict(
                zip(cols, [round(float(w), 4) for w in weights], strict=True)
            ),
        }

    interim = Path(paths["interim"])
    preds.to_parquet(interim / "holdout_predictions.parquet", index=False)

    # Metrics, intervals, differences and Diebold-Mariano tests.
    names = [column(m, f) for m, f in models]
    sums = per_time(preds, names)
    index = block_indices(
        len(sums),
        frozen["bootstrap_block_length"],
        frozen["bootstrap_resamples"],
        frozen["seeds"][0],
    )
    samples = bootstrap_rmspe(sums, names, index)
    naive_rmspe = rmspe(preds["target"], preds["rv1_600"])
    table_rows = []
    for j, (model, features) in enumerate(models):
        low, high = interval(samples[:, j])
        name = names[j]
        table_rows.append(
            {
                "model": model,
                "features": features,
                "rmspe": rmspe(preds["target"], preds[name]),
                "ci_low": low,
                "ci_high": high,
                "naive_rmspe": naive_rmspe,
                "n_rows": len(preds),
                "seeds": " ".join(str(s) for s in info[name]["seeds"]) or "-",
                "n_features": info[name]["n_features"],
                "train_seconds": info[name]["train_seconds"],
            }
        )
    table = pd.DataFrame(table_rows)
    table.to_csv(out / "model_table.csv", index=False)

    diff_rows = []
    for label, a, b in PAIRS:
        ca, cb = column(*a), column(*b)
        if ca in preds and cb in preds:
            diff_rows.append(difference_row(config, preds, ca, cb, label))
    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(out / "differences.csv", index=False)

    losses = pd.DataFrame(
        {n: squared_pct_error(preds["target"], preds[n]) for n in names}
    )
    losses["position"] = preds["position"].to_numpy()
    by_time = losses.groupby("position").mean().sort_index()
    dm_rows = []
    for ladder, rungs in (("features", FEATURE_LADDER), ("models", MODEL_LADDER)):
        present = [column(*r) for r in rungs if column(*r) in by_time]
        for lower, upper in zip(present[:-1], present[1:], strict=True):
            test = diebold_mariano(by_time[upper], by_time[lower])
            dm_rows.append({"ladder": ladder, "model": upper, "against": lower, **test})
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(out / "dm_tests.csv", index=False)

    metrics = {
        "freeze": freeze,
        "n_rows": len(preds),
        "n_time_ids": int(preds["time_id"].nunique()),
        "naive_rmspe": naive_rmspe,
        "models": {
            n: {"rmspe": r["rmspe"], **info[n]}
            for n, r in zip(names, table_rows, strict=True)
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    show = table.assign(
        model=table["model"].map(LABELS),
        interval=[
            f"{a:.4f} to {b:.4f}"
            for a, b in zip(table["ci_low"], table["ci_high"], strict=True)
        ],
    )
    block = [
        f"Holdout: {metrics['n_time_ids']:,} time_ids, {len(preds):,} rows; every model"
        " refitted on the full development set; naive RMSPE on the same rows"
        f" {naive_rmspe:.4f}.",
        "",
        md_table(
            show,
            {
                "model": "",
                "features": "",
                "rmspe": "{:.4f}",
                "interval": "",
                "seeds": "",
                "train_seconds": "{:.0f}",
            },
        ),
        "",
        "Paired differences on the holdout (95% block-bootstrap intervals):",
        "",
        md_table(
            diffs,
            {
                "comparison": "",
                "rmspe_a": "{:.4f}",
                "rmspe_b": "{:.4f}",
                "difference": "{:+.4f}",
                "ci_low": "{:+.4f}",
                "ci_high": "{:+.4f}",
            },
        ),
        "",
        "Diebold-Mariano tests on adjacent rungs (negative statistic: the model"
        " beats the rung below it):",
        "",
        md_table(
            dm,
            {
                "ladder": "",
                "model": "",
                "against": "",
                "statistic": "{:+.2f}",
                "p_value": "{:.4f}",
                "lag": "{:.0f}",
            },
        ),
    ]
    (Path(paths["results"]) / "block_holdout.md").write_text("\n".join(block), "utf-8")
    write_run_record(out, "holdout", config, frozen["seeds"])
    logger.info("holdout written to %s", out)
    return True
