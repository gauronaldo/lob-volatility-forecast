"""Training on the saved folds: feature frames, selection, runs, tuning (S5, S7).

Every run loads its folds from `lobvol.cv`; nothing here builds a split.
Out-of-fold predictions go to `data/interim/oof/` and metrics to
`results/metrics/`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lobvol.config import load_config
from lobvol.cross_section import (
    KEYS,
    apply_clusters,
    apply_stock_level,
    fit_clusters,
    fit_stock_level,
    same_time_features,
)
from lobvol.cv import (
    fold_masks,
    load_dev_targets,
    load_roles,
    load_split,
    order_status,
    primary_scheme,
)
from lobvol.evaluation import rmspe
from lobvol.features import feature_group
from lobvol.models import blend, forest, gbdt, har, mlp, naive
from lobvol.runlog import write_run_record

logger = logging.getLogger(__name__)

NOT_FEATURES = ["time_id", "position", "target"]
FEATURE_TABLES = {
    "A": [],
    "B": [],
    "C_strict": ["nn_past"],
    "C_comp": ["nn_any"],
    "C_strict_gap": ["nn_past_gap"],
    "D": ["nn_past", "lag"],
    "D_gap": ["nn_past_gap", "lag_gap"],
}
# Feature selection is decided on the strict set and reused by its variants.
SELECTION_BASE = {"C_comp": "C_strict", "C_strict_gap": "C_strict", "D_gap": "D"}
SELECTING = {"gbdt", "forest", "mlp"}
SEEDED = {"gbdt", "mlp"}
_FRAMES: dict[tuple[str, str], pd.DataFrame] = {}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def model_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Load configs/models/<name>.yaml next to the base config."""
    return load_config(
        Path(config.get("config_dir", "configs")) / "models" / f"{name}.yaml"
    )


def run_name(model: str, features: str, scheme: str) -> str:
    """Return the file stem of one run."""
    return f"{model}__{features}__{scheme}"


def oof_path(config: dict[str, Any], name: str) -> Path:
    """Return the path of the out-of-fold predictions of one run."""
    return Path(config["paths"]["interim"]) / "oof" / f"{name}.parquet"


def metrics_path(config: dict[str, Any], name: str) -> Path:
    """Return the path of the metrics of one run."""
    return Path(config["paths"]["results"]) / "metrics" / f"{name}.json"


def log_trial(config: dict[str, Any], record: dict[str, Any]) -> None:
    """Append one configuration tried to results/trials.json (rule 7)."""
    path = Path(config["paths"]["results"]) / "trials.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    trials = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    trials.append(record)
    path.write_text(json.dumps(trials, indent=1), encoding="utf-8")


def resolve_set(config: dict[str, Any], features: str) -> str:
    """Map temporal feature sets to B when the order fallback applies."""
    if order_status(config)["status"] == "ok" or features in ("A", "B"):
        return features
    return "B"


# --------------------------------------------------------------------------
# Feature frames
# --------------------------------------------------------------------------
def assemble_frame(
    bucket: pd.DataFrame,
    rows: pd.DataFrame,
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Join bucket, same-time and temporal features onto the given rows.

    `rows` has stock_id, time_id, position and, when allowed, target. The
    same-time features use every stock of the bucket table at that time_id.
    """
    frame = rows.merge(bucket, on=KEYS).merge(same_time_features(bucket), on=KEYS)
    for table in tables:
        frame = frame.merge(table, on=KEYS, how="left")
    return frame.sort_values(KEYS, ignore_index=True)


def load_frame(config: dict[str, Any], features: str) -> pd.DataFrame:
    """Return the development rows with every fold-independent feature of a set."""
    feat_dir = Path(config["paths"]["features"])
    key = (str(feat_dir), features)
    if key not in _FRAMES:
        bucket = pd.read_parquet(feat_dir / "bucket.parquet")
        roles = load_roles(config)
        dev = roles.loc[roles["role"] == "development", ["time_id", "position"]]
        rows = load_dev_targets(config).merge(dev, on="time_id")
        tables = [
            pd.read_parquet(feat_dir / f"{name}.parquet")
            for name in FEATURE_TABLES[features]
        ]
        _FRAMES[key] = assemble_frame(bucket, rows, tables)
    return _FRAMES[key]


def feature_names(frame: pd.DataFrame) -> list[str]:
    """Return the model inputs of a frame (stock_id included)."""
    return [c for c in frame.columns if c not in NOT_FEATURES]


def fold_frame(
    frame: pd.DataFrame, features: str, train_mask: np.ndarray, config: dict[str, Any]
) -> pd.DataFrame:
    """Add the features fitted on the training rows of one fold (rule 3)."""
    if features == "A":
        return frame
    train = frame[train_mask]
    stats = fit_stock_level(train)
    labels = fit_clusters(
        train, config["features"]["n_clusters"], config["frozen"]["seeds"][0]
    )
    fitted = [apply_stock_level(frame, stats), apply_clusters(frame, labels)]
    return pd.concat([frame, *fitted], axis=1)


def noise_selection(
    train: pd.DataFrame, features: list[str], config: dict[str, Any]
) -> list[str]:
    """Drop the features whose gain is below the best Gaussian noise column."""
    seeds = config["frozen"]["seeds"][: config["features"]["noise_columns"]]
    noise = {
        f"noise_{s}": np.random.default_rng(s).normal(size=len(train)) for s in seeds
    }
    fitted = gbdt.fit(
        train.assign(**noise),
        features + list(noise),
        model_config(config, "gbdt")["ladder"],
        seeds[0],
        config["frozen"]["inner_split_fraction"],
        config["compute"]["threads"],
    )
    gain = gbdt.importance(fitted)
    threshold = gain[list(noise)].max()
    return [f for f in features if f == "stock_id" or gain[f] >= threshold]


def selected_features(
    config: dict[str, Any], features: str, scheme: str, fold: int
) -> list[str]:
    """Return the cached feature list of (feature set, scheme, fold)."""
    base = SELECTION_BASE.get(features, features)
    path = Path(config["paths"]["interim"]) / "selection"
    path = path / f"{base}__{scheme}__fold{fold}.json"
    if not path.exists():
        frame = load_frame(config, base)
        train_mask, _ = fold_masks(load_split(config, scheme), fold)
        full = fold_frame(frame, base, train_mask, config)
        kept = noise_selection(full[train_mask], feature_names(full), config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(kept), encoding="utf-8")
        logger.info("selection %s: kept %d features", path.stem, len(kept))
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------
def tuned_params(config: dict[str, Any]) -> dict[str, Any]:
    """Return the tuned LightGBM parameters, running the one study if needed."""
    path = Path(config["paths"]["results"]) / "gbdt_tuned.json"
    if not path.exists():
        tune_gbdt(config)
    return json.loads(path.read_text(encoding="utf-8"))["params"]


def tune_gbdt(config: dict[str, Any]) -> None:
    """One Optuna TPE study on one feature set and one seed (METHOD.md 5.4)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    frozen, mcfg = config["frozen"], model_config(config, "gbdt")
    scheme = primary_scheme(config)
    features = resolve_set(config, frozen["optuna_feature_set"])
    seed, inner = frozen["optuna_seed"], frozen["inner_split_fraction"]
    threads = config["compute"]["threads"]
    frame, split = load_frame(config, features), load_split(config, scheme)
    folds = []
    for k in range(frozen["walk_forward_folds"]):
        train_mask, val_mask = fold_masks(split, k)
        full = fold_frame(frame, features, train_mask, config)
        cols = selected_features(config, features, scheme, k)
        folds.append((full[train_mask], full[val_mask], cols))

    def objective(trial: optuna.Trial) -> float:
        params = dict(mcfg["tuned"])
        for name, (low, high, log) in mcfg["search"].items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high, log=log)
            else:
                params[name] = trial.suggest_float(name, low, high, log=log)
        scores, trees = [], []
        start = time.perf_counter()
        for train, val, cols in folds:
            fitted = gbdt.fit(train, cols, params, seed, inner, threads)
            scores.append(rmspe(val["target"], gbdt.predict(fitted, val)))
            trees.append(fitted["trees"])
        value = float(np.mean(scores))
        log_trial(
            config,
            {
                "kind": "optuna_trial",
                "number": trial.number,
                "model": "gbdt",
                "features": features,
                "scheme": scheme,
                "seed": seed,
                "params": params,
                "fold_rmspe": scores,
                "trees": trees,
                "rmspe_mean": value,
                "seconds": round(time.perf_counter() - start, 1),
            },
        )
        logger.info("trial %d: %.5f", trial.number, value)
        return value

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=frozen["optuna_trials"])
    best = {**mcfg["tuned"], **study.best_params}
    out = {
        "params": best,
        "rmspe_mean": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
        "features": features,
        "scheme": scheme,
        "seed": seed,
    }
    path = Path(config["paths"]["results"]) / "gbdt_tuned.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------
def fit_predict(
    model: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
    seed: int,
    config: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one model on training rows and predict y for the validation rows."""
    inner = config["frozen"]["inner_split_fraction"]
    threads = config["compute"]["threads"]
    if model == "naive":
        return naive.predict(val), {}
    if model == "har_ridge":
        return har.fit_predict(train, val, model_config(config, model), inner)
    if model == "forest":
        mcfg = model_config(config, model)
        return forest.fit_predict(train, val, cols, mcfg, seed, threads), {}
    if model == "mlp":
        mcfg = model_config(config, model)
        return mlp.fit_predict(train, val, cols, mcfg, seed, inner, threads)
    if model in ("gbdt", "gbdt_fixed"):
        if params is None:
            params = (
                tuned_params(config)
                if model == "gbdt"
                else model_config(config, "gbdt")["ladder"]
            )
        fitted = gbdt.fit(train, cols, params, seed, inner, threads)
        return gbdt.predict(fitted, val), {"trees": fitted["trees"], "fitted": fitted}
    raise ValueError(f"unknown model: {model}")


def summarise(oof: pd.DataFrame, n_folds: int) -> dict[str, Any]:
    """Return fold and pooled RMSPE of a model and of naive on the same rows."""
    folds = []
    for k in range(n_folds):
        part = oof[oof["fold"] == k]
        folds.append(
            {
                "fold": k,
                "n_rows": len(part),
                "rmspe": rmspe(part["target"], part["pred"]),
                "naive_rmspe": rmspe(part["target"], part["rv1_600"]),
            }
        )
    scored = oof[oof["fold"] >= 0]
    values = [f["rmspe"] for f in folds]
    return {
        "folds": folds,
        "rmspe_mean": float(np.mean(values)),
        "rmspe_sd": float(np.std(values, ddof=1)),
        "rmspe_pooled": rmspe(scored["target"], scored["pred"]),
        "naive_pooled": rmspe(scored["target"], scored["rv1_600"]),
        "naive_mean": float(np.mean([f["naive_rmspe"] for f in folds])),
        "n_rows": len(scored),
    }


def run_one(
    config: dict[str, Any],
    model: str,
    features: str,
    scheme: str,
    force: bool = False,
    drop_group: str | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Train one model on one feature set under one scheme; write OOF and metrics."""
    features = resolve_set(config, features)
    if model == "blend":
        return run_blend(config, features, scheme, force)
    name = run_name(model, features, scheme)
    if drop_group is not None:
        name = f"ablation__{drop_group.replace(' ', '_')}__{name}"
    if metrics_path(config, name).exists() and not force:
        logger.info("%s exists; skipping (use --force to redo)", name)
        return json.loads(metrics_path(config, name).read_text(encoding="utf-8"))
    frozen = config["frozen"]
    n_folds = frozen["walk_forward_folds"]
    if seeds is None:
        seeds = frozen["seeds"] if model in SEEDED else frozen["seeds"][:1]
    if model == "gbdt":
        tuned_params(config)  # the one-off study must not count as training time
    frame, split = load_frame(config, features), load_split(config, scheme)
    if not frame[KEYS].equals(split[KEYS]):
        raise RuntimeError("feature frame and split rows are not aligned")
    preds = np.full((len(frame), len(seeds)), np.nan)
    fold_of = np.full(len(frame), -1)
    seconds, extras, n_features = 0.0, [], []
    for k in range(n_folds):
        train_mask, val_mask = fold_masks(split, k)
        full = fold_frame(frame, features, train_mask, config)
        cols = feature_names(full)
        if model in SELECTING:
            kept = set(selected_features(config, features, scheme, k))
            cols = [c for c in cols if c in kept]
        if drop_group is not None:
            cols = [c for c in cols if feature_group(c) != drop_group]
        train, val = full[train_mask], full[val_mask]
        fold_of[val_mask] = k
        n_features.append(len(cols))
        for j, seed in enumerate(seeds):
            start = time.perf_counter()
            pred, extra = fit_predict(model, train, val, cols, seed, config)
            seconds += time.perf_counter() - start
            preds[val_mask, j] = pred
            fitted = extra.pop("fitted", None)
            extras.append({"fold": k, "seed": seed, **extra})
        if (
            drop_group is None
            and fitted is not None
            and k == n_folds - 1
            and [model, features] == config["runs"]["shap"]["target"]
            and scheme == primary_scheme(config)
        ):
            save_shap(config, fitted, val)
        logger.info("%s fold %d done (%d features)", name, k, len(cols))

    oof = frame[KEYS + ["target", "rv1_600"]].copy()
    oof["fold"] = fold_of
    oof["pred"] = preds.mean(axis=1)
    for j, seed in enumerate(seeds):
        oof[f"pred_s{seed}"] = preds[:, j]
    metrics = summarise(oof, n_folds)
    metrics.update(
        model=model,
        features=features,
        scheme=scheme,
        drop_group=drop_group,
        seeds=seeds,
        n_features=n_features,
        train_seconds=round(seconds, 1),
        fits=extras,
        tuned=model == "gbdt",
    )
    out = oof_path(config, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    oof.drop(columns=["target", "rv1_600"]).to_parquet(out, index=False)
    metrics_path(config, name).parent.mkdir(parents=True, exist_ok=True)
    metrics_path(config, name).write_text(json.dumps(metrics, indent=1), "utf-8")
    log_trial(config, {"kind": "run", "name": name, **metrics})
    write_run_record(
        metrics_path(config, name).parent / "run_records", name, config, seeds
    )
    logger.info(
        "%s: RMSPE %.5f (naive %.5f)",
        name,
        metrics["rmspe_mean"],
        metrics["naive_mean"],
    )
    return metrics


def run_blend(
    config: dict[str, Any], features: str, scheme: str, force: bool = False
) -> dict[str, Any]:
    """Blend member OOF predictions with weights fitted on earlier folds only."""
    name = run_name("blend", features, scheme)
    if metrics_path(config, name).exists() and not force:
        return json.loads(metrics_path(config, name).read_text(encoding="utf-8"))
    n_folds = config["frozen"]["walk_forward_folds"]
    members = [
        (m, resolve_set(config, f)) for m, f in model_config(config, "blend")["members"]
    ]
    frame = load_frame(config, "A")
    oof = frame[KEYS + ["target", "rv1_600"]].copy()
    for m, f in members:
        run_one(config, m, f, scheme)
        member = pd.read_parquet(oof_path(config, run_name(m, f, scheme)))
        oof["fold"] = member["fold"].to_numpy()
        oof[f"{m}__{f}"] = member["pred"].to_numpy()
    cols = [f"{m}__{f}" for m, f in members]
    oof["pred"] = np.nan
    weights = []
    temporal = scheme in ("walk_forward", "blocked_time")
    for k in range(n_folds):
        val = oof["fold"] == k
        # Out-of-fold predictions only: earlier folds in temporal schemes.
        fit_rows = (oof["fold"] >= 0) & ((oof["fold"] < k) if temporal else ~val)
        if fit_rows.any():
            part = oof[fit_rows]
            w = blend.fit_weights(part[cols].to_numpy(), part["target"].to_numpy())
        else:
            w = np.full(len(cols), 1.0 / len(cols))
        weights.append(dict(zip(cols, [round(float(x), 4) for x in w], strict=True)))
        oof.loc[val, "pred"] = blend.predict(oof.loc[val, cols].to_numpy(), w)
    metrics = summarise(oof, n_folds)
    metrics.update(
        model="blend",
        features=features,
        scheme=scheme,
        drop_group=None,
        seeds=[],
        members=cols,
        weights=weights,
        train_seconds=0.0,
        tuned=False,
    )
    out = oof_path(config, name)
    oof[KEYS + ["fold", "pred"]].to_parquet(out, index=False)
    metrics_path(config, name).write_text(json.dumps(metrics, indent=1), "utf-8")
    log_trial(config, {"kind": "run", "name": name, **metrics})
    logger.info("%s: RMSPE %.5f", name, metrics["rmspe_mean"])
    return metrics


def save_shap(
    config: dict[str, Any], fitted: dict[str, Any], val: pd.DataFrame
) -> None:
    """Write mean |SHAP| per feature on a sample of validation rows.

    Uses shap.TreeExplainer. If shap cannot be imported (its numba dependency
    lags behind NumPy releases), LightGBM's own TreeSHAP is used instead; it
    is the routine TreeExplainer calls for LightGBM models.
    """
    rows = min(config["runs"]["shap"]["rows"], len(val))
    sample = val[fitted["features"]].sample(rows, random_state=0)
    try:
        import shap

        values = shap.TreeExplainer(fitted["booster"]).shap_values(sample)
        method = "shap.TreeExplainer"
    except ImportError:
        values = fitted["booster"].predict(sample, pred_contrib=True)[:, :-1]
        method = "lightgbm pred_contrib (TreeSHAP)"
    table = pd.DataFrame(
        {"feature": fitted["features"], "mean_abs_shap": np.abs(values).mean(axis=0)}
    )
    table["group"] = table["feature"].map(feature_group)
    table["method"] = method
    table["rows"] = rows
    path = Path(config["paths"]["results"]) / "shap.csv"
    table.sort_values("mean_abs_shap", ascending=False).to_csv(path, index=False)


def check_s5_gates(config: dict[str, Any]) -> bool:
    """Check the sanity anchors of S5 on development rows and record them."""
    gates = config["gates"]
    frame = load_frame(config, "A")
    naive_all = rmspe(frame["target"], frame["rv1_600"])
    sanity_path = metrics_path(config, run_name("gbdt_fixed", "A", "group_time"))
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))["rmspe_mean"]
    low, high = gates["naive_rmspe"]
    result = {
        "naive_rmspe_development_rows": naive_all,
        "naive_range": [low, high],
        "naive_ok": bool(low <= naive_all <= high),
        "sanity_rmspe_group_time": sanity,
        "sanity_max": gates["sanity_rmspe_max"],
        "sanity_ok": bool(sanity < gates["sanity_rmspe_max"]),
        "n_rows": len(frame),
    }
    path = Path(config["paths"]["results"]) / "gates_s5.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("S5 gates: %s", result)
    return result["naive_ok"] and result["sanity_ok"]


def run_matrix(config: dict[str, Any], force: bool = False) -> bool:
    """Run the leakage ladder, check the S5 gates, then the model ladder."""
    runs = config["runs"]
    ok = order_status(config)["status"] == "ok"
    schemes = runs["ladder"]["schemes"] if ok else ["kfold_rows", "group_time"]
    for scheme in schemes:
        run_one(
            config, runs["ladder"]["model"], runs["ladder"]["features"], scheme, force
        )
    if not check_s5_gates(config):
        logger.error("S5 gate failed; stopping before the model ladder (rule 10)")
        return False
    primary = primary_scheme(config)
    done = set()
    for model, features in runs["models"]:
        key = (model, resolve_set(config, features))
        if key not in done:
            run_one(config, model, features, primary, force)
            done.add(key)
    ablation = runs["ablation"]
    features = resolve_set(config, ablation["features"])
    full = load_frame(config, features)
    sample_mask = np.ones(len(full), dtype=bool)
    groups = {
        feature_group(c)
        for c in feature_names(fold_frame(full, features, sample_mask, config))
    }
    base_seed = [ablation["seed"]]
    for group in sorted(groups):
        run_one(config, ablation["model"], features, primary, force, group, base_seed)
    return True
