"""Holdout cut, the four split schemes and the embargo (S4).

See METHOD.md section 5.2. Every model loads the folds saved here; no model
builds its own split. Development targets are only reachable through
`load_dev_targets`, which never returns embargo or holdout rows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lobvol.data import TRAIN_CSV, read_targets
from lobvol.runlog import write_run_record

logger = logging.getLogger(__name__)

UNUSED, TRAIN, VAL = 0, 1, 2
SCHEMES = ["kfold_rows", "group_time", "blocked_time", "walk_forward"]
TEMPORAL_SCHEMES = ["blocked_time", "walk_forward"]
AUDIT_MARKER = "## Target distribution (development rows, added by S4)"


def cut_holdout(position: pd.DataFrame, fraction: float, embargo: int) -> pd.DataFrame:
    """Label time_ids as development, embargo or holdout along the recovered order."""
    roles = position.sort_values("position", ignore_index=True).copy()
    n = len(roles)
    n_holdout = int(round(n * fraction))
    role = np.full(n, "development", dtype=object)
    role[n - n_holdout :] = "holdout"
    role[n - n_holdout - embargo : n - n_holdout] = "embargo"
    roles["role"] = role
    return roles


def cut_holdout_random(position: pd.DataFrame, fraction: float, seed: int):
    """Fallback of section 8.1: a random share of time_ids is the holdout."""
    roles = position.sort_values("time_id", ignore_index=True).copy()
    rng = np.random.default_rng(seed)
    n_holdout = int(round(len(roles) * fraction))
    role = np.full(len(roles), "development", dtype=object)
    role[rng.permutation(len(roles))[:n_holdout]] = "holdout"
    roles["role"] = role
    return roles


def _blocks(n: int, n_blocks: int) -> np.ndarray:
    """Assign n ordered items to contiguous blocks of nearly equal size."""
    return np.minimum(np.arange(n) * n_blocks // n, n_blocks - 1)


def time_folds(
    n_time: int, scheme: str, n_folds: int, embargo: int, seed: int
) -> np.ndarray:
    """Return an (n_time, n_folds) role matrix for time_ids in temporal order."""
    roles = np.full((n_time, n_folds), UNUSED, dtype=np.int8)
    index = np.arange(n_time)
    if scheme == "group_time":
        group = np.empty(n_time, dtype=np.int64)
        group[np.random.default_rng(seed).permutation(n_time)] = _blocks(
            n_time, n_folds
        )
        for k in range(n_folds):
            roles[:, k] = np.where(group == k, VAL, TRAIN)
    elif scheme == "blocked_time":
        block = _blocks(n_time, n_folds)
        for k in range(n_folds):
            val = index[block == k]
            near = (index >= val[0] - embargo) & (index <= val[-1] + embargo)
            roles[:, k] = np.where(block == k, VAL, np.where(near, UNUSED, TRAIN))
    elif scheme == "walk_forward":
        block = _blocks(n_time, n_folds + 1)
        for k in range(n_folds):
            val_start = index[block == k + 1][0]
            roles[index < val_start - embargo, k] = TRAIN
            roles[block == k + 1, k] = VAL
    else:
        raise ValueError(f"not a time-based scheme: {scheme}")
    return roles


def make_split(
    rows: pd.DataFrame, scheme: str, n_folds: int, embargo: int, seed: int
) -> pd.DataFrame:
    """Return the fold roles of every development row for one scheme.

    `rows` has stock_id, time_id and position (development rows only).
    """
    out = rows[["stock_id", "time_id"]].reset_index(drop=True).copy()
    if scheme == "kfold_rows":
        fold = np.empty(len(out), dtype=np.int64)
        fold[np.random.default_rng(seed).permutation(len(out))] = _blocks(
            len(out), n_folds
        )
        for k in range(n_folds):
            out[f"fold_{k}"] = np.where(fold == k, VAL, TRAIN).astype(np.int8)
        return out
    ordered = rows[["time_id", "position"]].drop_duplicates().sort_values("position")
    roles = time_folds(len(ordered), scheme, n_folds, embargo, seed)
    lookup = pd.DataFrame(
        roles, index=ordered["time_id"].to_numpy(), columns=range(n_folds)
    )
    for k in range(n_folds):
        out[f"fold_{k}"] = lookup[k].reindex(out["time_id"]).to_numpy()
    return out


def inner_split(position: np.ndarray, fraction: float) -> np.ndarray:
    """Mark the rows whose time_id is among the last share of training time_ids."""
    ordered = np.unique(position)
    n_inner = max(1, int(round(len(ordered) * fraction)))
    return position >= ordered[len(ordered) - n_inner]


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def order_status(config: dict[str, Any]) -> dict[str, Any]:
    """Return the status written by the time-order stage."""
    path = Path(config["paths"]["interim"]) / "order_status.json"
    return json.loads(path.read_text(encoding="utf-8"))


def primary_scheme(config: dict[str, Any]) -> str:
    """Return walk_forward, or group_time when the order fallback applies."""
    return "walk_forward" if order_status(config)["status"] == "ok" else "group_time"


def load_roles(config: dict[str, Any]) -> pd.DataFrame:
    """Return time_id, position and role (development, embargo, holdout)."""
    return pd.read_parquet(Path(config["paths"]["splits"]) / "holdout.parquet")


def load_split(config: dict[str, Any], scheme: str) -> pd.DataFrame:
    """Return the saved fold roles of one scheme."""
    return pd.read_parquet(Path(config["paths"]["splits"]) / f"{scheme}.parquet")


def load_dev_targets(config: dict[str, Any]) -> pd.DataFrame:
    """Return the targets of development rows only."""
    roles = load_roles(config)
    dev = roles.loc[roles["role"] == "development", "time_id"]
    return read_targets(config["paths"]["raw"], dev)


def fold_masks(split: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean training and validation masks of one fold."""
    role = split[f"fold_{fold}"].to_numpy()
    return role == TRAIN, role == VAL


def _target_section(targets: pd.DataFrame) -> list[str]:
    """Describe the target distribution of the development rows."""
    y = targets["target"]
    q = y.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return [
        AUDIT_MARKER,
        "",
        f"- Development rows: {len(y):,}; time_ids: {targets['time_id'].nunique():,}",
        f"- dtype: {y.dtype}; NaN: {int(y.isna().sum())}; zeros:"
        f" {int((y == 0).sum())}; non-positive: {int((y <= 0).sum())}",
        "",
        "| Quantile | target |",
        "| --- | --- |",
        *[f"| {p:.2f} | {v:.6f} |" for p, v in q.items()],
        "",
    ]


def run_splits(config: dict[str, Any], force: bool = False) -> Path:
    """Cut the holdout, save every split scheme and record the target distribution."""
    paths, frozen = config["paths"], config["frozen"]
    split_dir = Path(paths["splits"])
    holdout_path = split_dir / "holdout.parquet"
    if holdout_path.exists() and not force:
        logger.info("%s exists; skipping (use --force to redo)", holdout_path)
        return holdout_path
    split_dir.mkdir(parents=True, exist_ok=True)
    seed = frozen["seeds"][0]
    position = pd.read_parquet(Path(paths["interim"]) / "time_order.parquet")
    ok = order_status(config)["status"] == "ok"
    if ok:
        roles = cut_holdout(position, frozen["holdout_fraction"], frozen["embargo"])
    else:
        roles = cut_holdout_random(position, frozen["holdout_fraction"], seed)
    roles.to_parquet(holdout_path, index=False)

    pairs = pd.read_csv(Path(paths["raw"]) / TRAIN_CSV, usecols=["stock_id", "time_id"])
    dev = roles[roles["role"] == "development"]
    rows = pairs.merge(dev[["time_id", "position"]], on="time_id")
    rows = rows.sort_values(["stock_id", "time_id"], ignore_index=True)
    schemes = SCHEMES if ok else ["kfold_rows", "group_time"]
    for scheme in schemes:
        split = make_split(
            rows, scheme, frozen["walk_forward_folds"], frozen["embargo"], seed
        )
        split.to_parquet(split_dir / f"{scheme}.parquet", index=False)

    audit = Path(paths["results"]) / "audit.md"
    text = audit.read_text(encoding="utf-8") if audit.exists() else ""
    text = text.split(AUDIT_MARKER)[0].rstrip("\n")
    section = _target_section(load_dev_targets(config))
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(text + "\n\n" + "\n".join(section), encoding="utf-8")
    write_run_record(Path(paths["results"]) / "run_records", "splits", config, [seed])
    counts = roles["role"].value_counts().to_dict()
    logger.info("holdout cut: %s; schemes: %s", counts, schemes)
    return holdout_path
