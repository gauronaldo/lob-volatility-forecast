"""`python -m lobvol evaluate`: tables, intervals, figures and report blocks.

Everything here is regenerated from the out-of-fold predictions and metrics
written by the training runs. Numbers in `reports/report.md` and `README.md`
live between `<!-- BEGIN:name -->` and `<!-- END:name -->` markers and are
replaced on every run.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lobvol.cross_section import KEYS
from lobvol.cv import order_status, primary_scheme
from lobvol.evaluation import (
    block_indices,
    bootstrap_rmspe,
    interval,
    per_time,
    rmspe,
    squared_pct_error,
)
from lobvol.pipeline import load_frame, oof_path, run_name
from lobvol.runlog import write_run_record

logger = logging.getLogger(__name__)

# Reference palette of the dataviz guide (light surface): first three
# categorical slots, which are distinguishable in every pairing.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

LABELS = {
    "naive": "Naive (RV of the last 10 min)",
    "har_ridge": "HAR ridge",
    "forest": "Random forest",
    "gbdt": "LightGBM",
    "gbdt_fixed": "LightGBM (fixed)",
    "mlp": "MLP",
    "blend": "Blend",
}
SCHEME_LABELS = {
    "kfold_rows": "Random rows (5-fold)",
    "group_time": "Grouped by time_id",
    "blocked_time": "Blocked in time + embargo",
    "walk_forward": "Walk-forward + embargo",
}
SET_LABELS = {
    "A": "A: in-bucket + same-time market",
    "B": "B: A + stock level + clusters",
    "C_strict": "C_strict: B + past neighbours",
    "C_strict_gap": "C_strict_gap: neighbours 10+ back",
    "C_comp": "C_comp: B + any neighbours (look-ahead)",
    "D": "D: C_strict + lags",
    "D_gap": "D_gap: lags and neighbours 10+ back",
}


# --------------------------------------------------------------------------
# Scored frames and bootstrap
# --------------------------------------------------------------------------
def base_frame(config: dict[str, Any]) -> pd.DataFrame:
    """Development rows with position, target and the naive forecast."""
    cols = KEYS + ["position", "target", "rv1_600"]
    return load_frame(config, "A")[cols].copy()


def add_predictions(
    config: dict[str, Any], frame: pd.DataFrame, runs: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """Add one column per run; rows not validated by a run get NaN.

    `runs` maps a column name to (run name, prediction column).
    """
    frame = frame.copy()
    for column, (name, pred) in runs.items():
        oof = pd.read_parquet(oof_path(config, name))
        values = oof[pred].where(oof["fold"] >= 0)
        frame[column] = values.to_numpy()
    return frame


def bootstrap(
    config: dict[str, Any], frame: pd.DataFrame, columns: list[str]
) -> tuple[dict[str, float], np.ndarray, int]:
    """Pooled RMSPE and paired block-bootstrap samples on the common rows."""
    frozen = config["frozen"]
    common = frame.dropna(subset=columns)
    point = {c: rmspe(common["target"], common[c]) for c in columns}
    sums = per_time(common, columns)
    index = block_indices(
        len(sums),
        frozen["bootstrap_block_length"],
        frozen["bootstrap_resamples"],
        frozen["seeds"][0],
    )
    return point, bootstrap_rmspe(sums, columns, index), len(common)


def difference_row(
    config: dict[str, Any], frame: pd.DataFrame, a: str, b: str, label: str
) -> dict[str, Any]:
    """RMSPE(a) - RMSPE(b) on the common rows with a paired 95% interval."""
    point, samples, n_rows = bootstrap(config, frame, [a, b])
    diff = samples[:, 0] - samples[:, 1]
    low, high = interval(diff)
    return {
        "comparison": label,
        "a": a,
        "b": b,
        "rmspe_a": point[a],
        "rmspe_b": point[b],
        "difference": point[a] - point[b],
        "ci_low": low,
        "ci_high": high,
        "share_resamples_a_better": float((diff < 0).mean()),
        "n_rows": n_rows,
    }


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def collect_metrics(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Read every metrics file of the model and leakage ladders."""
    folder = Path(config["paths"]["results"]) / "metrics"
    out = []
    for path in sorted(folder.glob("*.json")):
        if not path.name.startswith("ablation__"):
            out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def model_table(config: dict[str, Any], base: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, feature set, scheme), as required by section 11."""
    rows = []
    for m in collect_metrics(config):
        name = run_name(m["model"], m["features"], m["scheme"])
        frame = add_predictions(config, base, {"pred": (name, "pred")})
        _, samples, _ = bootstrap(config, frame, ["pred"])
        low, high = interval(samples[:, 0])
        rows.append(
            {
                "model": m["model"],
                "features": m["features"],
                "scheme": m["scheme"],
                "rmspe_mean": m["rmspe_mean"],
                "rmspe_sd_folds": m["rmspe_sd"],
                "rmspe_pooled": m["rmspe_pooled"],
                "ci_low": low,
                "ci_high": high,
                "naive_rmspe_mean": m["naive_mean"],
                "naive_rmspe_pooled": m["naive_pooled"],
                "n_rows": m["n_rows"],
                "seeds": " ".join(str(s) for s in m["seeds"]) or "-",
                "n_features": int(np.median(m.get("n_features", [0]) or [0])),
                "train_seconds": m["train_seconds"],
                "tuned_on_this_metric": bool(m.get("tuned", False)),
            }
        )
    return pd.DataFrame(rows).sort_values(["scheme", "rmspe_mean"], ignore_index=True)


def difference_table(config: dict[str, Any], base: pd.DataFrame) -> pd.DataFrame:
    """Paired differences that answer the research questions."""
    p = primary_scheme(config)
    pairs = [
        (
            "RQ3 look-ahead: C_strict minus C_comp",
            ("gbdt", "C_strict", p),
            ("gbdt", "C_comp", p),
        ),
        ("RQ1: LightGBM A minus naive", ("gbdt", "A", p), ("naive", "A", p)),
        ("RQ1: B minus A", ("gbdt", "B", p), ("gbdt", "A", p)),
        ("RQ1: C_strict minus B", ("gbdt", "C_strict", p), ("gbdt", "B", p)),
        ("RQ1: D minus C_strict", ("gbdt", "D", p), ("gbdt", "C_strict", p)),
        ("RQ1: D minus B", ("gbdt", "D", p), ("gbdt", "B", p)),
        (
            "Order error: C_strict_gap minus C_strict",
            ("gbdt", "C_strict_gap", p),
            ("gbdt", "C_strict", p),
        ),
        (
            "Order error: C_strict_gap minus B",
            ("gbdt", "C_strict_gap", p),
            ("gbdt", "B", p),
        ),
        ("Order error: D_gap minus D", ("gbdt", "D_gap", p), ("gbdt", "D", p)),
        ("Order error: D_gap minus B", ("gbdt", "D_gap", p), ("gbdt", "B", p)),
        ("Model: HAR ridge minus naive", ("har_ridge", "A", p), ("naive", "A", p)),
        ("Model: forest minus HAR ridge", ("forest", "D", p), ("har_ridge", "A", p)),
        ("Model: LightGBM D minus forest", ("gbdt", "D", p), ("forest", "D", p)),
        ("RQ4: MLP minus LightGBM D", ("mlp", "D", p), ("gbdt", "D", p)),
        ("Model: blend minus LightGBM D", ("blend", "D", p), ("gbdt", "D", p)),
        (
            "RQ2: group_time minus kfold_rows",
            ("gbdt_fixed", "A", "group_time"),
            ("gbdt_fixed", "A", "kfold_rows"),
        ),
        (
            "RQ2: blocked_time minus group_time",
            ("gbdt_fixed", "A", "blocked_time"),
            ("gbdt_fixed", "A", "group_time"),
        ),
        (
            "RQ2: walk_forward minus blocked_time",
            ("gbdt_fixed", "A", "walk_forward"),
            ("gbdt_fixed", "A", "blocked_time"),
        ),
    ]
    rows = []
    for label, a, b in pairs:
        names = {"a": run_name(*a), "b": run_name(*b)}
        if not all(oof_path(config, n).exists() for n in names.values()):
            continue
        frame = add_predictions(
            config, base, {k: (n, "pred") for k, n in names.items()}
        )
        row = difference_row(config, frame, "a", "b", label)
        row.update(a=names["a"], b=names["b"])
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_table(config: dict[str, Any], base: pd.DataFrame) -> pd.DataFrame:
    """Change in RMSPE when one feature group is removed from the tuned model."""
    spec = config["runs"]["ablation"]
    p = primary_scheme(config)
    full_name = run_name(spec["model"], spec["features"], p)
    folder = Path(config["paths"]["results"]) / "metrics"
    rows = []
    for path in sorted(folder.glob(f"ablation__*__{full_name}.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        runs = {
            "without": (path.stem, "pred"),
            "full": (full_name, f"pred_s{spec['seed']}"),
        }
        frame = add_predictions(config, base, runs)
        row = difference_row(config, frame, "without", "full", m["drop_group"])
        rows.append(
            {
                "group": m["drop_group"],
                "rmspe_without_group": row["rmspe_a"],
                "rmspe_full": row["rmspe_b"],
                "delta": row["difference"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "n_rows": row["n_rows"],
                "seed": spec["seed"],
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("delta", ascending=False, ignore_index=True)


def shap_groups(config: dict[str, Any]) -> pd.DataFrame:
    """Aggregate mean |SHAP| by feature group."""
    path = Path(config["paths"]["results"]) / "shap.csv"
    if not path.exists():
        return pd.DataFrame()
    shap = pd.read_csv(path)
    groups = shap.groupby("group")["mean_abs_shap"].agg(["sum", "count"])
    groups["share"] = groups["sum"] / groups["sum"].sum()
    groups = groups.rename(columns={"sum": "mean_abs_shap", "count": "n_features"})
    return groups.sort_values("share", ascending=False).reset_index()


def error_analysis(config: dict[str, Any], base: pd.DataFrame) -> dict[str, Any]:
    """RMSPE by stock, market-volatility decile and position; calibration."""
    p = primary_scheme(config)
    results = Path(config["paths"]["results"])
    wanted = {
        "har_ridge": run_name("har_ridge", "A", p),
        "gbdt_A": run_name("gbdt", "A", p),
        "gbdt_D": run_name("gbdt", "D", p),
        "blend": run_name("blend", "D", p),
    }
    runs = {k: (n, "pred") for k, n in wanted.items() if oof_path(config, n).exists()}
    frame = add_predictions(config, base, runs).dropna(subset=list(runs))
    frame["naive"] = frame["rv1_600"]
    models = ["naive", *runs]
    best = "gbdt_D" if "gbdt_D" in runs else models[-1]
    market = frame.groupby("time_id")["rv1_600"].transform("mean")
    by_time = frame.assign(market=market).drop_duplicates("time_id")
    edges = by_time["market"].quantile(np.linspace(0, 1, 11)).to_numpy()
    frame["decile"] = np.clip(np.searchsorted(edges, market, side="right"), 1, 10)
    frame["time_bin"] = pd.qcut(frame["position"], 10, labels=False) + 1

    def grouped(key: str) -> pd.DataFrame:
        rows = []
        for value, part in frame.groupby(key):
            row = {key: value, "n_rows": len(part)}
            row.update({m: rmspe(part["target"], part[m]) for m in models})
            rows.append(row)
        return pd.DataFrame(rows)

    by_decile, by_bin, by_stock = (
        grouped("decile"),
        grouped("time_bin"),
        grouped("stock_id"),
    )
    by_decile.to_csv(results / "error_by_market_rv_decile.csv", index=False)
    by_bin.to_csv(results / "error_by_time.csv", index=False)
    by_stock.sort_values(best, ascending=False).to_csv(
        results / "error_by_stock.csv", index=False
    )

    frame["pred_decile"] = pd.qcut(frame[best], 10, labels=False) + 1
    calib = frame.groupby("pred_decile").agg(
        mean_predicted=(best, "mean"),
        mean_realized=("target", "mean"),
        n=(best, "size"),
    )
    calib["ratio"] = calib["mean_predicted"] / calib["mean_realized"]
    calib.reset_index().to_csv(results / "calibration.csv", index=False)

    audit = results / "audit_stocks.csv"
    out: dict[str, Any] = {"models": models, "best": best, "n_rows": len(frame)}
    if audit.exists():
        stocks = pd.read_csv(audit)
        least = int(stocks.loc[stocks["median_updates"].idxmin(), "stock_id"])
        rest = frame[frame["stock_id"] != least]
        out["least_liquid_stock"] = least
        out["rmspe_all"] = rmspe(frame["target"], frame[best])
        out["rmspe_without_least_liquid"] = rmspe(rest["target"], rest[best])
        only = frame[frame["stock_id"] == least]
        out["rmspe_least_liquid_only"] = rmspe(only["target"], only[best])
    worst = by_stock.sort_values(best, ascending=False).head(5)
    out["worst_stocks"] = worst[["stock_id", best, "naive"]].round(4).to_dict("records")
    out["decile"] = by_decile.round(5).to_dict("records")
    (results / "error_summary.json").write_text(json.dumps(out, indent=1), "utf-8")
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _figure(n_rows: int, title: str, subtitle: str, width: float = 8.0):
    """Create a figure with a finding as title and the context as subtitle."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = textwrap.fill(title, 74)
    subtitle = textwrap.fill(subtitle, 104)
    # 0.6 in below the subtitle leaves room for a legend above the plot.
    head = 0.34 * (title.count("\n") + 1) + 0.24 * (subtitle.count("\n") + 1) + 0.6
    height = head + 0.46 * n_rows + 0.9
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.text(
        0.02,
        1 - 0.12 / height,
        title,
        ha="left",
        va="top",
        color=INK,
        fontsize=12.5,
        fontweight="bold",
        linespacing=1.25,
    )
    y_sub = 1 - (0.12 + 0.34 * (title.count("\n") + 1) + 0.06) / height
    fig.text(
        0.02,
        y_sub,
        subtitle,
        ha="left",
        va="top",
        color=INK_2,
        fontsize=8.5,
        linespacing=1.3,
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=1 - head / height, bottom=0.75 / height)
    return fig, ax


def _dots(ax, labels, values, lows, highs, colors, fmt="{:.4f}") -> None:
    """Draw a dot with its 95% interval per row and the value at the end."""
    y = np.arange(len(labels))[::-1]
    for yi, v, lo, hi, c in zip(y, values, lows, highs, colors, strict=True):
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(
            v,
            yi,
            "o",
            color=c,
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.annotate(
            fmt.format(v),
            (hi, yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=8.5,
        )
    ax.set_yticks(y, labels)
    for tick in ax.get_yticklabels():
        tick.set_color(INK_2)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.margins(x=0.12)


def _save(fig, config: dict[str, Any], name: str) -> str:
    """Save a figure under results/figures and return its file name."""
    import matplotlib.pyplot as plt

    folder = Path(config["paths"]["results"]) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return name


def _legend(ax, entries: list[tuple[str, str]]) -> None:
    """Add a quiet legend; text stays in ink, the marks carry the colour."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=c,
            linewidth=2,
            markersize=7,
            markeredgecolor=SURFACE,
            label=label,
        )
        for label, c in entries
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=1.6,
        ncol=len(entries),
        frameon=False,
        fontsize=8.5,
    )
    for text in legend.get_texts():
        text.set_color(INK_2)


def _rows(table: pd.DataFrame, **where: str) -> pd.DataFrame:
    """Filter the model table."""
    mask = np.ones(len(table), dtype=bool)
    for key, value in where.items():
        mask &= (table[key] == value).to_numpy()
    return table[mask]


def figure_ladder(config, table: pd.DataFrame, diffs: pd.DataFrame) -> str | None:
    """Figure 1: one fixed LightGBM scored under every validation scheme."""
    rows = _rows(table, model="gbdt_fixed", features="A").set_index("scheme")
    order = [s for s in SCHEME_LABELS if s in rows.index]
    if len(order) < 2:
        return None
    rows = rows.loc[order]
    first, last = rows.iloc[0], rows.iloc[-1]
    rise = last["rmspe_pooled"] / first["rmspe_pooled"] - 1
    title = (
        f"The same model scores {first['rmspe_pooled']:.3f} on random rows and"
        f" {last['rmspe_pooled']:.3f} walk-forward: {rise:+.0%} RMSPE once validation"
        " respects time"
    )
    subtitle = (
        "Pooled out-of-fold RMSPE with 95% block-bootstrap intervals. LightGBM, fixed"
        f" hyperparameters and trees, feature set A, seed 0, development rows only"
        f" ({int(rows['n_rows'].max()):,} rows; walk-forward validates"
        f" {int(last['n_rows']):,}). Naive on the same rows: about"
        f" {last['naive_rmspe_pooled']:.3f}."
    )
    fig, ax = _figure(len(order), title, subtitle)
    colors = [BLUE if s == primary_scheme(config) else MUTED for s in order]
    _dots(
        ax,
        [SCHEME_LABELS[s] for s in order],
        rows["rmspe_pooled"],
        rows["ci_low"],
        rows["ci_high"],
        colors,
    )
    ax.set_xlabel("RMSPE (lower is better)", color=INK_2, fontsize=8.5)
    fig.subplots_adjust(left=0.30)
    return _save(fig, config, "fig1_leakage_ladder.png")


def figure_neighbours(config, table: pd.DataFrame, diffs: pd.DataFrame) -> str | None:
    """Figure 2: strict against competition neighbours."""
    p = primary_scheme(config)
    rows = _rows(table, model="gbdt", scheme=p).set_index("features")
    order = [s for s in ["B", "C_strict_gap", "C_strict", "C_comp"] if s in rows.index]
    rq3 = diffs[diffs["comparison"].str.startswith("RQ3")]
    if len(order) < 3 or rq3.empty:
        return None
    d = rq3.iloc[0]
    gain = rows.loc["B", "rmspe_pooled"] - rows.loc["C_comp", "rmspe_pooled"]
    share = d["difference"] / gain if gain > 0 else float("nan")
    if d["ci_low"] > 0:
        title = (
            f"Neighbours from the future lower RMSPE by {d['difference']:.4f}"
            f" (95% CI {d['ci_low']:.4f} to {d['ci_high']:.4f}): {share:.0%} of the"
            " neighbour gain is look-ahead"
        )
    else:
        title = (
            f"No clear gain from future neighbours: C_strict minus C_comp ="
            f" {d['difference']:+.4f} (95% CI {d['ci_low']:+.4f} to"
            f" {d['ci_high']:+.4f})"
        )
    subtitle = (
        f"Pooled out-of-fold RMSPE under {p}, 95% block-bootstrap intervals. Tuned"
        " LightGBM, same hyperparameters and feature list for every set, seeds 0-2,"
        f" {int(rows['n_rows'].max()):,} development rows. Development RMSPE of a tuned"
        " model is optimistic; the paired difference is the quantity to read."
    )
    fig, ax = _figure(len(order), title, subtitle)
    colors = [ORANGE if s == "C_comp" else BLUE for s in order]
    _dots(
        ax,
        [SET_LABELS[s] for s in order],
        rows.loc[order, "rmspe_pooled"],
        rows.loc[order, "ci_low"],
        rows.loc[order, "ci_high"],
        colors,
    )
    ax.set_xlabel("RMSPE (lower is better)", color=INK_2, fontsize=8.5)
    _legend(ax, [("Past information only", BLUE), ("Uses future time_ids", ORANGE)])
    fig.subplots_adjust(left=0.36)
    return _save(fig, config, "fig2_strict_vs_competition.png")


def figure_models(
    config, table: pd.DataFrame, holdout: pd.DataFrame | None
) -> str | None:
    """Figure 3: the model ladder with intervals."""
    p = primary_scheme(config)
    spec = [
        ("naive", "A"),
        ("har_ridge", "A"),
        ("forest", "D"),
        ("gbdt", "D"),
        ("mlp", "D"),
        ("blend", "D"),
    ]
    rows = []
    for model, features in spec:
        row = _rows(table, model=model, features=features, scheme=p)
        if len(row):
            rows.append(row.iloc[0])
    if len(rows) < 3:
        return None
    rows = pd.DataFrame(rows)
    best = rows.loc[rows["rmspe_pooled"].idxmin()]
    naive = rows.iloc[0]
    cut = 1 - best["rmspe_pooled"] / naive["rmspe_pooled"]
    title = (
        f"On development folds, {LABELS[best['model']]} is the best honest model at"
        f" {best['rmspe_pooled']:.3f} RMSPE, {cut:.0%} below naive persistence"
    )
    if best["model"] == "naive":
        title = (
            f"No model beats naive persistence ({naive['rmspe_pooled']:.3f} RMSPE)"
            " under honest validation"
        )
    subtitle = (
        f"Pooled out-of-fold RMSPE under {p} with 95% block-bootstrap intervals,"
        f" {int(best['n_rows']):,} development rows. LightGBM, forest, MLP and blend"
        " use feature set D (strict); HAR ridge uses its four log inputs. LightGBM and"
        " MLP average seeds 0-2."
    )
    if holdout is not None and len(holdout):
        keys = set(zip(rows["model"], rows["features"], strict=True))
        on_ladder = holdout[
            [
                (m, f) in keys
                for m, f in zip(holdout["model"], holdout["features"], strict=True)
            ]
        ]
        h_best = on_ladder.loc[on_ladder["rmspe"].idxmin()]
        h_naive = on_ladder[on_ladder["model"] == "naive"].iloc[0]
        h_cut = 1 - h_best["rmspe"] / h_naive["rmspe"]
        dev_rank = rows.set_index("model")["rmspe_pooled"].rank()
        hold_rank = on_ladder.set_index("model")["rmspe"].rank()
        same = int((dev_rank == hold_rank.reindex(dev_rank.index)).sum())
        agreement = (
            "the same ranking as on the development folds"
            if same == len(dev_rank)
            else f"{same} of {len(dev_rank)} models keep their development rank"
        )
        title = (
            f"On the holdout, {LABELS[h_best['model']]} reaches {h_best['rmspe']:.3f}"
            f" RMSPE, {h_cut:.0%} below naive persistence; {agreement}"
        )
        subtitle += (
            f" Holdout: {int(h_best['n_rows']):,} rows of the last 20% of time_ids,"
            " each model refitted once on the full development set."
        )
    labels = [f"{LABELS[r['model']]}" for _, r in rows.iterrows()]
    fig, ax = _figure(len(rows), title, subtitle)
    _dots(
        ax,
        labels,
        rows["rmspe_pooled"],
        rows["ci_low"],
        rows["ci_high"],
        [BLUE] * len(rows),
        fmt="{:.3f}",
    )
    if holdout is not None and len(holdout):
        y = np.arange(len(rows))[::-1]
        for yi, (_, r) in zip(y, rows.iterrows(), strict=True):
            h = _rows(holdout, model=r["model"], features=r["features"])
            if len(h):
                h = h.iloc[0]
                ax.plot(
                    [h["ci_low"], h["ci_high"]],
                    [yi - 0.28] * 2,
                    color=ORANGE,
                    linewidth=2,
                    solid_capstyle="round",
                )
                ax.plot(
                    h["rmspe"],
                    yi - 0.28,
                    "o",
                    color=ORANGE,
                    markersize=8,
                    markeredgecolor=SURFACE,
                    markeredgewidth=2,
                )
                ax.annotate(
                    f"{h['rmspe']:.3f}",
                    (h["ci_high"], yi - 0.28),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    color=INK_2,
                    fontsize=8,
                )
        _legend(ax, [("Development (walk-forward)", BLUE), ("Holdout", ORANGE)])
    ax.set_xlabel("RMSPE (lower is better)", color=INK_2, fontsize=8.5)
    fig.subplots_adjust(left=0.30)
    return _save(fig, config, "fig3_model_ladder.png")


def figure_ablation(config, ablation: pd.DataFrame) -> str | None:
    """Figure 4: ablation by feature group."""
    if ablation.empty:
        return None
    top = ablation.iloc[0]
    helpful = int((ablation["ci_low"] > 0).sum())
    title = (
        f"Removing the {top['group'].lower()} group costs the most:"
        f" {top['delta']:+.4f} RMSPE; {helpful} of {len(ablation)} groups have a"
        " clearly positive contribution"
    )
    subtitle = (
        "Change in pooled out-of-fold RMSPE when one group is removed from tuned"
        f" LightGBM on feature set D ({primary_scheme(config)}, seed"
        f" {int(top['seed'])}, {int(top['n_rows']):,} rows), with paired 95%"
        " block-bootstrap intervals. Positive means the group helps."
    )
    fig, ax = _figure(len(ablation), title, subtitle)
    colors = [BLUE if lo > 0 else MUTED for lo in ablation["ci_low"]]
    _dots(
        ax,
        list(ablation["group"]),
        ablation["delta"],
        ablation["ci_low"],
        ablation["ci_high"],
        colors,
        fmt="{:+.4f}",
    )
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel(
        "RMSPE without the group minus RMSPE with it", color=INK_2, fontsize=8.5
    )
    fig.subplots_adjust(left=0.24)
    return _save(fig, config, "fig4_ablation.png")


def figure_deciles(config, errors: dict[str, Any]) -> str | None:
    """Figure 5: error by market-volatility decile."""
    table = pd.DataFrame(errors["decile"])
    best = errors["best"]
    series = [m for m in ("naive", "har_ridge", best) if m in table.columns]
    if len(series) < 2:
        return None
    names = {
        "naive": "Naive",
        "har_ridge": "HAR ridge",
        "gbdt_D": "LightGBM (D)",
        "blend": "Blend",
    }
    low, high = table[best].iloc[0], table[best].iloc[-1]
    worst = table.loc[table[best].idxmax()]
    naive_top = table["naive"].iloc[-1]
    title = (
        f"Naive persistence breaks down in the most volatile decile"
        f" ({naive_top:.3f} RMSPE); {names.get(best, best)} stays at {high:.3f} and"
        f" errs most in the calmest decile ({low:.3f})"
        if naive_top > table["naive"].iloc[:-1].max() and low > high
        else f"{names.get(best, best)} RMSPE is {low:.3f} in the calmest decile of"
        f" market volatility and {high:.3f} in the most volatile; it peaks at"
        f" {worst[best]:.3f} in decile {int(worst['decile'])}"
    )
    subtitle = (
        "Out-of-fold RMSPE by decile of market RV (mean RV over stocks in the feature"
        f" window), {primary_scheme(config)}, {errors['n_rows']:,} development rows."
    )
    fig, ax = _figure(8, title, subtitle)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for name, color in zip(series, [MUTED, ORANGE, BLUE], strict=False):
        ax.plot(
            table["decile"],
            table[name],
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            solid_capstyle="round",
        )
        ax.annotate(
            names.get(name, name),
            (10, table[name].iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK_2,
            fontsize=8.5,
        )
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.6, 11.6)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Decile of market RV (1 = calmest)", color=INK_2, fontsize=8.5)
    ax.set_ylabel("RMSPE", color=INK_2, fontsize=8.5)
    _legend(
        ax,
        [
            (names.get(n, n), c)
            for n, c in zip(series, [MUTED, ORANGE, BLUE], strict=False)
        ],
    )
    fig.subplots_adjust(left=0.09)
    return _save(fig, config, "fig5_error_by_market_volatility.png")


# --------------------------------------------------------------------------
# Report blocks
# --------------------------------------------------------------------------
def md_table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
    """Render the selected columns of a frame as a Markdown table."""
    lines = ["| " + " | ".join(formats) + " |", "|" + " --- |" * len(formats)]
    for _, row in frame.iterrows():
        cells = []
        for column, fmt in formats.items():
            value = row[column]
            cells.append(fmt.format(value) if fmt else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def replace_blocks(path: Path, blocks: dict[str, str]) -> None:
    """Replace the text between BEGIN/END markers of a Markdown file."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for name, body in blocks.items():
        begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
        if begin in text and end in text:
            head, rest = text.split(begin, 1)
            _, tail = rest.split(end, 1)
            text = f"{head}{begin}\n{body}\n{end}{tail}"
    path.write_text(text, encoding="utf-8")


def status_blocks(config: dict[str, Any]) -> dict[str, str]:
    """Report blocks for the S3 order checks and the S5 sanity gates."""
    results = Path(config["paths"]["results"])
    blocks = {}
    status = order_status(config)
    lines = [
        f"Method: `{status['method']}`; status: **{status['status']}**.",
        "",
        "| Check | Value | Result |",
        "| --- | --- | --- |",
    ]
    for name, check in status["checks"].items():
        lines.append(
            f"| {name} | {check['value']:.4g} | {'ok' if check['ok'] else 'FAILED'} |"
        )
    agreement = status.get("agreement", {})
    if agreement:
        lines += [
            "",
            "| Method on real data | Rank displacement p99 against the method used |"
            " Best-successor share | Lag-1 autocorrelation of market log-RV |",
            "| --- | --- | --- | --- |",
        ]
        for name, stats in agreement.items():
            lines.append(
                f"| {name} | {stats['displacement_p99']:.0f} |"
                f" {stats['best_successor_share']:.3f} | {stats['rv_autocorr']:.3f} |"
            )
    blocks["order"] = "\n".join(lines)
    path = results / "gates_s5.json"
    if path.exists():
        g = json.loads(path.read_text(encoding="utf-8"))
        low, high = g["naive_range"]
        blocks["gates"] = "\n".join(
            [
                "| Gate (development rows) | Value | Required | Result |",
                "| --- | --- | --- | --- |",
                f"| Naive RMSPE, {g['n_rows']:,} rows |"
                f" {g['naive_rmspe_development_rows']:.4f} | {low} to {high} |"
                f" {'ok' if g['naive_ok'] else 'FAILED'} |",
                "| LightGBM, feature set A, group_time |"
                f" {g['sanity_rmspe_group_time']:.4f} | below {g['sanity_max']} |"
                f" {'ok' if g['sanity_ok'] else 'FAILED'} |",
            ]
        )
    return blocks


def verdict(low: float, high: float) -> str:
    """Name the sign of a difference whose 95% interval is [low, high]."""
    if high < 0:
        return "lower RMSPE (interval below zero)"
    if low > 0:
        return "higher RMSPE (interval above zero)"
    return "no clear difference (interval contains zero)"


def findings_block(diffs: pd.DataFrame, holdout_diffs: pd.DataFrame | None) -> str:
    """One line per research comparison: development and holdout differences."""
    lines = [
        "| Comparison (A minus B) | Development | Holdout |",
        "| --- | --- | --- |",
    ]
    hold = (
        holdout_diffs.set_index("comparison")
        if holdout_diffs is not None and len(holdout_diffs)
        else None
    )
    for _, row in diffs.iterrows():
        dev = (
            f"{row['difference']:+.4f} ({row['ci_low']:+.4f} to {row['ci_high']:+.4f}):"
            f" {verdict(row['ci_low'], row['ci_high'])}"
        )
        cell = "not evaluated"
        if hold is not None and row["comparison"] in hold.index:
            h = hold.loc[row["comparison"]]
            cell = (
                f"{h['difference']:+.4f} ({h['ci_low']:+.4f} to {h['ci_high']:+.4f}):"
                f" {verdict(h['ci_low'], h['ci_high'])}"
            )
        lines.append(f"| {row['comparison']} | {dev} | {cell} |")
    return "\n".join(lines)


def data_block(config: dict[str, Any]) -> str:
    """Dataset size and the split of time_ids into roles."""
    from lobvol.cv import load_roles

    roles = load_roles(config)["role"].value_counts()
    base = base_frame(config)
    lines = [
        f"- Stocks: {base['stock_id'].nunique()}; time_ids:"
        f" {int(roles.sum()):,} (development {int(roles.get('development', 0)):,},"
        f" embargo {int(roles.get('embargo', 0)):,}, holdout"
        f" {int(roles.get('holdout', 0)):,}).",
        f"- Development rows: {len(base):,}.",
    ]
    tuned = Path(config["paths"]["results"]) / "gbdt_tuned.json"
    if tuned.exists():
        t = json.loads(tuned.read_text(encoding="utf-8"))
        params = ", ".join(
            f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
            for k, v in t["params"].items()
        )
        lines.append(
            f"- LightGBM tuning: {t['n_trials']} Optuna TPE trials on feature set"
            f" {t['features']}, {t['scheme']}, seed {t['seed']}; best mean fold"
            f" RMSPE {t['rmspe_mean']:.4f} (trial {t['best_trial']}). Chosen: {params}."
        )
    return "\n".join(lines)


def load_holdout(config: dict[str, Any]) -> pd.DataFrame | None:
    """Return the holdout model table once S8 has run."""
    path = Path(config["paths"]["results"]) / "holdout" / "model_table.csv"
    return pd.read_csv(path) if path.exists() else None


def run_evaluate(config: dict[str, Any]) -> dict[str, Any]:
    """Write tables, intervals, figures and the generated report blocks."""
    results = Path(config["paths"]["results"])
    base = base_frame(config)
    table = model_table(config, base)
    table.to_csv(results / "model_table.csv", index=False)
    diffs = difference_table(config, base)
    diffs.to_csv(results / "differences.csv", index=False)
    ablation = ablation_table(config, base)
    ablation.to_csv(results / "ablation.csv", index=False)
    groups = shap_groups(config)
    if len(groups):
        groups.to_csv(results / "shap_groups.csv", index=False)
    errors = error_analysis(config, base)
    holdout = load_holdout(config)
    figures = {
        "fig1": figure_ladder(config, table, diffs),
        "fig2": figure_neighbours(config, table, diffs),
        "fig3": figure_models(config, table, holdout),
        "fig4": figure_ablation(config, ablation),
        "fig5": figure_deciles(config, errors),
    }

    show = table.assign(
        model=table["model"].map(LABELS),
        interval=[
            f"{a:.4f} to {b:.4f}"
            for a, b in zip(table["ci_low"], table["ci_high"], strict=True)
        ],
        tuned=np.where(table["tuned_on_this_metric"], "yes", "no"),
    )
    blocks = {
        "model_table": md_table(
            show,
            {
                "model": "",
                "features": "",
                "scheme": "",
                "rmspe_mean": "{:.4f}",
                "rmspe_sd_folds": "{:.4f}",
                "rmspe_pooled": "{:.4f}",
                "interval": "",
                "naive_rmspe_mean": "{:.4f}",
                "n_rows": "{:,}",
                "seeds": "",
                "train_seconds": "{:.0f}",
                "tuned": "",
            },
        ),
        "differences": md_table(
            diffs,
            {
                "comparison": "",
                "rmspe_a": "{:.4f}",
                "rmspe_b": "{:.4f}",
                "difference": "{:+.4f}",
                "ci_low": "{:+.4f}",
                "ci_high": "{:+.4f}",
                "n_rows": "{:,}",
            },
        ),
    }
    if len(ablation):
        blocks["ablation"] = md_table(
            ablation,
            {
                "group": "",
                "rmspe_without_group": "{:.4f}",
                "delta": "{:+.4f}",
                "ci_low": "{:+.4f}",
                "ci_high": "{:+.4f}",
            },
        )
    if len(groups):
        blocks["shap"] = md_table(
            groups, {"group": "", "n_features": "{}", "share": "{:.1%}"}
        )
    blocks["errors"] = md_table(
        pd.DataFrame(errors["decile"]),
        {
            "decile": "{:.0f}",
            "n_rows": "{:,.0f}",
            **{m: "{:.4f}" for m in errors["models"]},
        },
    )
    blocks.update(status_blocks(config))
    holdout_diffs = None
    path = results / "holdout" / "differences.csv"
    if path.exists():
        holdout_diffs = pd.read_csv(path)
    blocks["findings"] = findings_block(diffs, holdout_diffs)
    blocks["data"] = data_block(config)
    path = results / "block_holdout.md"
    if path.exists():
        blocks["holdout"] = path.read_text(encoding="utf-8").strip()
    reports = Path(config["paths"].get("reports", "reports"))
    replace_blocks(reports / "report.md", blocks)
    readme = Path(config.get("config_dir", "configs")).parent / "README.md"
    replace_blocks(readme, blocks)

    summary = {
        "order": order_status(config),
        "primary_scheme": primary_scheme(config),
        "figures": figures,
        "errors": {k: v for k, v in errors.items() if k != "decile"},
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    write_run_record(
        results / "run_records", "evaluate", config, config["frozen"]["seeds"]
    )
    logger.info(
        "evaluate: %d runs, %d differences, figures %s", len(table), len(diffs), figures
    )
    return summary


__all__ = ["run_evaluate", "squared_pct_error"]
