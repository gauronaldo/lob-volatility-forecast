"""Price recovery, 1-D ordering of time_ids and its validation (S3).

See METHOD.md section 5.1. A sequence (`seq`) is an array of row indices into
the sorted list of time_ids, in estimated temporal order.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lobvol.data import BOOK_DIR, list_stock_ids, read_book, read_targets
from lobvol.runlog import write_run_record

logger = logging.getLogger(__name__)

FLOAT32_NOISE = 5e-7  # gaps below this are float noise, not price levels
PRICE_COLUMNS = ["bid_price1", "ask_price1", "bid_price2", "ask_price2"]
METHODS = ["isomap", "greedy_2opt", "isomap_2opt", "end_start_chain"]


# --------------------------------------------------------------------------
# Step 1: price level
# --------------------------------------------------------------------------
def stock_prices(book: pd.DataFrame, tick: float) -> pd.DataFrame:
    """Recover P0 and the log real price at the start and end of each bucket."""
    book = book.sort_values(["time_id", "seconds_in_bucket"], kind="stable")
    time_ids, bucket = np.unique(book["time_id"].to_numpy(), return_inverse=True)
    n = len(time_ids)
    prices = np.ascontiguousarray(book[PRICE_COLUMNS].to_numpy(dtype=np.float32))

    # Distinct price levels per bucket, sorted by (bucket, price).
    bits = prices.view(np.uint32).astype(np.int64)
    key = np.unique((bucket[:, None].astype(np.int64) << 32) | bits)
    b = key >> 32
    p = (key & 0xFFFFFFFF).astype(np.uint32).view(np.float32).astype(np.float64)
    same = b[1:] == b[:-1]
    gap = np.where(same, np.diff(p), np.inf)
    gap[gap <= FLOAT32_NOISE] = np.inf
    g0 = np.full(n, np.inf)
    np.minimum.at(g0, b[1:], gap)
    starts = np.r_[0, np.flatnonzero(~same) + 1]
    d = p - p[starts][b]

    def fit_tick(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Refine a tick estimate by least squares over growing level ranges."""
        with np.errstate(divide="ignore", invalid="ignore"):
            for max_k in (3, 8, 20, 60):
                ratio = d / g[b]
                k = np.round(ratio)
                use = (k >= 1) & (k <= max_k) & (np.abs(ratio - k) < 0.3)
                num = np.bincount(b, weights=np.where(use, d * k, 0.0), minlength=n)
                den = np.bincount(b, weights=np.where(use, k * k, 0.0), minlength=n)
                g = np.where(den > 0, num / np.maximum(den, 1e-300), g)
            ratio = d / g[b]
            k = np.round(ratio)
            seen = (k >= 1) & (k <= 60)
            good = seen & (np.abs(ratio - k) < 0.15)
        total = np.bincount(b, weights=seen, minlength=n)
        share = np.bincount(b, weights=good, minlength=n) / np.maximum(total, 1)
        return g, np.where(total >= 3, share, 0.0)

    # The smallest gap may span several ticks: accept the largest tick on
    # which (nearly) every level is an integer multiple.
    g = np.full(n, np.inf)
    settled = np.zeros(n, dtype=bool)
    for m in (1, 2, 3, 4):
        candidate, share = fit_tick(g0 / m)
        ok = ~settled & (share >= 0.98)
        g[ok] = candidate[ok]
        settled |= ok

    p0 = np.where(np.isfinite(g) & settled, tick / g, np.nan)
    mid = 0.5 * (book["bid_price1"].to_numpy() + book["ask_price1"].to_numpy())
    first = np.r_[0, np.flatnonzero(np.diff(bucket)) + 1]
    last = np.r_[first[1:] - 1, len(bucket) - 1]
    with np.errstate(invalid="ignore"):
        frame = pd.DataFrame(
            {
                "time_id": time_ids,
                "p0": p0,
                "log_start": np.log(p0 * mid[first]),
                "log_end": np.log(p0 * mid[last]),
            }
        )
    return frame


def flag_gross_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    """Set to NaN the buckets whose P0 is far from every plausible stock price."""
    ratio = frame["p0"] / frame["p0"].median()
    off = (ratio < 0.2) | (ratio > 5.0)
    frame = frame.copy()
    frame.loc[off, ["p0", "log_start", "log_end"]] = np.nan
    return frame


def flag_local_outliers(
    table: pd.DataFrame, position: pd.Series, window: int, max_dev: float
) -> pd.DataFrame:
    """Set to NaN the prices far from the stock's neighbours in the recovered order.

    A P0 that is off by an integer factor (only multi-tick gaps were observed)
    cannot be told from a real price move before the order is known; along
    the order it stands out against the rolling median.
    """
    table = table.copy()
    table["_pos"] = table["time_id"].map(position)
    table = table.sort_values(["stock_id", "_pos"], ignore_index=True)
    local = table.groupby("stock_id")["log_start"].transform(
        lambda x: x.rolling(window, center=True, min_periods=5).median()
    )
    off = (table["log_start"] - local).abs() > np.log1p(max_dev)
    table.loc[off, ["p0", "log_start", "log_end"]] = np.nan
    return table.drop(columns="_pos")


def build_price_table(raw: Path, cache: Path, tick: float, force: bool = False):
    """Return (stock_id, time_id, p0, log_start, log_end) for every bucket."""
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for stock_id in list_stock_ids(raw / BOOK_DIR):
        part = cache / f"stock_id={stock_id}.parquet"
        if not part.exists() or force:
            frame = stock_prices(read_book(raw, stock_id), tick)
            frame = flag_gross_outliers(frame)
            frame.insert(0, "stock_id", stock_id)
            frame.to_parquet(part, index=False)
        parts.append(pd.read_parquet(part))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Step 2: price matrices and distances
# --------------------------------------------------------------------------
def price_matrices(table: pd.DataFrame, max_nan_share: float) -> dict[str, Any]:
    """Build the standardised start and end price matrices (time_id x stock).

    Missing prices stay NaN: imputing a trending price with its column median
    would create jumps far larger than the distance between neighbours.
    """
    start = table.pivot(index="time_id", columns="stock_id", values="log_start")
    end = table.pivot(index="time_id", columns="stock_id", values="log_end")
    keep = start.columns[start.isna().mean() <= max_nan_share]
    dropped = [int(c) for c in start.columns.difference(keep)]
    start, end = start[keep], end[keep].reindex(start.index)
    mean, std = start.mean(), start.std().replace(0.0, 1.0)
    return {
        "time_ids": start.index.to_numpy(),
        "start": ((start - mean) / std).to_numpy(),
        "end": ((end - mean) / std).to_numpy(),
        "log_start": start.to_numpy(),
        "stocks": [int(c) for c in keep],
        "dropped": dropped,
    }


def _trimmed_rms(diff: np.ndarray, trim: float) -> np.ndarray:
    """Row-wise RMS of differences, ignoring NaN and the largest share of them."""
    sq = diff**2
    n_cols = sq.shape[1]
    n_keep = max(1, n_cols - int(np.ceil(trim * n_cols)))
    sq = np.where(np.isnan(sq), np.inf, sq)
    kept = np.partition(sq, n_keep - 1, axis=1)[:, :n_keep]
    finite = np.isfinite(kept)
    count = finite.sum(axis=1)
    total = np.where(finite, kept, 0.0).sum(axis=1)
    return np.sqrt(np.where(count > 0, total / np.maximum(count, 1), np.inf))


def robust_distance(x: np.ndarray, trim: float) -> np.ndarray:
    """Return the trimmed RMS distance between every pair of rows of x."""
    dist = np.empty((len(x), len(x)))
    for i in range(len(x)):
        dist[i] = _trimmed_rms(x - x[i], trim)
    finite = np.isfinite(dist)
    dist[~finite] = dist[finite].max() * 10.0 if finite.any() else 1.0
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


# --------------------------------------------------------------------------
# Step 3: ordering
# --------------------------------------------------------------------------
def order_isomap(dist: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Order the rows by a 1-D Isomap embedding on a k-NN graph of distances."""
    from sklearn.manifold import Isomap

    k = min(n_neighbors, len(dist) - 1)
    model = Isomap(n_neighbors=k, n_components=1, metric="precomputed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = model.fit_transform(dist)[:, 0]
    return np.argsort(embedding, kind="stable")


def order_greedy(dist: np.ndarray) -> np.ndarray:
    """Build a path by repeatedly moving to the nearest unvisited row."""
    n = len(dist)
    visited = np.zeros(n, dtype=bool)
    seq = np.empty(n, dtype=np.int64)
    current = 0
    for step in range(n):
        seq[step] = current
        visited[current] = True
        if step < n - 1:
            current = int(np.argmin(np.where(visited, np.inf, dist[current])))
    return seq


def _two_opt_sweep(seq: np.ndarray, dist: np.ndarray) -> bool:
    """Reverse segments of an open path while that shortens it; one sweep."""
    n = len(seq)
    improved = False
    for i in range(n - 1):
        j = np.arange(i + 1, n)
        pi, pj = seq[i], seq[j]
        delta = np.zeros(len(j))
        if i > 0:
            a = seq[i - 1]
            delta += dist[a, pj] - dist[a, pi]
        nxt = seq[j[:-1] + 1]
        delta[:-1] += dist[pi, nxt] - dist[pj[:-1], nxt]
        best = int(np.argmin(delta))
        if delta[best] < -1e-12:
            seq[i : j[best] + 1] = seq[i : j[best] + 1][::-1]
            improved = True
    return improved


def _or_opt_sweep(seq: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, bool]:
    """Move single rows to a better place in an open path; one sweep."""
    n = len(seq)
    improved = False
    for idx in range(n):
        node = seq[idx]
        rest = np.delete(seq, idx)
        if idx == 0:
            gain = dist[node, seq[1]]
        elif idx == n - 1:
            gain = dist[seq[n - 2], node]
        else:
            before, after = seq[idx - 1], seq[idx + 1]
            gain = dist[before, node] + dist[node, after] - dist[before, after]
        cost = np.empty(n)
        cost[0] = dist[node, rest[0]]
        cost[-1] = dist[rest[-1], node]
        cost[1:-1] = (
            dist[rest[:-1], node] + dist[node, rest[1:]] - dist[rest[:-1], rest[1:]]
        )
        best = int(np.argmin(cost))
        if best != idx and cost[best] < gain - 1e-12:
            seq = np.insert(rest, best, node)
            improved = True
    return seq, improved


def refine_path(seq: np.ndarray, dist: np.ndarray, max_sweeps: int) -> np.ndarray:
    """Shorten an open path with alternating 2-opt and single-row moves."""
    seq = seq.copy()
    for _ in range(max_sweeps):
        improved = _two_opt_sweep(seq, dist)
        seq, moved = _or_opt_sweep(seq, dist)
        if not (improved or moved):
            break
    return seq


def end_start_cost(start: np.ndarray, end: np.ndarray, trim: float) -> np.ndarray:
    """Return cost[i, j] = distance from the end of row i to the start of row j."""
    cost = np.empty((len(start), len(start)))
    for i in range(len(start)):
        cost[i] = _trimmed_rms(start - end[i], trim)
    finite = np.isfinite(cost)
    cost[~finite] = cost[finite].max() * 10.0 if finite.any() else 1.0
    np.fill_diagonal(cost, np.inf)
    return cost


def _greedy_edges(cost: np.ndarray, n_candidates: int) -> np.ndarray:
    """Link each row to a successor, cheapest edges first, without cycles."""
    n = len(cost)
    k = min(n_candidates, n - 1)
    near = np.argpartition(cost, k - 1, axis=1)[:, :k]
    src = np.repeat(np.arange(n), k)
    dst = near.ravel()
    by_cost = np.argsort(cost[src, dst], kind="stable")
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    successor = np.full(n, -1, dtype=np.int64)
    has_pred = np.zeros(n, dtype=bool)
    for e in by_cost:
        i, j = int(src[e]), int(dst[e])
        if successor[i] >= 0 or has_pred[j]:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        successor[i], has_pred[j], parent[ri] = j, True, rj
    return successor


def _join_fragments(successor: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Join path fragments tail-to-head, cheapest joins first, into one sequence."""
    has_pred = np.zeros(len(successor), dtype=bool)
    has_pred[successor[successor >= 0]] = True
    fragments = []
    for head in np.flatnonzero(~has_pred):
        chain = [int(head)]
        while successor[chain[-1]] >= 0:
            chain.append(int(successor[chain[-1]]))
        fragments.append(chain)
    while len(fragments) > 1:
        tails = np.array([f[-1] for f in fragments])
        heads = np.array([f[0] for f in fragments])
        join = cost[np.ix_(tails, heads)].copy()
        np.fill_diagonal(join, np.inf)
        a, b = np.unravel_index(int(np.argmin(join)), join.shape)
        merged = fragments[a] + fragments[b]
        fragments = [f for i, f in enumerate(fragments) if i not in (a, b)] + [merged]
    return np.array(fragments[0], dtype=np.int64)


def _segment_move_sweep(seq: np.ndarray, cost: np.ndarray, length: int):
    """Move segments of a directed path to a cheaper place, keeping their direction."""
    n = len(seq)
    improved = False
    i = 0
    while i + length <= n:
        seg = seq[i : i + length]
        rest = np.concatenate([seq[:i], seq[i + length :]])
        first, last = seg[0], seg[-1]
        if len(rest) < 2:
            break
        gain = 0.0
        if i > 0:
            gain += cost[seq[i - 1], first]
        if i + length < n:
            gain += cost[last, seq[i + length]]
        if 0 < i and i + length < n:
            gain -= cost[seq[i - 1], seq[i + length]]
        insert = np.empty(len(rest) + 1)
        insert[0] = cost[last, rest[0]]
        insert[-1] = cost[rest[-1], first]
        insert[1:-1] = (
            cost[rest[:-1], first] + cost[last, rest[1:]] - cost[rest[:-1], rest[1:]]
        )
        best = int(np.argmin(insert))
        if best != i and insert[best] < gain - 1e-12:
            seq = np.concatenate([rest[:best], seg, rest[best:]])
            improved = True
        i += 1
    return seq, improved


def order_chain(cost: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Chain time_ids so that each one starts where the previous one ended.

    The cost is asymmetric (end of i to start of j), so the path is built in
    the forward direction and segments are never reversed.
    """
    seq = _join_fragments(_greedy_edges(cost, cfg["chain_candidates"]), cost)
    for _ in range(cfg["refine_sweeps"]):
        moved_any = False
        for length in range(1, cfg["chain_max_segment"] + 1):
            seq, moved = _segment_move_sweep(seq, cost, length)
            moved_any |= moved
        if not moved_any:
            break
    return seq


def order_rows(
    dist: np.ndarray, method: str, cfg: dict[str, Any], cost: np.ndarray | None = None
) -> np.ndarray:
    """Order the rows with one method; `end_start_chain` needs the directed cost."""
    if method == "end_start_chain":
        return order_chain(cost, cfg)
    if method == "isomap":
        return order_isomap(dist, cfg["isomap_neighbors"])
    if method == "greedy_2opt":
        return refine_path(order_greedy(dist), dist, cfg["refine_sweeps"])
    if method == "isomap_2opt":
        seq = order_isomap(dist, cfg["isomap_neighbors"])
        return refine_path(seq, dist, cfg["refine_sweeps"])
    raise ValueError(f"unknown ordering method: {method}")


def positions(seq: np.ndarray) -> np.ndarray:
    """Return the position of every row given a sequence of rows."""
    pos = np.empty(len(seq), dtype=np.int64)
    pos[seq] = np.arange(len(seq))
    return pos


# --------------------------------------------------------------------------
# Step 4: direction
# --------------------------------------------------------------------------
def price_direction(seq: np.ndarray, start: np.ndarray, end: np.ndarray, trim: float):
    """Return the mean end(i)->start(i+1) and end(i+1)->start(i) distances."""
    a, b = seq[:-1], seq[1:]
    forward = _trimmed_rms(end[a] - start[b], trim)
    backward = _trimmed_rms(end[b] - start[a], trim)
    ok = np.isfinite(forward) & np.isfinite(backward)
    return float(forward[ok].mean()), float(backward[ok].mean())


def orient(seq: np.ndarray, mats: dict[str, Any], trim: float):
    """Put a sequence in the forward direction; return it with both distances."""
    forward, backward = price_direction(seq, mats["start"], mats["end"], trim)
    if forward > backward:
        return seq[::-1], backward, forward
    return seq, forward, backward


def recover_order(table: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Two-pass order recovery: order, drop local price outliers, order again."""
    trim = cfg["trim"]
    mats = price_matrices(table, cfg["max_nan_share"])
    cost = end_start_cost(mats["start"], mats["end"], trim)
    seq, _, _ = orient(order_chain(cost, cfg), mats, trim)
    first_pass = pd.Series(positions(seq), index=mats["time_ids"])
    clean = flag_local_outliers(
        table, first_pass, cfg["outlier_window"], cfg["outlier_max_dev"]
    )
    mats = price_matrices(clean, cfg["max_nan_share"])
    dist = robust_distance(mats["start"], trim)
    cost = end_start_cost(mats["start"], mats["end"], trim)
    sequences = {}
    for method in METHODS:
        sequences[method] = orient(order_rows(dist, method, cfg, cost), mats, trim)
    return {
        "table": clean,
        "mats": mats,
        "dist": dist,
        "cost": cost,
        "sequences": sequences,
    }


def path_quality(seq: np.ndarray, cost: np.ndarray) -> dict[str, float]:
    """Describe a sequence by its end-to-start edges."""
    edges = cost[seq[:-1], seq[1:]]
    best = np.argmin(cost, axis=1)
    return {
        "edge_median": float(np.median(edges)),
        "edge_mean": float(edges.mean()),
        "best_successor_share": float((best[seq[:-1]] == seq[1:]).mean()),
    }


def target_direction(position: pd.Series, rv: pd.DataFrame, targets: pd.DataFrame):
    """Return corr(log target_i, log RV_{i+1}) and corr(log RV_i, log target_{i+1}).

    `position` maps time_id to its position; only the time_ids present in
    `targets` are used. Both series are demeaned per stock first.
    """
    frame = rv.merge(targets, on=["stock_id", "time_id"])
    frame = frame[(frame["rv"] > 0) & (frame["target"] > 0)].copy()
    frame["pos"] = frame["time_id"].map(position)
    frame["log_rv"] = np.log(frame["rv"])
    frame["log_y"] = np.log(frame["target"])
    for c in ("log_rv", "log_y"):
        frame[c] -= frame.groupby("stock_id")[c].transform("mean")
    nxt = frame[["stock_id", "pos", "log_rv", "log_y"]].copy()
    nxt["pos"] -= 1
    pair = frame.merge(nxt, on=["stock_id", "pos"], suffixes=("", "_next"))
    forward = np.corrcoef(pair["log_y"], pair["log_rv_next"])[0, 1]
    backward = np.corrcoef(pair["log_rv"], pair["log_y_next"])[0, 1]
    return float(forward), float(backward)


# --------------------------------------------------------------------------
# Step 5: validation
# --------------------------------------------------------------------------
def order_accuracy(seq: np.ndarray, true_position: np.ndarray) -> dict[str, float]:
    """Compare an estimated sequence with the true positions of the rows."""
    est = positions(seq)
    rank_true = pd.Series(true_position).rank().to_numpy()
    displacement = np.abs(est - true_position)
    by_time = np.argsort(true_position)
    inverted = est[by_time][1:] < est[by_time][:-1]
    return {
        "spearman": float(np.corrcoef(est, rank_true)[0, 1]),
        "displacement_median": float(np.median(displacement)),
        "displacement_p99": float(np.percentile(displacement, 99)),
        "adjacent_inverted": float(inverted.mean()),
    }


def jump_statistic(seq: np.ndarray, log_start: np.ndarray) -> float:
    """Median over consecutive rows of the cross-sectional median |log-price jump|."""
    diff = np.abs(log_start[seq[1:]] - log_start[seq[:-1]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.nanmedian(np.nanmedian(diff, axis=1)))


def lag1_autocorr(values: np.ndarray) -> float:
    """Return the lag-1 autocorrelation of a series."""
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def synthetic_check(
    config: dict[str, Any], seed: int | None = None
) -> dict[str, dict[str, float]]:
    """Recover the order of a fresh synthetic dataset with every method.

    The data is written to a new temporary directory each time, so files of
    an earlier synthetic dataset can never mix with it.
    """
    import tempfile

    from lobvol.synth import generate

    cfg = config["order"]
    params = dict(cfg["synth"])
    if seed is not None:
        params["seed"] = seed
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        work = Path(folder)
        synth = generate(work / "raw", **params)
        table = build_price_table(
            synth.raw_dir, work / "prices", config["frozen"]["tick_size"], force=True
        )
        recovered = recover_order(table, cfg)
    truth = synth.true_order.set_index("time_id")["position"]
    true_position = truth.reindex(recovered["mats"]["time_ids"]).to_numpy()
    if np.isnan(true_position).any():
        raise RuntimeError("synthetic check: recovered time_ids do not match the truth")
    results = {}
    for method, (seq, _, _) in recovered["sequences"].items():
        stats = order_accuracy(seq, true_position.astype(np.int64))
        stats["direction_correct"] = float(stats["spearman"] > 0)
        results[method] = stats
    return results


def run_order(config: dict[str, Any], force: bool = False) -> bool:
    """Recover the time order, validate it and return whether every check passed."""
    paths = config["paths"]
    interim, results = Path(paths["interim"]), Path(paths["results"])
    out = interim / "time_order.parquet"
    status_path = interim / "order_status.json"
    if out.exists() and status_path.exists() and not force:
        logger.info("%s exists; skipping (use --force to redo)", out)
        return json.loads(status_path.read_text())["status"] == "ok"
    cfg, frozen, gates = config["order"], config["frozen"], config["gates"]
    results.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(frozen["seeds"][0])
    n_perm = gates["order_permutations"]

    # Synthetic data with a known order; also selects the method.
    synth = synthetic_check(config)
    method = cfg["method"]
    if method == "auto":
        method = min(
            METHODS,
            key=lambda m: (
                np.nan_to_num(synth[m]["displacement_p99"], nan=np.inf),
                np.nan_to_num(synth[m]["adjacent_inverted"], nan=np.inf),
            ),
        )
    chosen = synth[method]
    base_seed = cfg["synth"]["seed"]
    other_seeds = {
        s: synthetic_check(config, seed=s)[method]
        for s in (base_seed + 1, base_seed + 2)
    }
    checks = {
        "synthetic Spearman rho": (
            chosen["spearman"],
            f">= {gates['order_spearman_min']}",
            chosen["spearman"] >= gates["order_spearman_min"],
        ),
        "synthetic 99th percentile displacement": (
            chosen["displacement_p99"],
            f"<= {gates['order_displacement_p99_max']}",
            chosen["displacement_p99"] <= gates["order_displacement_p99_max"],
        ),
    }

    # Real data.
    raw_table = build_price_table(
        Path(paths["raw"]), interim / "prices", frozen["tick_size"], force
    )
    recovered = recover_order(raw_table, cfg)
    table, mats = recovered["table"], recovered["mats"]
    table.to_parquet(interim / "price_table.parquet", index=False)
    np.save(interim / "dist_price.npy", recovered["dist"].astype(np.float32))
    time_ids = mats["time_ids"]
    seq, forward, backward = recovered["sequences"][method]
    position = pd.Series(positions(seq), index=time_ids)

    # Direction confirmation from targets, on the first 80% only (rule 2).
    n_dev = int(round(len(seq) * (1 - frozen["holdout_fraction"])))
    early = set(time_ids[seq[:n_dev]])
    bucket = pd.read_parquet(
        Path(paths["features"]) / "bucket.parquet",
        columns=["stock_id", "time_id", "rv1_600"],
    ).rename(columns={"rv1_600": "rv"})
    targets = read_targets(paths["raw"], early)
    t_forward, t_backward = target_direction(position, rv=bucket, targets=targets)
    del targets
    checks["direction cues agree"] = (
        t_forward - t_backward,
        "> 0 (targets), with the price cue already forward",
        bool(t_forward > t_backward),
    )

    # Jump ratio against random orders.
    observed_jump = jump_statistic(seq, mats["log_start"])
    random_jumps = np.array(
        [
            jump_statistic(rng.permutation(len(seq)), mats["log_start"])
            for _ in range(n_perm)
        ]
    )
    ratio = observed_jump / random_jumps.mean()
    p_value = (1 + np.sum(random_jumps <= observed_jump)) / (n_perm + 1)
    checks["jump ratio (consecutive / random pairs)"] = (
        ratio,
        f"<= {gates['order_jump_ratio_max']}",
        ratio <= gates["order_jump_ratio_max"],
    )
    checks["jump permutation p-value"] = (
        p_value,
        f"< {gates['order_p_value_max']}",
        p_value < gates["order_p_value_max"],
    )

    # Lag-1 autocorrelation of market log-RV against shuffled orders.
    positive = bucket[bucket["rv"] > 0]
    market = np.log(positive["rv"]).groupby(positive["time_id"]).mean()
    market = market.reindex(time_ids).to_numpy()[seq]
    observed_ac = lag1_autocorr(market)
    shuffled = np.array([lag1_autocorr(rng.permutation(market)) for _ in range(n_perm)])
    threshold = float(np.percentile(shuffled, gates["order_autocorr_percentile"]))
    checks["lag-1 autocorrelation of market log-RV"] = (
        observed_ac,
        f"> {threshold:.4f} (99th percentile of shuffled orders)",
        observed_ac > threshold,
    )

    # Agreement between methods on real data (reported, not gated).
    market_by_row = market[positions(seq)]
    agreement = {}
    for other, (other_seq, _, _) in recovered["sequences"].items():
        stats = order_accuracy(other_seq, position.to_numpy())
        stats.update(path_quality(other_seq, recovered["cost"]))
        stats["rv_autocorr"] = lag1_autocorr(market_by_row[other_seq])
        agreement[other] = stats

    passed = all(ok for _, _, ok in checks.values())
    status = {
        "status": "ok" if passed else "fallback",
        "method": method,
        "checks": {
            k: {"value": float(v), "ok": bool(ok)} for k, (v, _, ok) in checks.items()
        },
        "agreement": agreement,
    }
    order = pd.DataFrame({"time_id": time_ids, "position": position.to_numpy()})
    order.sort_values("position").to_parquet(out, index=False)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    n_raw_nan = int(raw_table["p0"].isna().sum())
    n_nan = int(table["p0"].isna().sum())
    lines = [
        "# Time-order validation (S3)",
        "",
        "Generated by `python -m lobvol order`.",
        "",
        f"**Gate (every check passes): {'PASS' if passed else 'FAIL -> fallback'}**",
        "",
        f"Method used: `{method}` (config `order.method: {cfg['method']}`).",
        "",
        "| Check | Value | Required | Result |",
        "| --- | --- | --- | --- |",
        *[
            f"| {k} | {v:.4g} | {req} | {'ok' if ok else 'FAILED'} |"
            for k, (v, req, ok) in checks.items()
        ],
        "",
        "## Synthetic data with a known order",
        "",
        f"{cfg['synth']['n_stocks']} stocks, {cfg['synth']['n_time_ids']} time_ids,"
        f" seed {cfg['synth']['seed']}.",
        "",
        "| Method | Spearman rho | Displacement median | Displacement p99 |"
        " Adjacent pairs inverted | Direction correct |",
        "| --- | --- | --- | --- | --- | --- |",
        *[
            f"| {m} | {s['spearman']:.5f} | {s['displacement_median']:.1f} |"
            f" {s['displacement_p99']:.1f} | {s['adjacent_inverted']:.3f} |"
            f" {'yes' if s['direction_correct'] else 'NO'} |"
            for m, s in synth.items()
        ],
        "",
        f"The gate uses seed {base_seed}. The method used on two more seeds"
        " (reported, not gated): "
        + "; ".join(
            f"seed {k}: rho {v['spearman']:.5f}, displacement p99"
            f" {v['displacement_p99']:.1f}, adjacent pairs inverted"
            f" {v['adjacent_inverted']:.3f}"
            for k, v in other_seeds.items()
        )
        + ".",
        "",
        "## Real data",
        "",
        f"- time_ids ordered: {len(seq)}; stocks used: {len(mats['stocks'])};"
        f" stocks dropped for more than {cfg['max_nan_share']:.0%} missing prices:"
        f" {mats['dropped'] or 'none'}",
        f"- Buckets with no usable price level: {n_raw_nan:,} after tick recovery,"
        f" {n_nan:,} after the local outlier pass, of {len(table):,}. They stay"
        " NaN and are ignored in distances.",
        f"- Direction from prices: mean distance end(i)->start(i+1) = {forward:.4f},"
        f" end(i+1)->start(i) = {backward:.4f}",
        f"- Direction from targets (first {n_dev} time_ids only):"
        f" corr(log target_i, log RV_i+1) = {t_forward:.4f},"
        f" corr(log RV_i, log target_i+1) = {t_backward:.4f}",
        f"- Median cross-sectional absolute log-price jump: consecutive"
        f" {observed_jump:.5f}, random pairs {random_jumps.mean():.5f}, ratio"
        f" {ratio:.4f}, permutation p-value {p_value:.4f}",
        f"- Lag-1 autocorrelation of market log-RV: {observed_ac:.4f}; shuffled"
        f" orders: mean {shuffled.mean():.4f}, 99th percentile {threshold:.4f}",
        "",
        "## Methods on real data (reported, not gated; there is no ground truth)",
        "",
        "Rank displacement is measured against the method used. An edge is the",
        "end-to-start price distance between consecutive time_ids; the best-",
        "successor share is the share of edges that link a time_id to the one",
        "whose start is closest to its end. The RV autocorrelation is an",
        "independent cue: prices, not RV, are used for ordering.",
        "",
        "| Method | Spearman rho | Displacement median | Displacement p99 |"
        " Edge median | Best-successor share | Lag-1 autocorr of market log-RV |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *[
            f"| {m} | {s['spearman']:.5f} | {s['displacement_median']:.1f} |"
            f" {s['displacement_p99']:.1f} | {s['edge_median']:.5f} |"
            f" {s['best_successor_share']:.3f} | {s['rv_autocorr']:.4f} |"
            for m, s in agreement.items()
        ],
        "",
    ]
    (results / "order_validation.md").write_text("\n".join(lines), encoding="utf-8")
    write_run_record(results / "run_records", "order", config, [frozen["seeds"][0]])
    if not passed:
        logger.warning("order checks failed; the fallback of section 8.1 applies")
    logger.info("wrote %s (method %s, status %s)", out, method, status["status"])
    return passed
