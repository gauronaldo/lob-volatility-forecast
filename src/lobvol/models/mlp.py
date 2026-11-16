"""PyTorch MLP with a stock embedding, trained on log q with a percentage loss.

With f the network output and q = y / x, the forecast is yhat = x * exp(f)
and its percentage error is 1 - exp(f - log q). The loss is the mean of that
squared, which is RMSPE^2 of y.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from lobvol.cv import inner_split
from lobvol.models import denominator, rv_floor


class Net(nn.Module):
    """Feed-forward network over numeric inputs and a stock embedding."""

    def __init__(self, n_inputs: int, n_stocks: int, params: dict[str, Any]) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_stocks + 1, params["embedding_dim"])
        layers: list[nn.Module] = []
        width = n_inputs + params["embedding_dim"]
        for hidden in params["hidden"]:
            layers += [
                nn.Linear(width, hidden),
                nn.SiLU(),
                nn.Dropout(params["dropout"]),
            ]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, stock: torch.Tensor) -> torch.Tensor:
        """Return the predicted log ratio."""
        return self.body(torch.cat([x, self.embedding(stock)], dim=1)).squeeze(1)


def _loss(f: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Mean squared percentage error of y."""
    return ((1.0 - torch.exp(torch.clamp(f - log_q, max=4.0))) ** 2).mean()


def _prepare(train: pd.DataFrame, features: list[str], seed: int) -> dict[str, Any]:
    """Fit the quantile transform, the stock coding and the RV floor on train."""
    numeric = [f for f in features if f != "stock_id"]
    transform = QuantileTransformer(
        n_quantiles=1000,
        output_distribution="normal",
        subsample=200_000,
        random_state=seed,
    ).fit(train[numeric])
    stocks = np.sort(train["stock_id"].unique())
    return {
        "numeric": numeric,
        "transform": transform,
        "stock_code": pd.Series(np.arange(1, len(stocks) + 1), index=stocks),
        "floor": rv_floor(train["rv1_600"]),
    }


def _tensors(frame: pd.DataFrame, state: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    """Transform a frame into inputs, stock codes and, if present, log q."""
    x = np.nan_to_num(state["transform"].transform(frame[state["numeric"]]))
    code = frame["stock_id"].map(state["stock_code"]).fillna(0).to_numpy()
    out = [torch.tensor(x, dtype=torch.float32), torch.tensor(code, dtype=torch.long)]
    if "target" in frame:
        q = frame["target"].to_numpy() / denominator(frame["rv1_600"], state["floor"])
        out.append(torch.tensor(np.log(q), dtype=torch.float32))
    return tuple(out)


def _train(
    data: tuple[torch.Tensor, ...],
    n_stocks: int,
    params: dict[str, Any],
    seed: int,
    epochs: int,
    watch: tuple[torch.Tensor, ...] | None = None,
) -> tuple[Net, int]:
    """Train for `epochs`, or until the watched loss stops improving."""
    torch.manual_seed(seed)
    x, stock, log_q = data
    net = Net(x.shape[1], n_stocks, params)
    opt = torch.optim.AdamW(
        net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, bad = np.inf, epochs, None, 0
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), params["batch_size"]):
            batch = perm[start : start + params["batch_size"]]
            opt.zero_grad()
            _loss(net(x[batch], stock[batch]), log_q[batch]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if watch is None:
            continue
        net.eval()
        with torch.no_grad():
            score = float(_loss(net(watch[0], watch[1]), watch[2]))
        if score < best - 1e-6:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= params["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_epoch


def fit_predict(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    seed: int,
    inner_fraction: float,
    threads: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Early-stop on the inner split, refit on all training rows, predict y."""
    if threads:
        torch.set_num_threads(threads)
    inner = inner_split(train["position"].to_numpy(), inner_fraction)
    state = _prepare(train[~inner], features, seed)
    n_stocks = len(state["stock_code"])
    _, epochs = _train(
        _tensors(train[~inner], state),
        n_stocks,
        params,
        seed,
        params["max_epochs"],
        watch=_tensors(train[inner], state),
    )
    state = _prepare(train, features, seed)
    n_stocks = len(state["stock_code"])
    net, _ = _train(_tensors(train, state), n_stocks, params, seed, epochs)
    net.eval()
    x, stock = _tensors(val.drop(columns="target", errors="ignore"), state)
    with torch.no_grad():
        f = net(x, stock).numpy().astype(np.float64)
    pred = np.exp(f) * denominator(val["rv1_600"], state["floor"])
    return pred, {"epochs": epochs}
