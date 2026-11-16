"""RMSPE against hand-computed values (METHOD.md section 7, test 2)."""

import numpy as np
import pytest

from lobvol.evaluation import rmspe


def test_rmspe_matches_hand_computed_value() -> None:
    # Percentage errors: (1 - 1.1) / 1 = -0.1 and (2 - 1.8) / 2 = 0.1.
    assert rmspe([1.0, 2.0], [1.1, 1.8]) == pytest.approx(0.1, abs=1e-12)


def test_weighted_mse_on_ratio_target_equals_squared_rmspe() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.001, 0.01, 200)  # current RV
    q = rng.uniform(0.5, 2.0, 200)  # ratio target
    q_hat = q * rng.uniform(0.8, 1.2, 200)
    weighted_mse = np.mean((q - q_hat) ** 2 / q**2)
    assert weighted_mse == pytest.approx(rmspe(q * x, q_hat * x) ** 2, rel=1e-12)


def test_rmspe_rejects_non_positive_targets() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        rmspe([0.0, 1.0], [0.1, 1.0])


def test_rmspe_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        rmspe([1.0, 2.0], [1.0])


def test_log_back_transform_beats_the_plain_exponential() -> None:
    # METHOD.md section 7, test 7: simulated lognormal data.
    rng = np.random.default_rng(0)
    mu, s = -5.5, 0.35
    y = np.exp(rng.normal(mu, s, 200_000))
    corrected = rmspe(y, np.full_like(y, np.exp(mu - 1.5 * s**2)))
    plain = rmspe(y, np.full_like(y, np.exp(mu)))
    assert corrected < plain
    # exp(mu - 1.5 s^2) is the optimum: nearby constants do no better.
    for shift in (-0.05, 0.05):
        other = rmspe(y, np.full_like(y, np.exp(mu - 1.5 * s**2 + shift)))
        assert corrected < other


def test_block_indices_are_consecutive_blocks() -> None:
    from lobvol.evaluation import block_indices

    index = block_indices(n=100, block_length=20, n_resamples=50, seed=0)
    assert index.shape == (50, 100)
    assert index.min() >= 0 and index.max() <= 99
    blocks = index.reshape(50, 5, 20)
    assert (np.diff(blocks, axis=2) == 1).all()


def test_bootstrap_interval_covers_the_pooled_rmspe() -> None:
    import pandas as pd

    from lobvol.evaluation import block_indices, bootstrap_rmspe, interval, per_time

    rng = np.random.default_rng(1)
    n_time, n_stocks = 300, 10
    frame = pd.DataFrame(
        {
            "position": np.repeat(np.arange(n_time), n_stocks),
            "target": np.exp(rng.normal(-5, 0.3, n_time * n_stocks)),
        }
    )
    frame["a"] = frame["target"] * np.exp(rng.normal(0, 0.2, len(frame)))
    frame["b"] = frame["target"] * np.exp(rng.normal(0, 0.3, len(frame)))
    sums = per_time(frame, ["a", "b"])
    samples = bootstrap_rmspe(sums, ["a", "b"], block_indices(n_time, 20, 500, 0))
    low, high = interval(samples[:, 0])
    assert low < rmspe(frame["target"], frame["a"]) < high
    diff_low, diff_high = interval(samples[:, 0] - samples[:, 1])
    assert diff_high < 0  # model a is better in every plausible resample


def test_diebold_mariano_detects_a_better_model() -> None:
    from lobvol.evaluation import diebold_mariano

    rng = np.random.default_rng(2)
    loss_a = rng.gamma(2.0, 1.0, 500)
    noise = rng.normal(0, 0.5, 500)
    noise -= noise.mean()
    same = diebold_mariano(loss_a, loss_a + noise)
    better = diebold_mariano(loss_a, loss_a + 0.3 + noise)
    assert same["p_value"] > 0.99
    assert better["p_value"] < 0.001 and better["statistic"] < 0
