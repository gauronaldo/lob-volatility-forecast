"""Shared fixtures: one synthetic dataset and a config that points at it."""

import copy
from pathlib import Path
from typing import Any

import pytest

from lobvol.config import load_config
from lobvol.synth import SynthData, generate

REPO_ROOT = Path(__file__).resolve().parents[1]
N_STOCKS = 24
N_TIME_IDS = 200


@pytest.fixture(scope="session")
def base_config() -> dict[str, Any]:
    return load_config(REPO_ROOT / "configs" / "base.yaml")


@pytest.fixture(scope="session")
def pipeline_synth(tmp_path_factory: pytest.TempPathFactory) -> SynthData:
    """A synthetic dataset large enough for order recovery and splits."""
    out = tmp_path_factory.mktemp("pipeline") / "raw"
    return generate(out, n_stocks=N_STOCKS, n_time_ids=N_TIME_IDS, seed=7)


@pytest.fixture(scope="session")
def synth_config(base_config: dict[str, Any], pipeline_synth: SynthData):
    """The base config with paths redirected to the synthetic dataset."""
    config = copy.deepcopy(base_config)
    root = pipeline_synth.raw_dir.parent
    config["paths"] = {
        "raw": str(pipeline_synth.raw_dir),
        "interim": str(root / "interim"),
        "features": str(root / "features"),
        "splits": str(root / "interim" / "splits"),
        "results": str(root / "results"),
        "reports": str(root / "reports"),
    }
    return config
