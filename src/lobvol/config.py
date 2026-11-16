"""Loading of the YAML configuration files under `configs/`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/base.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return config
