"""Run records: config, package versions, git commit and seeds next to outputs."""

from __future__ import annotations

import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

STACK = (
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "torch",
    "optuna",
    "shap",
    "PyYAML",
)


def package_versions() -> dict[str, str | None]:
    """Return the installed version of every package in the project stack."""
    versions: dict[str, str | None] = {}
    for name in STACK:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit() -> str | None:
    """Return the current git commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_run_record(
    out_dir: str | Path,
    command: str,
    config: dict[str, Any],
    seeds: list[int],
) -> Path:
    """Write `run_record_<command>.json` into the output directory of a command."""
    record = {
        "command": command,
        "config": config,
        "seeds": seeds,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "packages": package_versions(),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"run_record_{command}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path
