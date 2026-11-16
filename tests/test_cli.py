"""The CLI: every documented command exists and the holdout is guarded."""

import json
from pathlib import Path

import pytest

from lobvol.cli import (
    ALL_SEQUENCE,
    COMMANDS,
    EXIT_REFUSED,
    HANDLERS,
    build_parser,
    main,
)
from lobvol.runlog import write_run_record

CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")


def test_every_documented_command_is_registered() -> None:
    documented = {
        "audit",
        "features",
        "order",
        "splits",
        "xsection",
        "train",
        "evaluate",
        "holdout",
        "all",
    }
    assert set(COMMANDS) == documented
    assert set(HANDLERS) == documented - {"all"}


def test_holdout_is_refused_without_confirm() -> None:
    assert main(["holdout", "--config", CONFIG]) == EXIT_REFUSED


def test_all_never_runs_the_holdout() -> None:
    assert "holdout" not in ALL_SEQUENCE
    assert ALL_SEQUENCE[-1] == "evaluate"


def test_train_accepts_the_documented_options() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--model",
            "gbdt",
            "--features",
            "C_strict",
            "--scheme",
            "walk_forward",
        ]
    )
    assert (args.model, args.features, args.scheme) == (
        "gbdt",
        "C_strict",
        "walk_forward",
    )


def test_train_needs_a_complete_run_description() -> None:
    assert main(["train", "--model", "naive", "--config", CONFIG]) == EXIT_REFUSED


def test_unknown_scheme_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--scheme", "my_own_split"])


def test_missing_config_fails_early(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["audit", "--config", str(tmp_path / "missing.yaml")])


def test_run_record_contains_the_required_context(tmp_path: Path) -> None:
    path = write_run_record(tmp_path, "audit", {"frozen": {"embargo": 10}}, [0, 1, 2])
    record = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "run_record_audit.json"
    assert record["config"] == {"frozen": {"embargo": 10}}
    assert record["seeds"] == [0, 1, 2]
    assert {"git_commit", "packages", "python", "command"} <= set(record)
    assert record["packages"]["numpy"] is not None
