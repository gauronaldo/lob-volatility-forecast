"""Command-line interface: `python -m lobvol <command>` (see README.md)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from lobvol.config import DEFAULT_CONFIG, load_config

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_NOT_IMPLEMENTED = 3
EXIT_GATE_FAILED = 4

MODELS = ["naive", "har_ridge", "forest", "gbdt", "gbdt_fixed", "mlp", "blend"]
FEATURE_SETS = ["A", "B", "C_strict", "C_comp", "D", "C_strict_gap", "D_gap"]
SCHEMES = ["kfold_rows", "group_time", "blocked_time", "walk_forward"]

# command -> (stage, help text)
COMMANDS = {
    "audit": ("S1", "audit the raw data and write results/audit.md"),
    "features": ("S2", "in-bucket features, one stock_id at a time"),
    "order": ("S3", "recover and validate the time order"),
    "splits": ("S4", "cut the holdout and generate the folds"),
    "xsection": ("S6", "cross-sectional and temporal features"),
    "train": ("S5/S7", "train one model, or the whole run matrix without options"),
    "evaluate": ("S7", "tables, intervals and figures"),
    "holdout": ("S8", "holdout evaluation; runs once, with --confirm"),
    "all": ("S1-S7", "run S1..S7 and evaluate; never the holdout"),
}
ALL_SEQUENCE = ["audit", "features", "order", "splits", "xsection", "train", "evaluate"]

Handler = Callable[[dict[str, Any], argparse.Namespace], int]


def _gate(passed: bool) -> int:
    """Translate a gate result into an exit code."""
    return EXIT_OK if passed else EXIT_GATE_FAILED


def _audit(config: dict[str, Any], args: argparse.Namespace) -> int:
    from lobvol.audit import run_audit

    return _gate(run_audit(config, force=args.force))


def _features(config: dict[str, Any], args: argparse.Namespace) -> int:
    from lobvol.features import run_features

    run_features(config, force=args.force)
    return EXIT_OK


def _order(config: dict[str, Any], args: argparse.Namespace) -> int:
    from lobvol.time_order import run_order

    return _gate(run_order(config, force=args.force))


def _splits(config: dict[str, Any], args: argparse.Namespace) -> int:
    from lobvol.cv import run_splits

    run_splits(config, force=args.force)
    return EXIT_OK


def _xsection(config: dict[str, Any], args: argparse.Namespace) -> int:
    from lobvol.cross_section import run_xsection

    settings = [args.setting] if args.setting else ["strict", "competition"]
    for setting in settings:
        run_xsection(config, setting, force=args.force)
    return EXIT_OK


def _train(config: dict[str, Any], args: argparse.Namespace) -> int:
    from lobvol.pipeline import run_matrix, run_one

    if args.model is None:
        return _gate(run_matrix(config, force=args.force))
    if args.features is None or args.scheme is None:
        logger.error("--model needs --features and --scheme")
        return EXIT_REFUSED
    run_one(config, args.model, args.features, args.scheme, force=args.force)
    return EXIT_OK


def _evaluate(config: dict[str, Any], args: argparse.Namespace) -> int:
    from lobvol.report import run_evaluate

    run_evaluate(config)
    return EXIT_OK


def _holdout(config: dict[str, Any], args: argparse.Namespace) -> int:
    from lobvol.holdout import run_holdout

    return _gate(run_holdout(config))


HANDLERS: dict[str, Handler] = {
    "audit": _audit,
    "features": _features,
    "order": _order,
    "splits": _splits,
    "xsection": _xsection,
    "train": _train,
    "evaluate": _evaluate,
    "holdout": _holdout,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one subcommand per pipeline command."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="base config")
    common.add_argument("--force", action="store_true", help="redo existing outputs")

    parser = argparse.ArgumentParser(prog="lobvol", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (stage, text) in COMMANDS.items():
        cmd = sub.add_parser(name, parents=[common], help=f"{stage}: {text}")
        if name == "xsection":
            cmd.add_argument("--setting", choices=["strict", "competition"])
        elif name == "train":
            cmd.add_argument("--model", choices=MODELS)
            cmd.add_argument("--features", choices=FEATURE_SETS)
            cmd.add_argument("--scheme", choices=SCHEMES)
        elif name == "holdout":
            cmd.add_argument("--confirm", action="store_true")
    return parser


def run(command: str, config: dict[str, Any], args: argparse.Namespace) -> int:
    """Run one command and return its exit code."""
    if command == "holdout" and not getattr(args, "confirm", False):
        logger.error(
            "The holdout runs once, with --confirm, and only after every model "
            "is frozen (METHOD.md rule 2)."
        )
        return EXIT_REFUSED
    if command == "all":
        defaults = {"setting": None, "model": None, "features": None, "scheme": None}
        for key, value in defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)
        for step in ALL_SEQUENCE:
            code = run(step, config, args)
            if code != EXIT_OK:
                logger.error("`%s` stopped the pipeline with exit code %d", step, code)
                return code
        return EXIT_OK
    return HANDLERS[command](config, args)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments, load the config and run the requested command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config(args.config)
    config["config_dir"] = str(Path(args.config).parent)
    return run(args.command, config, args)
