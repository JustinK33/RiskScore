"""The ``riskscore`` command line: the supported way to drive the library.

Every subcommand is a thin adapter over one library call. Nothing is computed
here that a caller importing :mod:`risk_score` could not compute, and nothing
here is imported by the library, so the package stays usable as a library and
the CLI stays deletable.

Three deliberate choices about how it behaves when things go wrong:

* **Tracebacks propagate.** The previous scripts wrapped every failure in
  ``raise SystemExit(str(error)) from None``, which discards the traceback and
  the ``__cause__`` - so a ``KeyError`` deep inside a transformer surfaced as one
  unattributable line. The only exceptions caught here are the two that are
  genuinely *user* errors and not bugs (a missing file, a malformed config), and
  those are reported without a traceback because there is nothing in it to read.
* **Exit codes are meaningful.** ``0`` success, ``1`` a bug (via the traceback),
  ``2`` bad usage from argparse, ``3`` a user error this module recognized.
* **The cache is on by default here and off in the library.** A CLI writing
  under ``data/cache`` is expected; a library function doing it unasked is not.
  ``--no-cache`` turns it off, and ``--cache-dir`` moves it.

``compare``, ``explain``, ``card``, ``serve`` and ``bench`` are not here yet -
they arrive with the modules that back them. A subcommand that exists and fails
is worse than one that does not exist, because ``--help`` stops being the answer
to what this tool can do.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from risk_score.artifacts import (
    DEFAULT_RETENTION,
    load_bundle,
    read_active_run_id,
    read_registry,
    rebuild_registry,
    set_active_run,
)
from risk_score.cache import DEFAULT_CACHE_DIR
from risk_score.config import RunConfig, load_run_config
from risk_score.logging_setup import configure_logging
from risk_score.modeling import SUPPORTED_MODEL_TYPES
from risk_score.pipeline import train_run
from risk_score.sample_data import make_synthetic_loans

LOGGER = logging.getLogger(__name__)

#: Told apart from a bug by argparse's 2 and an uncaught exception's 1. A user
#: error is one this module can describe in a sentence: the file is not there,
#: the config does not parse, the run id does not exist.
EXIT_USER_ERROR = 3

DEFAULT_OUTPUT_DIR = Path("reports")

#: What ``train`` prints, in order. Declared rather than inlined so the summary
#: is one list to edit, and read with :func:`_number` rather than indexed: a run
#: that published successfully must not then exit non-zero because the CLI asked
#: for a metric this version of the pipeline does not emit.
TRAIN_SUMMARY_KEYS = (
    ("auc_roc", "AUC (test)"),
    ("average_precision", "average precision (test)"),
    ("ks_statistic", "KS (test)"),
    ("brier_score", "Brier (calibrated)"),
    ("brier_score_uncalibrated", "Brier (uncalibrated)"),
    ("expected_calibration_error", "ECE (test)"),
    ("selected_threshold", "threshold (validation)"),
    ("approval_rate", "approval rate (test)"),
    ("default_rate", "default rate (test)"),
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns the process exit code.

    Returning rather than calling ``sys.exit`` so that tests can assert on the
    code without catching ``SystemExit``; the console-script wrapper below does
    the exiting.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level="DEBUG" if args.verbose else None)

    handler: Any = args.handler
    try:
        exit_code: int = handler(args)
    except (FileNotFoundError, IsADirectoryError) as error:
        # The one class of failure where a traceback is pure noise: the frames
        # would all be library code and the useful information is the path.
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USER_ERROR
    except ValueError as error:
        # Config validation and argument checking both raise ValueError with a
        # message written to be read by whoever typed the command.
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USER_ERROR
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """The whole command surface, so ``--help`` is the documentation."""
    parser = argparse.ArgumentParser(
        prog="riskscore",
        description="Train, inspect and publish credit default risk models.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log at DEBUG instead of INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_train(subparsers)
    _add_runs(subparsers)
    _add_activate(subparsers)
    _add_make_sample_data(subparsers)
    return parser


def _add_output_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="the run tree holding registry.json, active_run.json and runs/ (default: reports)",
    )


def _add_train(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "train",
        help="fit one model on one extract and publish it as a run",
        description=(
            "Reads the extract, applies the outcome-maturity embargo, splits by "
            "time, fits on train, calibrates and picks a threshold on validation, "
            "and reports on test exactly once."
        ),
    )
    parser.add_argument("data", type=Path, help="path to the raw extract CSV")
    _add_output_dir(parser)
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODEL_TYPES),
        default="logistic_regression",
        help="estimator to fit (default: logistic_regression)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML run configuration; omitted means the shipped defaults",
    )
    parser.add_argument(
        "--include-lender-priced",
        action="store_true",
        help=(
            "admit int_rate/grade/sub_grade/installment. These are the lender's "
            "own price, so the resulting model cannot score an applicant nobody "
            "has priced yet. See docs/decisions/0005."
        ),
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="publish and register the run without pointing the service at it",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=DEFAULT_RETENTION,
        help=f"how many runs to retain, newest first (default: {DEFAULT_RETENTION})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"parquet cache for the parsed extract (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="parse the CSV even if a cached copy exists",
    )
    parser.set_defaults(handler=_train)


def _train(args: argparse.Namespace) -> int:
    config = load_run_config(args.config) if args.config else RunConfig()
    if args.include_lender_priced:
        # Copied rather than mutated: RunConfig is frozen precisely so that the
        # configuration a run reports is the configuration it used.
        config = replace(config, include_lender_priced=True)

    result = train_run(
        args.data,
        config=config,
        output_dir=args.output_dir,
        model_type=args.model,
        make_active=not args.no_activate,
        keep_runs=args.keep,
        cache_dir=None if args.no_cache else args.cache_dir,
    )

    # Printed rather than logged: this is the answer to the question the user
    # asked, and it belongs on stdout where it can be piped. The log, which went
    # to stderr, is also published inside the run directory.
    rows = result.metadata.rows
    print(f"run {result.run_id}")
    print(f"  {result.run_dir}")
    print(f"  {'rows':<26} " + "  ".join(f"{name}={count}" for name, count in rows.items()))
    for key, label in TRAIN_SUMMARY_KEYS:
        print(f"  {label:<26} {_number(result.payload.get(key))}")
    if not result.payload.get("calibration_method"):
        return 0
    print(f"  {'calibration':<26} {result.payload['calibration_method']} on validation")
    return 0


def _add_runs(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "runs",
        help="list published runs, newest first",
        description=(
            "Reads reports/registry.json, the append-only index every run writes "
            "itself into. The active run - the one the service loads at boot - is "
            "marked. --rebuild reconstructs the index from the run manifests, "
            "which is the recovery path if the index is lost or truncated."
        ),
    )
    _add_output_dir(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the registry entries verbatim instead of a table",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="reconstruct the registry from the run manifests before listing",
    )
    parser.set_defaults(handler=_runs)


def _runs(args: argparse.Namespace) -> int:
    entries = rebuild_registry(args.output_dir) if args.rebuild else read_registry(args.output_dir)
    if args.json:
        print(json.dumps(entries, indent=2))
        return 0

    if not entries:
        print(f"no runs under {args.output_dir}")
        return 0

    active = read_active_run_id(args.output_dir)
    # Measured rather than guessed: a run id is a timestamp plus a model name
    # plus a tier plus a sha, and `with_lender_priced` alone is 12 characters
    # longer than `origination_only`. A hardcoded width misaligns the moment
    # somebody trains the other tier.
    width = max((len(str(entry.get("run_id", "?"))) for entry in entries), default=6)
    print(f"{'':2}{'run_id':<{width}}{'auc':>9}{'brier':>9}{'ece':>9}")
    # Newest first, which is the reverse of the registry's own order: the
    # registry is append-only so a reader can trust its ordering, and "which is
    # newest" is the question a listing answers.
    for entry in reversed(entries):
        metrics = entry.get("metrics", {})
        marker = "*" if entry.get("run_id") == active else " "
        print(
            f"{marker:2}{entry.get('run_id', '?')!s:<{width}}"
            f"{_number(metrics.get('auc_roc')):>9}"
            f"{_number(metrics.get('brier_score')):>9}"
            f"{_number(metrics.get('expected_calibration_error')):>9}"
        )
    if active:
        print("\n* = active, the run the service loads at boot")
    return 0


def _number(value: object) -> str:
    """A missing metric reads as ``-``, never as ``0.000``.

    An absent number rendered as zero is the kind of thing that gets read as a
    catastrophic model rather than as missing data.
    """
    return f"{value:.4f}" if isinstance(value, int | float) else "-"


def _add_activate(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "activate",
        help="point the service at a different published run",
        description=(
            "The rollback path: runs are immutable, so reverting a bad model is "
            "repointing active_run.json rather than retraining. The bundle is "
            "loaded first, so a run that cannot be served is refused here rather "
            "than at the service's next restart."
        ),
    )
    parser.add_argument("run_id", help="the run to serve")
    _add_output_dir(parser)
    parser.set_defaults(handler=_activate)


def _activate(args: argparse.Namespace) -> int:
    run_dir = Path(args.output_dir) / "runs" / args.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No run {args.run_id} under {Path(args.output_dir) / 'runs'}")

    # Deliberately loaded and thrown away. Activating a run whose pickle is
    # unreadable would move the failure to the service's next boot, where it is
    # an outage rather than a message on someone's terminal.
    bundle = load_bundle(run_dir)
    set_active_run(args.output_dir, args.run_id)
    identity = f"{bundle.metadata.model_type}, {bundle.metadata.feature_tier}"
    print(f"active run {args.run_id} ({identity})")
    return 0


def _add_make_sample_data(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "make-sample-data",
        help="write a synthetic Lending Club-shaped extract",
        description=(
            "So the project can be demonstrated without the 1.19 GB download. "
            "The output is deliberately messy in the same ways the real extract "
            "is - percent strings, ' 36 months', 'Aug-2003', junk columns - and "
            "carries the same survivorship bias, so the embargo has something to "
            "remove."
        ),
    )
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=Path("data/sample/loans.csv"),
        help="output CSV path (default: data/sample/loans.csv)",
    )
    parser.add_argument("--rows", type=int, default=8000, help="number of loans (default: 8000)")
    parser.add_argument("--seed", type=int, default=20130101, help="RNG seed; output is exact")
    parser.add_argument(
        "--float-rates",
        action="store_true",
        help="write int_rate/revol_util as floats instead of '13.56%%' strings",
    )
    parser.set_defaults(handler=_make_sample_data)


def _make_sample_data(args: argparse.Namespace) -> int:
    if args.rows < 1:
        raise ValueError(f"--rows must be at least 1, got {args.rows}")

    frame = make_synthetic_loans(
        n_rows=args.rows,
        seed=args.seed,
        percent_strings=not args.float_rates,
    )
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)

    print(f"wrote {len(frame)} rows x {len(frame.columns)} columns to {destination}")
    return 0


if __name__ == "__main__":
    # `python -m risk_score.cli`. The installed `riskscore` script points at
    # `main` directly, and setuptools' wrapper passes its return value to
    # `sys.exit`, so the exit code is the same either way.
    raise SystemExit(main())
