"""Tests for the command line.

Through :func:`main` with an explicit ``argv`` rather than through a subprocess:
the parsing, the dispatch and the exit code are the surface worth testing, and a
subprocess would add a second interpreter start to every case for no extra
coverage. The one thing that genuinely needs the installed entry point - that
``riskscore`` resolves at all - is asserted against ``pyproject.toml`` instead.

What is asserted here is mostly *behaviour under bad input*, because that is
where the previous ``scripts/`` entry points were wrong: they swallowed every
traceback into a one-line ``SystemExit``.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from risk_score.artifacts import read_active_run_id, read_registry
from risk_score.cli import EXIT_USER_ERROR, TRAIN_SUMMARY_KEYS, build_parser, main
from risk_score.pipeline import METRICS_FILENAME

#: The synthetic default of 4000 rows takes a few seconds to fit; these tests are
#: about argument handling, so they use the smallest extract that still yields
#: positives in all three partitions.
CLI_ROWS = 1200


@pytest.fixture
def sample_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """An extract written by the CLI itself, so `train` consumes what
    `make-sample-data` produces rather than a fixture that only resembles it."""
    destination = tmp_path / "loans.csv"
    assert main(["make-sample-data", str(destination), "--rows", str(CLI_ROWS)]) == 0
    capsys.readouterr()
    return destination


def train(tmp_path: Path, csv: Path, *extra: str) -> list[str]:
    """One `train` invocation against a tmp report tree, cache off."""
    return ["train", str(csv), "--output-dir", str(tmp_path / "reports"), "--no-cache", *extra]


# --- the parser ----------------------------------------------------------------


def test_the_entry_point_declared_in_pyproject_actually_resolves() -> None:
    """The one thing a subprocess-free test cannot check by importing: that the
    console script names a module and a function that exist."""
    manifest = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    target = manifest["project"]["scripts"]["riskscore"]

    assert target == "risk_score.cli:main"


def test_a_bare_invocation_is_a_usage_error_not_a_crash() -> None:
    """`required=True` on the subparsers, so `riskscore` alone prints usage and
    exits 2 rather than raising `AttributeError: handler`."""
    with pytest.raises(SystemExit) as raised:
        main([])

    assert raised.value.code == 2


@pytest.mark.parametrize("command", ["train", "runs", "activate", "make-sample-data"])
def test_every_subcommand_is_reachable_and_documented(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--help` is the only documentation a command line has at the moment
    somebody needs it, so each subcommand must be listed at the top level and say
    what it does when asked directly.

    Asserted through the rendered help text rather than by walking argparse's
    private ``_subparsers``, so the test breaks when the *output* regresses and
    not when argparse rearranges its internals.
    """
    assert command in build_parser().format_help()

    with pytest.raises(SystemExit) as raised:
        main([command, "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert f"usage: riskscore {command}" in help_text
    # The description sits between the usage block and the arguments, so its
    # presence is what distinguishes a documented command from a bare signature.
    assert len(help_text.splitlines()) > 6, f"{command} has no description"


def test_an_unknown_model_is_rejected_by_the_parser(tmp_path: Path, sample_csv: Path) -> None:
    """`choices` comes from `SUPPORTED_MODEL_TYPES`, so the error names the
    supported values and no work starts."""
    with pytest.raises(SystemExit) as raised:
        main(train(tmp_path, sample_csv, "--model", "random_forest"))

    assert raised.value.code == 2
    assert not (tmp_path / "reports").exists()


# --- make-sample-data ----------------------------------------------------------


def test_make_sample_data_writes_a_csv_the_pipeline_can_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "nested" / "loans.csv"

    assert main(["make-sample-data", str(destination), "--rows", "50", "--seed", "7"]) == 0

    # The parent is created, because "the directory does not exist" is not a
    # useful thing to make somebody fix by hand.
    assert destination.exists()
    assert "wrote 50 rows" in capsys.readouterr().out


def test_make_sample_data_is_deterministic(tmp_path: Path) -> None:
    """Byte-identical for one seed, which is what lets the demo, the docs and CI
    all quote the same numbers."""
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    main(["make-sample-data", str(first), "--rows", "40", "--seed", "3"])
    main(["make-sample-data", str(second), "--rows", "40", "--seed", "3"])

    assert first.read_bytes() == second.read_bytes()


def test_a_nonsense_row_count_is_a_user_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 3, so a script can tell "you typed it wrong" from "it is broken"."""
    assert main(["make-sample-data", str(tmp_path / "x.csv"), "--rows", "0"]) == EXIT_USER_ERROR

    assert "--rows must be at least 1" in capsys.readouterr().err
    assert not (tmp_path / "x.csv").exists()


def test_the_percent_string_flag_changes_the_extract(tmp_path: Path) -> None:
    """Both real extracts exist - `'13.56%'` in one, `13.56` in the other - and
    the flag is how the demo reproduces either."""
    strings, floats = tmp_path / "s.csv", tmp_path / "f.csv"
    main(["make-sample-data", str(strings), "--rows", "40"])
    main(["make-sample-data", str(floats), "--rows", "40", "--float-rates"])

    assert "%" in strings.read_text(encoding="utf-8")
    assert "%" not in floats.read_text(encoding="utf-8")


# --- train ---------------------------------------------------------------------


def test_train_publishes_a_run_and_reports_it_on_stdout(
    tmp_path: Path, sample_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The summary is the answer to the question the user asked, so it goes to
    stdout where it can be piped; the log went to stderr."""
    assert main(train(tmp_path, sample_csv)) == 0
    out = capsys.readouterr().out

    registry = read_registry(tmp_path / "reports")
    assert len(registry) == 1
    run_id = registry[0]["run_id"]
    assert f"run {run_id}" in out
    assert (tmp_path / "reports" / "runs" / run_id / METRICS_FILENAME).exists()
    # Every declared summary line rendered a number rather than the "-" that
    # stands for a metric this pipeline no longer emits.
    for _, label in TRAIN_SUMMARY_KEYS:
        assert f"{label:<26} " in out
    assert " -\n" not in out


def test_train_activates_the_run_unless_told_not_to(tmp_path: Path, sample_csv: Path) -> None:
    root = tmp_path / "reports"
    main(train(tmp_path, sample_csv))
    first = read_active_run_id(root)

    main(train(tmp_path, sample_csv, "--no-activate"))

    # Two runs published, and the second did not take over serving - which is
    # what a comparison run needs.
    assert len(read_registry(root)) == 2
    assert read_active_run_id(root) == first


def test_the_tier_flag_reaches_the_run_id_and_the_manifest(
    tmp_path: Path, sample_csv: Path
) -> None:
    """`--include-lender-priced` is the one flag whose effect must be visible
    without opening the model, because a score built on the lender's own price
    cannot be used on an unpriced applicant."""
    main(train(tmp_path, sample_csv, "--include-lender-priced"))
    entry = read_registry(tmp_path / "reports")[0]

    assert "with_lender_priced" in entry["run_id"]
    assert entry["feature_tier"] == "with_lender_priced"


def test_a_missing_extract_is_reported_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The frames would all be library code; the path is the whole message."""
    assert main(train(tmp_path, tmp_path / "absent.csv")) == EXIT_USER_ERROR

    captured = capsys.readouterr()
    assert "absent.csv" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "reports").exists()


def test_a_broken_config_is_reported_without_a_traceback(
    tmp_path: Path, sample_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Config validation raises `ValueError` with a message written for whoever
    typed the command, so printing it beats printing where it was raised."""
    config = tmp_path / "config.yaml"
    config.write_text("include_lender_price: true\n", encoding="utf-8")

    assert main(train(tmp_path, sample_csv, "--config", str(config))) == EXIT_USER_ERROR

    captured = capsys.readouterr()
    assert "include_lender_price" in captured.err
    assert "Traceback" not in captured.err


def test_a_bug_keeps_its_traceback(
    tmp_path: Path, sample_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of replacing the old `scripts/` entry points.

    `raise SystemExit(str(error)) from None` discarded both the traceback and the
    `__cause__`, so a `KeyError` inside a transformer arrived as one
    unattributable line. Anything that is not a recognized user error propagates.
    """

    def explode(*args: object, **kwargs: object) -> None:
        raise KeyError("credit_utilization")

    monkeypatch.setattr("risk_score.cli.train_run", explode)

    with pytest.raises(KeyError, match="credit_utilization"):
        main(train(tmp_path, sample_csv))


def test_the_cache_is_on_by_default_and_off_on_request(tmp_path: Path, sample_csv: Path) -> None:
    """Opposite defaults on purpose: a CLI writing under a cache directory is
    expected, a library call doing it unasked is a surprise."""
    cache = tmp_path / "cache"
    base = ["train", str(sample_csv), "--output-dir", str(tmp_path / "reports")]

    main([*base, "--no-cache", "--cache-dir", str(cache)])
    assert not cache.exists()

    main([*base, "--cache-dir", str(cache)])
    assert list(cache.glob("*.parquet"))


# --- runs and activate ---------------------------------------------------------


def test_runs_on_an_empty_tree_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exit 0 with a sentence, not a traceback and not an empty table: "no runs
    yet" is a normal state, most obviously on a fresh clone."""
    assert main(["runs", "--output-dir", str(tmp_path / "reports")]) == 0

    assert "no runs" in capsys.readouterr().out


def test_runs_lists_newest_first_and_marks_the_active_one(
    tmp_path: Path, sample_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "reports"
    main(train(tmp_path, sample_csv))
    main(train(tmp_path, sample_csv, "--no-activate", "--include-lender-priced"))
    capsys.readouterr()

    assert main(["runs", "--output-dir", str(root)]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if "logistic_regression" in line]

    ids = [entry["run_id"] for entry in read_registry(root)]
    assert [line.split()[-4] for line in lines] == list(reversed(ids))
    # The active run is the older one here, because the newer was published with
    # `--no-activate`, so the marker cannot be confused with "first row".
    assert lines[0].startswith(" ")
    assert lines[1].startswith("*")


def test_runs_json_emits_the_registry_verbatim(
    tmp_path: Path, sample_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """So the dashboard and any script read the same bytes the library wrote,
    rather than reparsing a table built for humans."""
    root = tmp_path / "reports"
    main(train(tmp_path, sample_csv))
    capsys.readouterr()

    main(["runs", "--output-dir", str(root), "--json"])

    assert json.loads(capsys.readouterr().out) == read_registry(root)


def test_runs_rebuild_reconstructs_a_lost_registry(
    tmp_path: Path, sample_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The registry is an index; the manifests are the data. Deleting it must be
    recoverable, because otherwise a truncated JSON file loses run history."""
    root = tmp_path / "reports"
    main(train(tmp_path, sample_csv))
    before = read_registry(root)
    (root / "registry.json").unlink()
    capsys.readouterr()

    assert main(["runs", "--output-dir", str(root), "--rebuild"]) == 0

    assert read_registry(root) == before
    assert before[0]["run_id"] in capsys.readouterr().out


def test_activate_rolls_serving_back_to_an_older_run(
    tmp_path: Path, sample_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Runs are immutable, so reverting a bad model is repointing one file rather
    than retraining - which is the difference between a minute and an hour."""
    root = tmp_path / "reports"
    main(train(tmp_path, sample_csv))
    first = read_registry(root)[0]["run_id"]
    main(train(tmp_path, sample_csv))
    assert read_active_run_id(root) != first
    capsys.readouterr()

    assert main(["activate", first, "--output-dir", str(root)]) == 0

    assert read_active_run_id(root) == first
    assert first in capsys.readouterr().out


def test_activating_an_unknown_run_is_a_user_error(
    tmp_path: Path, sample_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "reports"
    main(train(tmp_path, sample_csv))
    active = read_active_run_id(root)
    capsys.readouterr()

    assert main(["activate", "no-such-run", "--output-dir", str(root)]) == EXIT_USER_ERROR

    # The failed activation left serving exactly where it was.
    assert read_active_run_id(root) == active
    assert "no-such-run" in capsys.readouterr().err


def test_activate_refuses_a_run_whose_bundle_cannot_be_loaded(
    tmp_path: Path, sample_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bundle is loaded and thrown away on purpose: activating an unreadable
    run would move the failure to the service's next boot, where it is an outage
    instead of a message on somebody's terminal."""
    root = tmp_path / "reports"
    main(train(tmp_path, sample_csv))
    main(train(tmp_path, sample_csv))
    healthy = read_active_run_id(root)
    broken = read_registry(root)[0]["run_id"]
    (root / "runs" / broken / "model.joblib").write_bytes(b"not a pickle")
    capsys.readouterr()

    with pytest.raises(Exception, match=r"."):
        main(["activate", broken, "--output-dir", str(root)])

    assert read_active_run_id(root) == healthy
