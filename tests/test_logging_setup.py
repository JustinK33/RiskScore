"""Tests for log configuration and the two context ids.

Logging is easy to write and easy to get subtly wrong in ways no test notices:
a handler configured twice duplicates every line, a contextvar reset to its
default instead of its previous value silently unbinds an outer run, and a file
handler left attached keeps writing into a directory that has been renamed. Each
of those has a test here.

Every test restores the root logger itself. There is no autouse fixture doing it,
because a leaked handler in *this* file would then be invisible, and the leak is
one of the things under test.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from risk_score.logging_setup import (
    JSON_ENV_VAR,
    LEVEL_ENV_VAR,
    ContextFilter,
    JsonFormatter,
    bind_request_id,
    bind_run_id,
    capture_run_log,
    configure_logging,
    request_id_var,
    run_id_var,
)


@contextmanager
def isolated_root() -> Iterator[logging.Logger]:
    """A root logger with no handlers, restored exactly afterwards.

    A context manager rather than a fixture, and that is not a style choice:
    pytest's logging plugin installs its own capture handler at the start of each
    test *phase*, so a fixture that empties `root.handlers` during setup is undone
    before the test body runs - and `configure_logging` would then see handlers
    and return early, making every assertion below vacuous. Clearing has to happen
    inside the call phase.
    """
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    root.handlers = []
    try:
        yield root
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers, root.level = handlers, level


def record(**extra: object) -> logging.LogRecord:
    made = logging.LogRecord("risk_score.test", logging.INFO, __file__, 1, "scored", None, None)
    for key, value in extra.items():
        setattr(made, key, value)
    return made


# --- the context ids -----------------------------------------------------------


def test_the_filter_attaches_both_ids_and_filters_nothing() -> None:
    """A filter on the *handler* enriches records from sklearn and uvicorn too,
    which is why it is not a logger adapter."""
    with bind_run_id("run-1"), bind_request_id("req-1"):
        made = record()
        assert ContextFilter().filter(made) is True
        # `getattr`, because these attributes exist only after the filter runs:
        # `logging.LogRecord` has no such fields and mypy is right about that.
        assert getattr(made, "run_id") == "run-1"  # noqa: B009
        assert getattr(made, "request_id") == "req-1"  # noqa: B009


def test_outside_a_run_or_request_the_ids_are_a_dash() -> None:
    """A dash rather than an empty string, so the fixed-width line stays aligned
    and a grep for the field still matches."""
    made = record()
    ContextFilter().filter(made)

    assert (getattr(made, "run_id"), getattr(made, "request_id")) == ("-", "-")  # noqa: B009


def test_nested_binding_restores_the_outer_run_not_the_default() -> None:
    """`riskscore compare` fits two models inside one outer run. Resetting to the
    default instead of the previous value would leave the outer run untagged for
    everything logged after the first inner one finished."""
    with bind_run_id("outer"):
        with bind_run_id("inner"):
            assert run_id_var.get() == "inner"
        assert run_id_var.get() == "outer"

    assert run_id_var.get() == "-"


def test_a_request_id_is_generated_when_the_caller_has_none() -> None:
    with bind_request_id() as generated:
        assert request_id_var.get() == generated
        assert len(generated) == 12

    with bind_request_id("from-the-gateway") as supplied:
        # A gateway that already issued an id must not have its trace split in two.
        assert supplied == "from-the-gateway"


# --- formatting ----------------------------------------------------------------


def test_the_json_formatter_emits_one_object_with_the_ids_and_extras() -> None:
    """`extra=` fields are merged rather than dropped, which is what makes
    `logger.info("scored", extra={"latency_ms": 3.1})` worth writing."""
    made = record(latency_ms=3.1, run_id="run-1", request_id="req-1")

    payload = json.loads(JsonFormatter().format(made))

    assert payload["message"] == "scored"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "risk_score.test"
    assert payload["run_id"] == "run-1"
    assert payload["request_id"] == "req-1"
    assert payload["latency_ms"] == 3.1
    # Standard record attributes are not copied in wholesale.
    assert "msecs" not in payload
    assert "args" not in payload


def test_an_unserializable_extra_field_does_not_break_the_log_call() -> None:
    """A log call is not the place to raise. A Path or a numpy scalar in `extra=`
    must appear as text rather than turn a diagnostic into a second exception."""
    payload = json.loads(JsonFormatter().format(record(dataset=Path("/tmp/loans.csv"))))

    assert payload["dataset"] == "/tmp/loans.csv"


def test_the_traceback_goes_into_the_log_and_stays_out_of_any_response() -> None:
    def failing_split() -> None:
        raise ValueError("split produced an empty partition")

    try:
        failing_split()
    except ValueError:
        import sys

        made = record()
        made.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(made))

    assert "ValueError: split produced an empty partition" in payload["exception"]


# --- configuration -------------------------------------------------------------


def test_configuring_twice_does_not_duplicate_every_line() -> None:
    """The failure this prevents is two identical lines per event, which reads as
    a loop rather than as a logging bug."""
    with isolated_root() as root:
        configure_logging()
        configure_logging()

        assert len(root.handlers) == 1


def test_the_level_and_format_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read directly rather than through the settings object, because the first
    thing worth logging is often a settings error."""
    monkeypatch.setenv(LEVEL_ENV_VAR, "DEBUG")
    monkeypatch.setenv(JSON_ENV_VAR, "1")

    with isolated_root() as root:
        configure_logging()

        assert root.level == logging.DEBUG
        assert isinstance(root.handlers[0].formatter, JsonFormatter)


def test_the_human_format_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wall of JSON in a terminal is a worse local experience than a grep that
    needs a regex, and running this locally is the common case."""
    monkeypatch.delenv(JSON_ENV_VAR, raising=False)

    with isolated_root() as root:
        configure_logging()

        formatter = root.handlers[0].formatter
        assert not isinstance(formatter, JsonFormatter)
        assert formatter is not None
        # Both ids are in the default line, so a grep for one is possible without
        # switching the whole process to JSON.
        assert "run=%(run_id)s" in (formatter._style._fmt or "")
        assert "req=%(request_id)s" in (formatter._style._fmt or "")


def test_logs_go_to_stderr_so_piped_stdout_stays_clean(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`riskscore runs | jq` must not have log lines interleaved into the data."""
    with isolated_root():
        configure_logging()
        logging.getLogger("risk_score.test").info("hello")

    captured = capsys.readouterr()
    assert "hello" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("json_output", [False, True])
def test_both_formats_timestamp_in_utc(
    json_output: bool, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run ids, manifests and the embargo snapshot are all UTC.

    A log line stamped in local time would make correlating a line to the run
    that emitted it offset arithmetic, and the offset changes silently between a
    laptop and a container where `TZ` is unset. Pinned by comparing against a UTC
    clock read either side of the call rather than against a fixed string, since
    there is no way to assert "this is UTC" from the text alone.
    """
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    before = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")

    with isolated_root():
        configure_logging(json_output=json_output)
        logging.getLogger("risk_score.test").info("stamped")

    after = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")
    line = capsys.readouterr().err
    stamp = json.loads(line)["timestamp"] if json_output else line.split()[0]

    assert stamp.startswith((before, after))
    # Milliseconds and an explicit `Z`, so the format is unambiguous to a reader
    # and sorts lexically like the run ids it sits next to.
    assert stamp.endswith("Z")
    assert stamp[-5] == "."


def test_the_noisy_libraries_are_pinned_to_warning() -> None:
    """matplotlib logs a line per font it inspects, which buries a run's own
    output in a report that draws one chart."""
    previous = logging.getLogger("matplotlib").level
    try:
        with isolated_root():
            configure_logging()

            assert logging.getLogger("matplotlib").level == logging.WARNING
    finally:
        logging.getLogger("matplotlib").setLevel(previous)


# --- the per-run log file ------------------------------------------------------


def test_the_run_log_captures_lines_with_their_run_id(tmp_path: Path) -> None:
    """Written inside the staging directory, so it is published by the same atomic
    rename as the metrics and a failed run takes its log with it."""
    destination = tmp_path / "run.log"

    with capture_run_log(destination), bind_run_id("run-1"):
        logging.getLogger("risk_score.test").info("fitting on 3791 rows")

    contents = destination.read_text(encoding="utf-8")
    assert "fitting on 3791 rows" in contents
    assert "run=run-1" in contents


def test_the_run_log_works_from_an_unconfigured_process(tmp_path: Path) -> None:
    """A handler is only offered records the *logger* already admitted, and an
    unconfigured root defaults to WARNING. Without lowering it the file would be
    empty and look like a run that logged nothing."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    root.handlers, root.level = [], logging.WARNING
    try:
        with capture_run_log(tmp_path / "run.log"):
            logging.getLogger("risk_score.test").info("admitted")
        assert "admitted" in (tmp_path / "run.log").read_text(encoding="utf-8")
        # And the level is put back, so the block does not silently make the rest
        # of the process chatty.
        assert root.level == logging.WARNING
    finally:
        root.handlers, root.level = handlers, level


def test_the_run_log_handler_is_detached_and_closed_afterwards(tmp_path: Path) -> None:
    """Left attached it keeps writing into a directory that has since been renamed
    by `staged_run`, which on POSIX succeeds silently and produces a log nobody
    can find."""
    root = logging.getLogger()
    before = len(root.handlers)

    with capture_run_log(tmp_path / "run.log"):
        pass
    logging.getLogger("risk_score.test").info("after the block")

    assert len(root.handlers) == before
    assert "after the block" not in (tmp_path / "run.log").read_text(encoding="utf-8")


def test_the_run_log_is_detached_even_when_the_run_fails(tmp_path: Path) -> None:
    root = logging.getLogger()
    before = len(root.handlers)

    with pytest.raises(RuntimeError, match="fit exploded"), capture_run_log(tmp_path / "run.log"):
        raise RuntimeError("fit exploded")

    assert len(root.handlers) == before
