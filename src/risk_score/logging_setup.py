"""Logging configuration, and the two ids that make a log line traceable.

The project had no logging at all. A training run printed nothing, and the
dashboard's retrain endpoint reported failure as an HTTP 500 with the traceback
in the response body - which is both a leak and, for anyone debugging later, a
record that exists only in the browser that happened to be open.

Two ids do most of the work here:

* ``run_id`` - which training run a line belongs to. Set once per run, and
  written into that run's own ``run.log`` inside the staged directory, so a
  published run carries its own log and a failed one takes its log with it.
* ``request_id`` - which HTTP request a line belongs to. Set per request by the
  service, returned in a response header, and included in the generic error body
  so a user can quote it and a maintainer can find the traceback that produced
  it without the body ever containing a path or a stack.

Both are :mod:`contextvars`, not thread-locals and not arguments threaded through
every signature. Contextvars are the only option that survives ``asyncio``: a
request handler that awaits mid-way is resumed on a task, and a thread-local set
before the await belongs to whatever else the event loop ran in between. They
also propagate into ``run_in_executor`` calls, which is how a background retrain
keeps its ``run_id`` without being passed one.

Format is a choice with one honest answer per audience, so both exist:
human-readable lines by default because a developer reads them in a terminal,
JSON when ``RISKSCORE_LOG_JSON=1`` because a log aggregator parses them. The
default is the local one, since running this project locally is the common case
and a wall of JSON in a terminal is a worse experience than a grep that needs a
regex.
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

#: The active training run, or ``"-"`` outside one. A dash rather than an empty
#: string so a fixed-width log line stays aligned and a grep for the field still
#: matches.
run_id_var: ContextVar[str] = ContextVar("run_id", default="-")

#: The active HTTP request, or ``"-"`` outside one.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

#: Environment variables read here. Logging is configured before settings are
#: parsed - the first thing worth logging is usually a settings error - so these
#: are read directly rather than through the service's settings object.
LEVEL_ENV_VAR = "RISKSCORE_LOG_LEVEL"
JSON_ENV_VAR = "RISKSCORE_LOG_JSON"

#: Fields already on every :class:`logging.LogRecord`. Anything else a caller
#: attached via ``extra=`` is a domain field and is merged into the JSON object,
#: which is what makes ``logger.info("scored", extra={"latency_ms": 3.1})`` useful
#: without a bespoke formatter per call site.
_STANDARD_RECORD_FIELDS = frozenset(
    set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "taskName"}
)

_HUMAN_FORMAT = (
    "%(asctime)s %(levelname)-7s %(name)s [run=%(run_id)s req=%(request_id)s] %(message)s"
)


class ContextFilter(logging.Filter):
    """Attach ``run_id`` and ``request_id`` to every record.

    A filter rather than a custom logger class or an adapter, because those only
    apply to loggers the project creates. A filter installed on the *handler*
    catches records from ``uvicorn``, ``sklearn``, and anything else in the
    process, so a warning from a library is attributable to the request that
    provoked it.

    Filters are also the documented place to mutate a record, and this one always
    returns ``True`` - it filters nothing, which is the standard idiom for
    enrichment.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_var.get()
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the context ids and any ``extra`` fields.

    ``default=str`` on the dump is deliberate: a log call is not the place to
    raise. A ``Path``, a ``Timestamp``, or a numpy scalar in ``extra=`` should
    appear as text rather than turn a diagnostic into a second exception.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            # The traceback goes in the log and never in an HTTP body.
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in vars(record).items():
            if key not in _STANDARD_RECORD_FIELDS and key not in payload:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(
    *,
    level: str | int | None = None,
    json_output: bool | None = None,
    force: bool = False,
) -> None:
    """Install one stderr handler for the whole process. Idempotent.

    Called from the CLI and from the service's lifespan handler, never from a
    library function: a module that configures logging on import steals the
    decision from the application embedding it, and the symptom is duplicated
    lines that nobody can find the source of.

    ``force=False`` (the default) leaves an existing configuration alone, so a
    process that has already been configured - by uvicorn's own ``--log-config``,
    or by a test - is not reconfigured behind its back.

    stderr rather than stdout, so a CLI command whose output is piped into
    another program does not have log lines interleaved into the data.
    """
    root = logging.getLogger()
    if root.handlers and not force:
        return

    resolved_level = level if level is not None else os.environ.get(LEVEL_ENV_VAR, "INFO")
    if json_output is None:
        json_output = os.environ.get(JSON_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}

    logging.config.dictConfig(
        {
            "version": 1,
            # False, not True: `disable_existing_loggers` silences every logger
            # created before this call, which on a late configure means the
            # library warnings emitted during import vanish.
            "disable_existing_loggers": False,
            "filters": {"context": {"()": ContextFilter}},
            "formatters": {
                "human": {"format": _HUMAN_FORMAT, "datefmt": "%Y-%m-%d %H:%M:%S"},
                "json": {"()": JsonFormatter},
            },
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                    "formatter": "json" if json_output else "human",
                    "filters": ["context"],
                }
            },
            "root": {"level": resolved_level, "handlers": ["stderr"]},
            "loggers": {
                # Chatty at INFO and never load-bearing: matplotlib logs a line
                # per font it inspects, and py4j-style transport noise from
                # urllib3 drowns the run's own lines.
                "matplotlib": {"level": "WARNING"},
                "PIL": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"},
            },
        }
    )


@contextmanager
def bind_run_id(run_id: str) -> Iterator[str]:
    """Tag every log line inside the block with ``run_id``.

    Resets to the previous value on exit rather than to the default, so nesting
    - a comparison run that fits two models - restores the outer run rather than
    clearing it. That is what ``ContextVar.reset`` is for and why the token is
    kept.
    """
    token = run_id_var.set(run_id)
    try:
        yield run_id
    finally:
        run_id_var.reset(token)


@contextmanager
def bind_request_id(request_id: str | None = None) -> Iterator[str]:
    """Tag every log line inside the block with a request id, generating one if
    the caller has none.

    ``uuid4().hex[:12]`` is short enough for a human to read back over the phone
    and wide enough that a collision within one process's logs is not a concern.
    A caller-supplied id is honoured, because a gateway that already issued one
    should not have its trace split in two.
    """
    resolved = request_id or uuid.uuid4().hex[:12]
    token = request_id_var.set(resolved)
    try:
        yield resolved
    finally:
        request_id_var.reset(token)


@contextmanager
def capture_run_log(path: str | Path, *, level: int = logging.INFO) -> Iterator[Path]:
    """Additionally write every log record to ``path`` for the duration.

    Used with :func:`risk_score.artifacts.staged_run`, so the file is written
    *inside* the staging directory and is published by the same atomic rename as
    the metrics. A run's log therefore ships with the run, and a failed run's log
    is discarded with it rather than being appended to a shared file where it
    outlives the artifacts it describes.

    The handler is removed and closed in a ``finally``: leaving it attached would
    keep writing into a directory that has since been renamed or deleted, which
    on POSIX succeeds silently and produces a log nobody can find.

    The root level is lowered for the duration if it is higher than ``level``,
    because a handler is only offered records the *logger* already admitted. An
    unconfigured process defaults to ``WARNING``, and without this a run.log
    written by a library caller would be empty and look like a run that logged
    nothing.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(destination, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_HUMAN_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    handler.addFilter(ContextFilter())
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    if previous_level > level or previous_level == logging.NOTSET:
        root.setLevel(level)
    try:
        yield destination
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(previous_level)
