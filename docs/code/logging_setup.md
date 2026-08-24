# `src/risk_score/logging_setup.py`

## Purpose

Make every line the project emits answerable to a run or a request, and configure that in exactly one place.

Before this module the project printed.
`print` cannot be turned down in production, cannot be turned up during an incident, carries no timestamp, no level, and no origin, and writes to stdout - so piping a command's output through `jq` mixed diagnostics into the data.
More importantly, once the service is serving concurrent requests and retraining in a subprocess, an unattributed line is close to useless: "read 1,481,392 rows" is only meaningful if you can tell which of two runs said it.

So there are two ideas here.
Ids travel in **contextvars**, and the format is chosen by **environment variable**.

Contextvars rather than thread-locals, and that is load-bearing rather than stylistic: a `ContextVar` survives `asyncio` await points and propagates into `run_in_executor`, so an id bound in a FastAPI handler is still attached to a line logged from a thread pool, while a thread-local would silently return the default there.

## Public API

| Name | What it is |
| --- | --- |
| `configure_logging` | `dictConfig` once: level, format, stderr handler, library levels. Idempotent unless `force=True`. |
| `bind_run_id` | Context manager: every line inside the block carries this run id. |
| `bind_request_id` | The same for one request; generates a short id when the caller has none. |
| `capture_run_log` | Context manager: additionally tee every line into a file, then detach and close. |
| `ContextFilter` | Attaches `run_id` and `request_id` to every record. Filters nothing. |
| `UtcFormatter` | The base formatter: UTC clock, ISO-8601 with milliseconds and a `Z`. |
| `JsonFormatter` | One JSON object per line, with `extra=` fields merged in. |
| `LEVEL_ENV_VAR`, `JSON_ENV_VAR` | `RISKSCORE_LOG_LEVEL`, `RISKSCORE_LOG_JSON`. |

## Inputs and outputs

Reads two environment variables and writes to stderr, plus one file per run when `capture_run_log` is active.

```
2026-08-24T18:33:20.563Z INFO    risk_score.pipeline [run=20260824T183320563Z-… req=-] split train=… rows train=629
```

With `RISKSCORE_LOG_JSON=1`:

```json
{"timestamp": "2026-08-24T18:33:20.563Z", "level": "INFO", "logger": "risk_score.pipeline", "message": "split train=… rows train=629", "run_id": "20260824T183320563Z-…", "request_id": "-"}
```

Both formats stamp in **UTC**, spelled with a `Z` and carrying milliseconds.
Run ids, manifest timestamps, and the embargo snapshot are all UTC, so a line stamped in local time would make correlating a line to the run that emitted it an exercise in offset arithmetic - and the offset changes silently between a laptop and a container, where `TZ` is usually unset.
The `UtcFormatter` base class is where that lives; `JsonFormatter` inherits it.

**stderr, not stdout,** so `riskscore runs --json | jq` works: the answer to the question goes to stdout and the narration goes to stderr.

Outside any run or request both ids read as `-` rather than as `None` or an empty string, so a human-format line keeps its column alignment and a JSON consumer always finds the key.

`capture_run_log` is what puts `run.log` inside the run directory it describes.
The file is opened inside the staging directory, so it is published by the same rename as the metrics, and a failed run discards its log along with the artifacts the log describes.

## Invariants and failure modes

**Configuring twice does not double every line.**
`configure_logging` returns early when the root logger already has handlers, unless `force=True`.
The library is importable from a host that configured logging its own way - a uvicorn worker, a notebook, another application - and stealing that configuration would be rude and would also duplicate output.

**`disable_existing_loggers: False`.**
The default `dictConfig` behaviour disables every logger that already exists, which silences module-level loggers created at import time - which is all of them, since each module does `LOGGER = logging.getLogger(__name__)` at the top.

**The context filter is attached to the *handler*, not to the loggers.**
A filter on a logger only sees records that logger emits, so `sklearn` and `matplotlib` records would arrive without the ids and the format string would raise `KeyError` while formatting them.
On the handler, every record is enriched no matter who logged it.

**`capture_run_log` lowers the root level and puts it back.**
A run log that is empty because the process was started at `WARNING` is a support call, so the capture temporarily lowers the threshold for the duration.
The handler is detached and closed in a `finally`, so a failed run does not leak a file descriptor or leave a handler pointed at a deleted directory - and there is a test for the failure path specifically.

**An unserializable `extra` field does not break the log call.**
`json.dumps(..., default=str)`, so a `Path`, a `Timestamp`, or a numpy scalar in `extra=` appears as text.
A logging call that raises inside an exception handler loses the original error, which is the worst possible time to be strict.

**Tracebacks go to the log and stay out of any response.**
The formatter renders `exc_info`.
The service's exception handler logs with the request id and returns a generic body, so a filesystem path or a dataset date range never reaches a client.

**Noisy libraries are pinned to `WARNING`.**
`matplotlib` at `DEBUG` emits thousands of font-cache lines per figure, which turns `-v` from useful into unusable.

## What must NOT live here

- **Calling `configure_logging` at import time.** Importing a library must not reconfigure the host's logging. The CLI and the service's lifespan call it; nothing else does.
- **Business logging.** No module here decides what is worth logging; each module logs its own facts.
- **Metrics or tracing.** A counter is not a log line, and a project this size does not need an exporter to prove it.
- **Log shipping, rotation, or retention.** The process writes to stderr and one file per run. Whatever runs the process owns the rest, which is what makes the container story trivial.

## Related tests

`tests/test_logging_setup.py`, 18 tests.

The suite has one structural quirk worth knowing about, because it caused five failures that looked like bugs in the code and were not.
**pytest's logging plugin re-attaches its capture handler to the root logger at the start of each test phase.**
A fixture that empties `root.handlers` during setup is therefore undone before the test body runs, so `configure_logging()` sees handlers, returns early, and every assertion downstream is vacuous.
Isolation has to happen inside the call phase, so it is an `isolated_root()` context manager used in the test body rather than a fixture.

- `test_configuring_twice_does_not_duplicate_every_line` is the idempotence guarantee.
- `test_the_filter_attaches_both_ids_and_filters_nothing` and `test_outside_a_run_or_request_the_ids_are_a_dash` cover the enrichment path.
- `test_nested_binding_restores_the_outer_run_not_the_default` is the contextvar reset-token behaviour, which a naive save-and-restore gets wrong.
- `test_logs_go_to_stderr_so_piped_stdout_stays_clean` is the one that keeps `--json` pipeable.
- `test_an_unserializable_extra_field_does_not_break_the_log_call` and `test_the_traceback_goes_into_the_log_and_stays_out_of_any_response` cover the two ways a formatter can make an incident worse.
- `test_both_formats_timestamp_in_utc` sets `TZ=America/New_York` and compares against a UTC clock read either side of the call, because there is no way to assert "this is UTC" from the text alone.
- `test_the_run_log_is_detached_even_when_the_run_fails` is the `finally` path.
- `test_the_run_log_works_from_an_unconfigured_process` matters because a subprocess retrain does not inherit the parent's handlers.

## Known limits

- **The ids are two fixed fields, not arbitrary structured context.** A dataset id or a job id has to ride in `extra=`, where it reaches the JSON format but not the human one. Two ids cover the two questions actually asked ("which run", "which request"), and a general context dict would need a merge policy nobody has asked for.
- **JSON output is one line per record with no schema.** Field names are stable by convention and by test, not by contract.
- **No sampling or rate limiting.** A pathological loop can fill a disk. The runbook's answer is the level environment variable.
- **`capture_run_log` captures the whole process,** not just the calling thread. Two concurrent runs in one process would interleave into both logs. The service retrains in a subprocess, so this does not arise, and single-flight job admission means it cannot.
- **Timestamps are UTC with no local-time rendering.** Deliberate, and the same choice as the run id, but it does mean a local `tail -f` shows a clock that is not the wall clock.
