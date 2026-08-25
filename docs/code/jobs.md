# `src/risk_score/api/jobs.py`

## Purpose

Run a retrain without breaking the service that triggered it, and store the data it runs on.

The dashboard this replaces retrained the model *on the request thread*.
One unauthenticated POST fitted a model on an uploaded CSV and overwrote the canonical `reports/` tree, pickle included, while every other request waited behind the fit.
That is three separate problems - an unbounded blocking call on the event loop, a destructive write to the live artifact tree, and arbitrary attacker-supplied data going into `pd.read_csv` - and this module is the answer to all three.

A retrain now runs in a **spawned child process** with a hard timeout, publishes atomically under a new run id, and the service swaps to it only after the child exits successfully.
`/predict` keeps serving the bundle it already has throughout, so no request ever sees a half-built model.
The data it trains on is uploaded separately, content-addressed, size-capped, and sniffed before it is stored.

## Public API

| Name | What it is |
| --- | --- |
| `JobRunner` | One retrain slot, its bounded history, and the thread that watches the child. Held on `app.state`. |
| `Job` | One retrain as a poller sees it: status, timings, run id, failure detail, exit code. |
| `TrainRequest` | What the child is asked to do, as picklable plain data. |
| `JobBusy` | Raised by `submit` when a retrain is already running. Surfaces as a 409. |
| `JobStatus`, `PENDING_STATUSES` | The status vocabulary, and which values mean "come back later". |
| `store_dataset`, `resolve_dataset`, `DatasetRef` | Content-addressed upload storage and lookup. |
| `looks_like_csv`, `UploadRejected` | The shallow format check, and the refusal it raises. |
| `HISTORY`, `TERMINATE_GRACE_SECONDS`, `UPLOAD_CHUNK_BYTES`, `SNIFF_BYTES` | The four tuning constants, each justified where it is defined. |

## Inputs and outputs

`submit` takes a `TrainRequest` and returns a `Job` immediately - the fit happens elsewhere.
The child writes a complete run directory through `artifacts.py`'s atomic staging, so the only observable output is a new run id appearing in the registry.

`store_dataset` takes anything with a blocking `read(size)` and writes `<datasets_dir>/<32 hex>.csv`.
The digest is the first 32 hex characters of the file's SHA-256, computed while streaming.
`Reader` is a `Protocol` rather than `IO[bytes]` because the real caller is not a file - it is an ASGI request body adapted to a blocking read - and `IO[bytes]` would demand `seek`, `tell`, `fileno` and a context manager that the adapter cannot provide and this function never calls.

`TrainRequest` is deliberately not a `RunConfig`: that object is reachable from the estimator registry, and pickling it would send far more across the process boundary than four scalars.
The child builds its own config from the same defaults the CLI uses.

## Invariants and failure modes

**A process, not a thread**, for three independent reasons.
A fit holds the GIL for minutes, so a thread would stall the event loop and every score with it.
matplotlib's pyplot carries global state that is not thread-safe, and a run writes figures.
And a process can be *killed*, which is the only way to enforce a timeout on scikit-learn code that never checks for cancellation.

**Spawn, not fork.**
The parent holds an event loop, a thread pool and an open pickle; forking that copies every one of them into a child about to spend ten minutes in BLAS.
Spawn costs about a second of interpreter startup, which is nothing against a fit.
It is also already the default on macOS, so fork would only ever have been the *untested* path.
The context is requested explicitly because fork is still the Linux default.

**One at a time, and a second request is a 409 rather than a queue.**
Two concurrent fits on the 1.19 GB extract is an out-of-memory kill.
A queue would accept work it cannot promise to do; a 409 tells the caller the truth immediately.
The slot is claimed under the lock *before* the process starts, so two simultaneous POSTs cannot both find it free.

**The slot is released on every path.**
This is the most important guarantee in the module and the one with the broadest `except`.
An exception escaping the watcher thread would leave `_active` set with no process behind it, so every later retrain would be refused with a 409 naming a job that finished long ago - a service that permanently believes it is busy, recoverable only by restart.
The fallback handler passes `only_if_running=True`, because overwriting a genuine "succeeded" with "failed" merely because the reload callback threw would be the worse of the two mistakes.

**A child that dies without reporting is distinguishable from one that raised.**
`_child` catches everything and sends the exception's type and message back down the pipe, because "why did the retrain fail" is the entire question a job status exists to answer.
No message plus a dead process means it never reached its own `except` - an OOM kill looks exactly like this - and that case records `exit_code`, so an operator can tell "the fit raised" from "the kernel killed it".
Those have completely different fixes.

**The write end of the pipe is closed in the parent immediately.**
While any copy of it is open, a read blocks instead of reporting EOF, so a child that died without sending would hang the watcher rather than be noticed.
`_receive` also guards the `recv`, because `poll` returns true at end of file as well as on data - a closed pipe is readable, it just reads as EOF - and without that guard a killed child raised `EOFError` inside the watcher thread.

**A timeout escalates and then gives up.**
`terminate`, wait `TERMINATE_GRACE_SECONDS`, then `kill`, then wait again.
A fit inside a BLAS call does not return to check for SIGTERM, so SIGKILL is not optional.
After the grace period the slot is released regardless, because holding it forever is the exact failure the timeout exists to prevent.

**Uploads are content-addressed, so an upload cannot overwrite a dataset.**
"Somebody replaced the file the model trained on" is not a state that exists here.
The same extract uploaded twice gets the same id, so a run's manifest points at something a reviewer can verify.
`existing` is decided immediately before the rename, because afterwards the two cases are indistinguishable - the destination name *is* the content, so a re-upload renames onto a file identical to itself.

**The size cap is enforced while reading, not from `Content-Length`.**
A chunked upload declares nothing, and a declared length is the caller's number.

**A partial upload leaves nothing.**
Written to a `.part` name in the destination directory and renamed once complete, so a dropped connection leaves a temporary file rather than a truncated dataset under a hash that does not describe it.
Same directory, because a rename is only atomic within one filesystem.
The cleanup catches `BaseException` rather than `Exception`, because a cancelled request arrives as `CancelledError` and it must not leave the partial file behind either.
`fsync` runs before the rename, so a crash cannot leave a correctly named dataset whose bytes never reached the disk - the one failure a content address cannot detect after the fact.

**Sniffing, not trusting the filename or the `Content-Type`.**
Both are supplied by the caller, and the thing being prevented is handing a joblib pickle to `pd.read_csv` and finding out what happens.
The check is deliberately shallow - empty, NUL bytes, not UTF-8, no comma in the first line.
Checking for NUL catches a pickle, a parquet file and a zip at once without a magic-number list to maintain.
This is not validation: the pipeline's own column contract is, and it runs in a process that can be killed, so the cost of accepting a bad CSV is a wasted retrain slot rather than a security problem.

**A dataset id gets two guards, like a run id.**
The pattern rejects anything that is not 32 hex characters, and the resolved path is checked for containment in case a symlink got there another way.
One guard on a filesystem path is one guard too few.

## What must NOT live here

- **FastAPI.** `on_success` is a callback injected by the app precisely so this module knows nothing about the framework. It is what reloads the bundle and drops the report cache.
- **Training logic.** `_child` calls `train_run` and does nothing else of consequence.
- **The authorization decision.** Whether a retrain or an upload is permitted at all is `routes_admin.py`'s question, behind `RISKSCORE_ALLOW_UPLOAD` and the API key.
- **A module-level runner.** It is constructed per app and held on `app.state`: two apps in one test process would otherwise share a slot, and a test that submitted a job would make an unrelated one return 409.
- **The durable record of what was trained.** That is the run registry. `HISTORY` exists only so a poller that asks about the job it just submitted gets an answer after the process is gone.

## Related tests

`tests/test_jobs.py`, using stub children for the runner's own behaviour so it is testable without a ten-second fit per case.
`self.target` is injected for exactly that reason and production never passes it.

- `test_a_second_submit_while_one_runs_is_refused` and `test_the_slot_is_released_once_the_child_finishes` are the single-flight contract.
- `test_a_hanging_child_is_killed_and_the_slot_released` is the timeout, and `test_a_killed_child_is_failed_with_its_exit_code` is the no-message path that an OOM produces.
- `test_a_failing_callback_does_not_fail_the_job` pins the `only_if_running` rule.
- `test_a_stream_that_breaks_midway_leaves_nothing` and `test_a_non_csv_upload_is_refused_after_streaming_and_leaves_nothing` are the cleanup guarantees; `test_a_size_cap_is_enforced_across_chunks` covers the cap on a body that declares nothing.
- `test_a_symlinked_dataset_never_escapes_the_directory` is the second guard in `resolve_dataset`.
- `test_the_real_child_reports_the_run_it_published` and `test_the_real_child_sends_the_failure_back_instead_of_raising` call `_child` directly, in-process. The end-to-end test below covers the same code spawned, where coverage.py cannot see it without subprocess instrumentation, and it never takes the failure branch - so the child read as untested while being the most consequential twenty lines here.
- `tests/test_routes_admin.py::test_a_real_retrain_publishes_a_run_and_the_service_swaps_to_it` is the end-to-end proof: a real child, a real fit, a new run id, and a service that scores with the new bundle afterwards.

## Known limits

- **No per-stage progress.** A child process cannot write into the parent's memory, so "now fitting the calibrator" needs an IPC channel - a `multiprocessing.Manager` proxy, which is a third process - for a status line. Reported instead: status, timings, the failure reason, and the elapsed seconds a poller can already compute. `ponytail:` the upgrade path is to pass a manager `dict` to `_child` and have the run's stage callbacks write into it.
- **The history dies with the process.** A restart loses every finished job. The registry survives, so what was *trained* is never lost; only "job 4f2a succeeded" is.
- **One slot per process, not per deployment.** Two replicas pointed at one report tree would each accept a retrain. The single-flight guarantee is in-process, and making it global needs a lock somewhere shared.
- **No cancellation endpoint.** A running job can only be waited out or the server restarted. The timeout is the backstop.
- **Uploads are never garbage-collected.** Content-addressed files accumulate under the datasets directory until somebody deletes them, which is always safe because a run's manifest records the digest rather than depending on the file.
