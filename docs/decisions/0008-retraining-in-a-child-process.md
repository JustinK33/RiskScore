# 0008 - Retraining in a spawned child process, one slot, 409 rather than a queue

Status: accepted.
Affects `src/risk_score/api/jobs.py`, `src/risk_score/api/routes_admin.py`, `src/risk_score/api/settings.py`, and `src/risk_score/api/app.py`.

## Context

The dashboard this service replaces had exactly one POST endpoint, and it did not score an applicant.
It retrained the model.

```python
# scripts/serve_dashboard.py, the version that was deleted
length = int(self.headers.get("Content-Length", 0))
body = self.rfile.read(length)
...
run_pipeline(uploaded_csv)  # on the request thread
```

Five separate problems, and they are worth separating because they have five different fixes.

**It ran on the request thread.**
`http.server.HTTPServer` is single-threaded, so a fit that takes ten minutes on the real extract is ten minutes during which the dashboard does not answer at all - not slowly, not partially: the socket is never accepted.
A `ThreadingHTTPServer` would have made it answer while stalling, which is worse, because a fit holds the GIL inside BLAS and every other request would have been served at whatever rate the fit chose to release it.

**It could not be stopped.**
Nothing in scikit-learn checks a cancellation flag.
A fit that has decided to take an hour takes an hour, and the only way to end it early is to end the process it is in - which, on the request thread, is the server.

**It wrote to the live tree.**
`run_pipeline` overwrote `reports/metrics.json`, `reports/figures/*.png`, and the pickle, in place, one file at a time.
ADR 0007 fixes that half - the run directory - but the *concurrency* half is here: two uploads arriving together were two fits writing the same paths, interleaved, with no lock and no detection.

**It fitted on whatever bytes arrived, unauthenticated.**
`pd.read_csv` on a stranger's upload, and the run ended by writing a pickle the server would later load.

**It drew figures.**
matplotlib's pyplot keeps a global figure registry that is not thread-safe, so even the well-behaved version of "do the fit on a worker thread" is unsound as long as the run renders a calibration plot - which it does, because the plot is one of the artifacts.

A service with a retrain endpoint needs an answer to all five before the endpoint is defensible.

## Decision

**One retrain at a time, in a process spawned per job, watched by a thread, with the slot released on every exit path.**

`JobRunner.submit` claims the slot under a lock, starts a `multiprocessing` process with the `spawn` context, closes the parent's copy of the pipe's write end, and starts a watcher thread.
The route returns `202` with a job id.
`GET /api/jobs/{job_id}` is where the answer arrives.

Six parts.

**1. A process, not a thread, for three independent reasons.**
A fit holds the GIL, so a thread would stall the event loop and every `/predict` with it.
pyplot's global state is not thread-safe and the run writes figures.
And a process can be killed, which is the only way to enforce a timeout on code that does not check for cancellation.
Any one of the three would be sufficient on its own; having three is why this is not a `run_in_executor` call.

**2. Spawn, not fork.**
The parent holds an event loop, a thread pool, and an open bundle.
Forking copies every one of them into a child that is about to spend ten minutes in BLAS, and a forked event loop is a documented source of hangs.
Spawn costs about a second of interpreter startup, which is nothing against a fit, and it is already the default on macOS - so fork would only ever have been the *untested* path.
The context is requested explicitly rather than left to the platform, because fork is still the default on Linux and the CI runners are Linux.

**3. One slot, and a second request is `409` with `Retry-After`, not a queue.**
Two concurrent fits on the 1.19 GB extract is an out-of-memory kill, and the process that gets killed is chosen by the kernel rather than by us.
A queue would accept work it cannot promise to do - the connection that submitted it is long gone by the time it runs, and nothing bounds how much is waiting.
A `409` tells the caller the truth immediately, and `Retry-After` means a poller does not have to invent an interval.

**4. The child sends its result over a pipe, and catches everything before it does.**
An uncaught exception in the child would leave the parent with an exit code and no reason, and "why did the retrain fail" is the entire question a job status exists to answer.
So `_child` wraps `train_run` in a bare `except Exception`, sends `{"error": "TypeName: message"}`, and closes the connection in a `finally`.
The payload is a few hundred bytes, well under the pipe buffer, so it cannot deadlock against the parent's `join`.
The parent closes its copy of the write end immediately after `start()`: while any copy is open a read blocks instead of reporting EOF, so a child that died without sending would hang the watcher rather than be noticed.

**5. A watcher thread, not an asyncio task, and the slot is released on every path.**
`Process.join` is blocking, and the alternative is polling `is_alive()` on the event loop - a busy wait that also delays the answer by up to its interval.
The escalation is `join(timeout)` → `terminate()` → `join(grace)` → `kill()` → `join(grace)`, because a fit inside a BLAS call does not return to check for `SIGTERM`.
After the grace period the slot is released regardless of whether the child is provably dead, since holding it forever is the exact failure the timeout exists to prevent.

The broad `except` around the whole watcher body is the most important handler in the module.
An exception escaping that thread would leave `_active` set with no process behind it, and every subsequent retrain would be refused with a `409` naming a job that finished long ago - a service that permanently believes it is busy, recoverable only by a restart.
Its `_finish` call passes `only_if_running=True`, so a failure in the post-success reload cannot overwrite a `succeeded` status with `failed`.

**6. The service swaps to the new run only after the child exits successfully.**
`on_success` is injected rather than imported, so this module knows nothing about FastAPI.
It runs after the slot is released and the status is written, so a poller sees `succeeded` even if the reload itself fails - and a failed reload is `/readyz`'s problem, which is where it will be visible.
Throughout the fit, `/predict` keeps serving the bundle it already had.

**And the data the job runs on is content-addressed, uploaded through a separate switch.**
`POST /api/runs` takes a `dataset_id`, never a path and never CSV text.
A path would let a caller name any file the process can read; a body would put a 64 MiB upload inside the request that also has to start a fit.
`store_dataset` streams to a `.part` file in the destination directory, hashes while writing, enforces the cap against bytes that *arrived* rather than against `Content-Length`, refuses anything containing a NUL byte, fsyncs, and renames.
Both flags - `RISKSCORE_ALLOW_UPLOAD` and `RISKSCORE_ALLOW_RETRAIN` - are off by default and checked before the API key, so an operator who has not enabled the feature is told that rather than asked for a credential that would not help.
`Settings` refuses to start at all if a mutating route is enabled on a non-loopback bind with no key configured.

## Consequences

**The retrain endpoint is defensible enough to ship switched off.**
Four independent barriers stand between a request and a pickle write: two feature flags, an API key whose absence is a boot failure on a public bind, a content-addressed dataset id, and a killable process with one slot.
It stays off by default anyway, documented as a local-demo feature, because remote-triggered execution over caller-supplied data is not something a portfolio project should have listening.

**`202` is a receipt, so the client has to poll.**
That is a real interface cost - a caller wanting one blocking call has to write a loop - and it is the only shape that does not hold a connection open for the length of a fit.
`Retry-After` appears on the `409` and on a still-running status, and is absent once the job finishes, which is itself the signal that there is nothing more to wait for.

**No per-stage progress.**
A child cannot write into the parent's memory, so a "now fitting the calibrator" string needs an IPC channel - a `multiprocessing.Manager` proxy, which is a third process - for a status line.
`ponytail:` status, timing, and the error, plus the elapsed seconds a poller can already compute.
Upgrade path if it is ever wanted: pass a manager `dict` to `_child` and have the run's stage callbacks write into it.

**A job's history is bounded at 32 entries, and an evicted job is a `404`.**
The run *registry* is the durable record of what was trained; the job history exists only so a poller that asks about the job it just submitted gets an answer after the process is gone.
An unknown job and an evicted one get the same answer, because both mean this service cannot tell you.

**`JobOut.detail` carries the child's exception text, which is why job status needs the key.**
It is a server-side error message, and the one place this service would otherwise volunteer an internal failure to a caller.

**Spawn costs about a second per job and re-imports the world in the child.**
Invisible against a fit and unavoidable given the reasons above.
It also means the child re-runs `configure_logging`, which is why that function is idempotent.

**`JobRunner` is per app, not a module global.**
Two apps in one test process would otherwise share a slot, and a test that submitted a job would make an unrelated one return `409`.
The cost is that it has to be constructed in `create_app` and reached through `app.state`.

**The child is injectable, and production never injects.**
`target=_child` is a constructor parameter purely so the runner's own behaviour - the timeout, the escalation, a child that dies without reporting - is testable without a ten-second fit per case.
`tests/test_jobs.py` is 25 tests on stub children for that reason, and `test_a_real_retrain_publishes_a_run_and_the_service_swaps_to_it` is the one that uses the real one.

**Nothing removes stored datasets.**
The per-file cap bounds one upload; nothing bounds how many.
Acceptable for a feature that is off by default and local, and the first thing to fix if it ever were not.

## Alternatives considered

**`ProcessPoolExecutor(max_workers=1)`**, which is what the plan for this phase originally specified.
It is the obvious answer, and it is the one the plan for this phase named.
Rejected on two counts.
A `Future` cannot be cancelled once running, so the timeout would still need `terminate()` on a process the executor owns and does not expose - and reaching into `_processes` to find it is exactly the kind of thing that breaks on a point release.
And a pool *reuses* its worker, so the second fit in a process runs with whatever matplotlib, numpy, and pandas left behind from the first, which is a class of bug that reproduces only on the second request.
One process per job, discarded afterwards, has neither problem, and the pool bought nothing else: there is only ever one worker, and the pipe is four lines.

**A thread, with a lock around the fit.**
Fails all three of the process reasons: the GIL stalls `/predict`, pyplot's globals are not thread-safe, and a thread cannot be killed at all - `threading` has no `terminate`, by design.
A stuck fit would mean a restart, and "restart the service to clear a stuck retrain" is not an operational answer for a service whose whole point is answering `/predict`.

**A separate worker service with a real queue - Celery, RQ, arq.**
The right answer for a system with many jobs, several workers, and retries that matter.
Here it adds a broker (Redis), a second process to deploy, a serialization boundary, and a client library, so that a *demo* endpoint can run one job at a time on the same machine.
The queue is the feature being deliberately declined: one slot and a `409` is a correct answer that fits in one file.

**Just run `riskscore train` as a subprocess and parse its stdout.**
Tempting - it reuses the CLI exactly, and the CLI is already the supported interface.
Rejected because the failure detail would have to be recovered by scraping a log, the result would have to be recovered by scraping a run id out of stdout, and neither survives a format change.
A pipe carrying a dict is the same amount of code and says what it means.
`spawn` already re-imports the interpreter, so the "reuse the entry point" argument buys less than it appears to.

**Accept a CSV directly on `POST /api/runs` and skip the dataset store.**
One fewer route and one fewer flag.
Rejected because it merges two decisions - *may this caller store bytes here* and *may this caller start a fit* - that an operator has good reason to answer differently, and it puts a 64 MiB body inside the request that also has to start a process.
Content addressing additionally makes "somebody replaced the file the model trained on" a state that does not exist, which a filename-keyed store cannot claim.

**No timeout, on the grounds that a fit takes as long as it takes.**
The slot is the shared resource, and a fit that will never finish holds it against every future retrain.
An unbounded job needs no timeout only if it also needs no slot.

**Queue submissions instead of refusing them, bounded at some small depth.**
Considered seriously, because a `409` is mildly annoying to script against.
Rejected because the honest bound is 1: nothing in this service benefits from a fit that starts after the client that asked for it has gone, and a bounded queue of depth 1 is a `409` with extra steps and an extra state to test.
