# `dashboard/js/retrain.js`

## Purpose

Retrain from a CSV, without blocking on the fit.

This replaces the one POST the old dashboard had, and it is worth stating exactly what that POST did, because every decision here is a response to part of it. It read a CSV out of a `<textarea>`, with no credential, retrained the whole model **on the request thread**, and overwrote the canonical `reports/` tree - including the pickle the service was serving - while every other request waited on the GIL. Five separate problems: unauthenticated, unbounded input, synchronous, destructive, and non-atomic.

The flow here is three steps because each one carries a different risk:

1. `POST /api/datasets` - storing bytes. Size-capped, CSV-sniffed, content-addressed under its own SHA-256, so uploading the same file twice stores nothing twice.
2. `POST /api/runs` - starting a fit. Authenticated, single-flight, answers `202` with a `job_id`. The fit runs in a child process (see [ADR 0008](../decisions/0008-retraining-in-a-child-process.md)).
3. `GET /api/jobs/{id}` - watching. An ordinary poll that cannot change anything.

`/predict` keeps serving the old bundle throughout, and the new run only becomes active when the child exits successfully.

**Nothing here is optimistic.** The new run is not selected, no panel is patched, and no status is inferred from a timer. The job's own status decides, and a success reloads every report from the server - because a retrain has changed what every panel on the page describes.

## Public API

| Name | What it is |
| --- | --- |
| `jobLine(job, {elapsedSeconds})` | One line of progress from a job payload. `""` for no job. |
| `mountRetrainPanel({section, form, banner, progress, submitButton, mutatingRoutes, onDone, now})` | Wire the panel, or return without doing anything. |

`MODEL_TYPES = ["logistic_regression", "xgboost"]` mirrors `RetrainIn.model_type`; `LENDER_PRICED_TIER = "with_lender_priced"` is the tier option that maps to `include_lender_priced: true`.

## Inputs and outputs

Takes the panel's nodes and the `mutating_routes` string from `/readyz`.
Calls `uploadDataset`, `startRetrain`, `waitForJob`, and `setApiKey` from `api.js`; renders through `setBanner` and `setText` from `dom.js`.
On success calls `onDone(run_id)`, which is `main.js`'s reload.

`now` is injectable so the elapsed counter is testable without a clock.

## Invariants and failure modes

**The panel stays hidden unless the service reports `mutating_routes === "both"`. This is the security-shaped decision in the file.**
The default is off. Not "either" - **both**: the browser has no server-side `dataset_id` to refer to, so uploading is the only way it can name one, and a retrain route with uploads switched off is reachable from the CLI and not from here. Offering a form whose every submission is a 403 is worse than offering nothing, because it reads as a broken feature rather than a switched-off one, and the operator who needs it is reading the runbook.

`mountRetrainPanel` also returns early on a missing `section` or `form`, so a markup change cannot leave a half-wired panel.

**The model options are populated from JS, not written in the HTML.**
So the markup cannot drift from the list the service accepts. The probe asserts `retrainModels >= 2` whenever the panel is visible, which catches a select a reader cannot choose anything from.

**The key is stored before the first request, not read per call.**
One `setApiKey` covers the upload, the start, and every poll. It goes to `sessionStorage`, so it does not outlive the tab, and `api.js` sends it only on the three routes that declared they need it.

**The tier is chosen by name, not by a checkbox.**
The page calls the two tiers `origination_only` and `with_lender_priced` everywhere else - in the identity strip, the run picker, the comparison table. A checkbox labelled `include_lender_priced` would be the same choice spelled a third way, and a reader would have to work out that it corresponds to the tier named beside it.

**No file selected is a warning, not a request.**
Checked before anything is uploaded.

**The submit button is disabled for the whole flow and re-enabled in a `finally`.**
A second submit during a running fit would get a `409`, which is correct server behaviour and a confusing thing to show someone who just clicked twice.

**`jobLine` says what is true rather than inventing a percentage.**
The service reports no stage, because a fit is one child process with no checkpoints to report from. So the line names the model, the dataset, and how long it has been running - and says that this is a full training run that takes as long as `riskscore train` does, which is the answer to the question the reader actually has.

**A failed job carries the child's own `detail` verbatim.**
It is the only description of what went wrong - "no rows survived the embargo" is a real and useful message - and this route already required a credential to reach, so there is no information-disclosure argument for redacting it.

**A failed job with no detail says so, rather than reading as a success.**
`job.detail || "no detail was reported."`. Without the fallback the line ends in `undefined`, and a line ending in `undefined` is one a reader skims past.

**A timed-out job is distinguished from a failed one.**
"Killed after exceeding the server's timeout" and "the fit failed" are different diagnoses: the first says raise the timeout or shrink the dataset, the second says read the detail.

**An unrecognised status is reported by name.**
`Unknown job status: sideways.` The alternative - a `default` that falls through to the running message - would render a terminal state as still-running forever. `api.js`'s `waitForJob` makes the same choice on the polling side: an unknown status is terminal.

**No job at all is `""`, not the string `undefined`.**
`jobLine(null)` returns the empty string, which `setText` renders as `MISSING`.

**A non-successful terminal status sets the danger banner *and* leaves the progress line.**
Two places, because the banner is what gets noticed and the progress line is what gets read.

**`onDone` is only called on `succeeded`, and it reloads from the server.**
Not from cache, and not by patching the panels. The run is published and active at that point, so every report on the page describes the previous model until it is re-read. `main.js`'s `reload` invalidates the whole cache first.

**Any thrown error becomes a danger banner.**
Which includes a 401 from a wrong key, a 413 from an oversized file, a 409 from a concurrent fit, and a timeout. `api.js` has already translated each into a sentence.

## What must NOT live here

- **Any training logic.** The fit happens in a child process on the server. This file uploads bytes and polls.
- **A second opinion about whether the feature is available.** `/readyz`'s `mutating_routes` is the answer. Reading an env var, a build flag, or a URL parameter here would let the page offer something the server refuses.
- **Optimism.** No inferred progress, no patched panels, no locally-constructed run id.
- **The polling loop.** `waitForJob` in `api.js` owns the interval, the failure tolerance, and the terminal-status rule.
- **The key in `localStorage`, in a URL, or in a log.** `sessionStorage` only.

## Related tests

`dashboard/js/retrain.test.js`, 9 tests. No browser, no server, no file picker.

The suite is deliberately narrow, and the test file says why: two things here are worth testing and the rest is DOM wiring a unit test cannot reach honestly.

`jobLine` is the only description an operator gets of a fit that takes minutes, and a line that reads like a success when the job failed - or that drops the `detail` - leaves them with a stale model and no reason. Seven tests: `a running job names the model, the dataset, and how long it has been going`, `a succeeded job names the run, because that is what to look at next`, `a failed job carries the child's own detail verbatim`, `a failed job with no detail says so rather than reading as a success` (which asserts the absence of `undefined`/`null` in the output), `a timed-out job is distinguished from a failed one`, `an unrecognised status is reported, not silently rendered as running`, and `no job is an empty line rather than the string undefined`.

The mount guard is the security question, and it gets both directions. `the panel stays hidden unless the service reports both features on` loops over `"none"`, `"upload"`, `"retrain"`, and `undefined` and asserts `section.hidden === true` for each, naming the failing value in the message. `the panel appears when both are on` is its counterpart - without it, "always hide" would pass the first test and remove the feature.

Both run against a `stubPanel()` exposing only the surface the guard touches: `{section: {hidden: true}, form: {addEventListener, elements: {namedItem}}}`. No jsdom.

The rest is covered where it can be observed:

- `scripts/probe_dashboard.mjs` reads `mutating_routes` from the target server's own `/readyz` and asserts `retrainVisible === expectRetrain` at every width in both themes. That is the only check that distinguishes "the panel is correctly switched off" from "the panel failed to mount" - they are the same pixels. Verified against both a default server (`retrain=off`) and one started with `RISKSCORE_ALLOW_UPLOAD=1 RISKSCORE_ALLOW_RETRAIN=1` (`retrain=2m`), so both branches of the assertion are exercised.
- The full upload → start → poll → reload path was driven end to end in headless Chrome against a flags-on server: file attached via `DOM.setFileInputFiles`, real submit, real fit, and the identity strip's run id observed changing afterwards.
- The server side is `tests/test_routes_admin.py` (auth, the off-by-default flags, single-flight `409`, the size cap) and `tests/test_jobs.py` (the child process, the timeout, content addressing).

## Known limits

- **No cancel.** A running fit cannot be stopped from the page. The server can kill it on timeout; an operator cannot. A `DELETE /api/jobs/{id}` would be the addition, and the child process is already killable - the gap is the route, not the mechanism.
- **No queue.** One fit at a time, and a second attempt gets a `409` rendered as "wait for it to finish". Deliberate: a queue means deciding what happens to a queued job when the service restarts.
- **The progress line cannot say how far along the fit is.** There is no stage reporting in the child, so the elapsed seconds are the only progress signal. Real stages would need the child to write progress somewhere the parent reads.
- **Reload is all-or-nothing.** A success invalidates the whole cache and refetches ten reports. Fine at this scale; a partial invalidation would be the optimization, and it would need to know which reports a retrain can change - which is all of them.
- **The dataset is uploaded before the key is validated.** A wrong key fails at the upload step, after the bytes have been sent. Checking the key first would mean an extra authenticated round trip on every retrain to save bandwidth on a mistyped key.
- **No upload progress.** A 64 MiB CSV shows "Uploading loans.csv" and nothing else until it lands. A progress bar needs `XMLHttpRequest` or a `ReadableStream` body with a counting `TransformStream`; `fetch` with a `File` body reports nothing.
