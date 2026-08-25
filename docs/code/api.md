# `dashboard/js/api.js`

## Purpose

The one place the dashboard talks to the service.

The old `app.js` had three inline `fetch` calls and no error model. Any non-OK response from any of them threw the same string - `"Report artifacts are missing. Run the baseline pipeline first."` - so a 401, a 503 during startup, and a genuinely absent report were one message on the page. The `request_id` in the body, which is the only handle on the traceback in the server log, was discarded.

Four things live here, and each one closes a specific failure:

**One error shape.** The service answers every failure with `{detail, request_id}` and a status. `ApiError` carries all three plus `Retry-After`, and every caller renders `error.message`.

**An in-memory cache of parsed payloads.** This is what makes the resize handler free: `main.js` redraws from the cached object and issues no request at all. The old resize handler refetched two endpoints per event and swallowed the outcome with `.catch(() => {})`, so a resize storm was a request storm whose failures were invisible - and out-of-order responses could paint stale data over fresh.

**In-flight de-duplication, keyed by URL.** Two panels wanting `/api/metrics` at boot share one request.

**A timeout on everything.** A `fetch` with no signal waits as long as the browser feels like. A dashboard stuck on "Loading reports" with no error is indistinguishable from a broken build.

HTTP-level revalidation is deliberately *not* reimplemented. The service sends `ETag` and `Cache-Control`, the browser handles both correctly, and `fetch` never even exposes the 304 - so a JS-side conditional-request layer would be a second cache to keep correct for no gain.

## Public API

| Name | What it is |
| --- | --- |
| `ApiError` | `Error` subclass with `status`, `detail`, `requestId`, `retryAfter`. |
| `request(path, options)` | The one request function. Options: `method`, `body`, `headers`, `signal`, `timeoutMs`, `authenticated`, `fetchImpl`. |
| `getJson(path, {signal, force})` | A cached, de-duplicated GET. |
| `peek(path)` | The cached payload or `undefined`. Never requests. |
| `invalidate(prefix = "")` | Drop cached payloads. No prefix drops everything. |
| `reportPath(name, runId)` | Appends `?run_id=` when there is one, encoded. |
| `getModel`, `getSchema`, `getMetrics`, `getCalibration`, `getThresholdCosts`, `getVintages`, `getDrift`, `getShapSummary`, `getComparison`, `getRuns`, `getRun` | Named endpoint wrappers over `getJson`. |
| `predict(applicant, {explain, topK, signal})` | `POST /predict`. Never cached. |
| `uploadDataset(file, {signal})` | `POST /api/datasets` with the raw `File` as the body. |
| `startRetrain({datasetId, modelType, includeLenderPriced})` | `POST /api/runs`. Returns the 202 receipt. |
| `getJob(jobId)` | One poll. |
| `waitForJob(jobId, {onUpdate, sleep, signal})` | Poll until terminal. |
| `getHealth({signal})` | `GET /readyz`. Never throws. |
| `parseRetryAfter(headerValue)` | Delta-seconds to milliseconds, or `null`. |
| `getApiKey()`, `setApiKey(key)` | The key, for this tab only. |
| `DEFAULT_POLL_MS` | `2000`. |

## Inputs and outputs

Paths in, parsed JSON out, `ApiError` on failure.
Imports nothing. Does not touch the DOM, and knows no metric names - the endpoint wrappers are the only domain vocabulary, and they are paths.

`request` returns `null` for a 204 and the parsed body otherwise.

`fetchImpl` is a parameter purely so `api.test.js` can drive the module with a stub and assert de-duplication, caching, and error translation with no server.

Two module-level `Map`s: `cache` holds settled payloads keyed by path, `pending` holds in-flight promises keyed by path. Neither is exported; `peek` and `invalidate` are the only handles.

## Invariants and failure modes

**Every failure is an `ApiError` with a status, including the ones that have no HTTP status.**
A network-level failure - server down, page offline, CORS refusal - arrives as a bare `TypeError` from `fetch` with no status and no body. It becomes `ApiError("Could not reach the server.", {status: 0})`. A timeout becomes `ApiError("The server did not answer within 15s.", {status: 0})`. Without this, a `.message` rendered into a banner reads `Failed to fetch` or `undefined`.

**A caller's own abort propagates unchanged.**
`request` builds `AbortSignal.any([signal, timeout])`, and on rejection checks `signal?.aborted` *first*. A cancelled request is not an error to report - the panel that asked for it has already been replaced - so wrapping it in an `ApiError` would make every navigation show a banner.

**`STATUS_HINTS` translates the statuses that mean something specific here.**
`409` is "a retrain is already running", not "Error 409". `503` is "no model loaded yet, train one with `riskscore train`". The server's own `detail` wins when present; these are the fallbacks for a response that never reached the app - a proxy 502, a Starlette-level rejection.

**A non-JSON error body is expected, not exceptional.**
`toApiError` parses in a `try` and falls back to `detail = ""`. Without that, a proxy's HTML error page would replace the real 502 with a `SyntaxError` about an unexpected `<`.

**`requestId` is never dropped.**
The service deliberately keeps filesystem paths and dataset date ranges out of error bodies and puts the traceback in its log under this id. A UI that discards it leaves the operator with "it returned 500" and nothing to join it to.

**`parseRetryAfter` honours only delta-seconds, and returns `null` for anything else.**
The HTTP-date form is legal and this service never sends it. The critical part is that an unrecognised value must be `null` and not `0`: a poller that reads "wait" as "retry immediately" is a hot loop against an endpoint that just asked it to back off.

**A rejected request never enters `cache`.**
The payload is written in a `.then`, so only a fulfilled request is remembered, and the next ask retries. Caching a failure would make one transient 503 during startup permanent for the life of the tab.

**The `pending` slot is only cleared if it is still ours.**
`if (pending.get(path) === promise) pending.delete(path)`. A `force` request started while an ordinary one is in flight would otherwise delete the other's entry on settling, leaving a live promise nobody can join.

**`force` bypasses the cache and still populates it.**
That is what the poll-and-refresh path needs: read through, then let a subsequent resize redraw from the fresh value.

**The API key is sent only on routes that declared they need it.**
`authenticated: true` is set on `uploadDataset`, `startRetrain`, and `getJob`, and nowhere else. A key is never volunteered to an unauthenticated report endpoint, never put in a URL, and never logged.

**The key lives in `sessionStorage`, not `localStorage`.**
It is a credential and should not outlive the tab it was given to. Both accessors are wrapped in `try`/`catch` because storage throws outright when cookies are blocked, and an unusable key store is not a reason to fail the whole page.

**`predict` is never cached.**
Two identical bodies are two decisions, and the response carries a `latency_ms` that would be a lie coming out of a `Map`.

**`uploadDataset` sends the raw `File`, not `FormData`.**
The route reads `request.stream()` directly so it can enforce its size cap against bytes that actually arrived rather than trusting `Content-Length`. A multipart body would be rejected as CSV containing a boundary. The timeout is raised to ten minutes because this is up to 64 MiB over whatever link the operator has.

**`getJob` goes through `request`, not `getJson`.**
A cached job status would report `running` forever.

**`waitForJob` cannot loop forever.**
Terminal statuses are `succeeded`, `failed`, and `timed_out` - and *any unrecognised status is also terminal*. A client that keeps polling on a status it does not understand never stops, and the version of that bug where the server gains a new status is invisible until it ships.

**A single failed poll is tolerated; a persistent one is not; a 404 is fatal at once.**
`POLL_FAILURES_ALLOWED = 3` consecutive failures. The service reloads the new bundle immediately after the child process exits, and a poll landing in that window can time out - so giving up on the first failure would report a successful retrain as a failure. A 404 is different in kind: the job has been evicted from the bounded history, and no amount of waiting brings it back.

**`onUpdate` fires for every poll including the last**, so the caller shows progress without owning a timer.

**`getHealth` never throws.**
It is called first on every load and its result decides what the header badge says. A 503 with a `detail` becomes `{status: "degraded", bundle_loaded: false, reason: detail}`, because that reason - "no active run" - is the most useful sentence on the page at that moment. Anything else becomes `reason: "The service is not reachable."`. Its timeout is 4s rather than 15s, because a health check that takes fifteen seconds to fail has already failed.

## What must NOT live here

- **The DOM.** No banners, no status pills, no nodes. `main.js` and `retrain.js` render what this module returns.
- **Formatting.** A payload comes back as the service sent it. `format.js` decides how a number reads.
- **Conditional-request handling.** `If-None-Match`, `304`, `ETag` comparison. The browser already does this with the headers the service sends.
- **Retry of anything other than a job poll.** A failed report GET is reported, not retried. Silent retries hide a server that is failing, and the reader is sitting in front of a reload button.
- **A second storage key.** The theme key lives in `main.js` and `index.html`; this file owns exactly `riskscore.apiKey`.

## Related tests

`dashboard/js/api.test.js`, 17 tests, all against a stub `fetchImpl`. No server, no network.

- Caching and de-duplication: `a second get of the same path issues no request`, `peek returns the cached payload and never requests`, `force refetches and replaces the cached payload`, `invalidate with a prefix leaves other paths cached`, and `concurrent gets of one path share a single request` - the last asserts a request *count* of one from two simultaneous callers, which is the property the boot sequence relies on.
- `a failed request is not cached, so the next ask retries` is the invariant that keeps a startup 503 from being permanent.
- Error translation: `an error body's detail and request id survive onto the thrown error`, `a status with no usable body still gets a human sentence`, `a network failure is an ApiError with status 0, not a raw TypeError`, and `a caller's own abort propagates rather than becoming an ApiError`.
- `the api key is sent only on authenticated routes` inspects the stub's recorded headers for both a report GET and a mutating POST. This is a security assertion, not a convenience one.
- `a run id becomes a query parameter and is encoded`.
- `retry-after is parsed as seconds, and anything odd is null` covers the empty string, an HTTP-date, a negative, and `NaN`.
- Polling, with an injected `sleep` so the suite does not wait: `polling stops on a terminal status and reports every poll`, `an unknown status is terminal, so a client cannot loop forever`, `a 404 gives up at once, because an evicted job never comes back`, and `a transient poll failure is tolerated, a persistent one is not`.

The HTTP contract on the other side of these paths is tested in `tests/test_api.py` and `tests/test_routes_admin.py`; the end-to-end join is `scripts/probe_dashboard.mjs`, which drives a real load, a real `POST /predict`, and a real run switch in Chrome.

## Known limits

- **`waitForJob` polls at a fixed 2s interval and ignores `Retry-After`.** The service sends the header on a running status, but threading a response header out of `request`'s return value for a value that is `DEFAULT_POLL_MS` by construction was not worth the shape change. Marked `ponytail:` in the source; read the header if a long fit ever makes a two-second poll wasteful.
- **The cache is unbounded and never expires.** It is bounded in practice by the number of distinct report paths, which is about a dozen per run id. A tab left open across many run switches accumulates one entry per path per run. A reload clears it; an LRU would be the fix if that ever mattered.
- **The cache is keyed by full path, including the query string.** So `/api/metrics` and `/api/metrics?run_id=X` are separate entries even when `X` is the active run. Correct but slightly wasteful on the first switch to the run that was already loaded.
- **No request coalescing across different paths.** Ten reports at boot are ten requests. HTTP/2 multiplexing handles this at the transport layer; a batch endpoint would be a server change for a saving that is not measurable over localhost.
- **`getHealth` swallowing every error means a genuine bug in `/readyz` reads as "not reachable".** Deliberate - the badge must render something - but it does mean the one place to look for the real reason is the server log, not the page.
