/**
 * The one place this dashboard talks to the service.
 *
 * Four things live here that were spread across the old `app.js`, or absent:
 *
 * **One error shape.** The service answers every failure with
 * `{detail, request_id}` and a status, so `ApiError` carries all three and every
 * caller renders `error.message`. The previous version threw
 * `new Error("Report artifacts are missing. Run the baseline pipeline first.")`
 * for *any* non-OK response from any of three endpoints, so a 401, a 503 during
 * startup, and a genuinely missing report were one message - and the request id,
 * which is the only way to find the traceback in the log, was discarded.
 *
 * **An in-memory cache of parsed payloads.** This is what makes the resize
 * handler free: `main.js` redraws from the cached object and issues no request at
 * all. The old handler refetched two endpoints per resize event and swallowed the
 * result with `.catch(() => {})`, so a resize storm was a request storm whose
 * failures were invisible. HTTP-level revalidation is left to the browser, which
 * already does it properly with the `ETag` and `Cache-Control` the service sends;
 * duplicating that in JS would be a second cache to keep correct for no gain,
 * since `fetch` never even exposes the 304.
 *
 * **In-flight de-duplication.** Two panels wanting `/api/metrics` at boot share
 * one request, keyed by URL. Without it, the first paint fires the same GET as
 * many times as there are readers of it.
 *
 * **A timeout on everything.** A `fetch` with no signal waits as long as the
 * browser feels like, and a dashboard stuck on "Loading reports" with no error is
 * indistinguishable from a broken build.
 */

/** Every request gives up after this. Long enough for a cold report render. */
const DEFAULT_TIMEOUT_MS = 15_000;

/** Where the operator's API key lives for the length of a tab. */
const KEY_STORAGE = "riskscore.apiKey";

/**
 * A failed request, with everything needed to report it.
 *
 * `requestId` is the load-bearing field: the service deliberately keeps detail
 * out of error bodies and puts the traceback in its log under this id, so a UI
 * that drops it leaves the operator with "it returned 500" and no way to join it
 * to anything.
 */
export class ApiError extends Error {
  constructor(message, { status = 0, detail = "", requestId = "", retryAfter = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.requestId = requestId;
    this.retryAfter = retryAfter;
  }
}

/**
 * Turn a response the service refused into an `ApiError` whose message is worth
 * showing a human.
 *
 * The status is translated rather than printed, because `409` means something
 * specific here (a retrain is already running) and "Error 409" tells a reader
 * nothing. The server's own `detail` wins when there is one; these are the
 * fallbacks for a response that did not come from the app - a proxy 502, say.
 */
const STATUS_HINTS = {
  401: "This action needs an API key. Set one in the retrain panel.",
  403: "This action is switched off on the server.",
  404: "Not found. The active run may not have this report.",
  409: "A retrain is already running. Wait for it to finish.",
  413: "That upload is larger than the server's limit.",
  422: "The server rejected the input.",
  500: "The server hit an internal error.",
  503: "The service has no model loaded yet. Train one with `riskscore train`.",
};

async function toApiError(response) {
  let detail = "";
  let requestId = "";
  try {
    // A non-JSON error body is entirely possible - a proxy or a Starlette-level
    // rejection that never reached the app - so parsing failure is expected and
    // must not replace the real status with a SyntaxError.
    const body = await response.json();
    if (body && typeof body === "object") {
      detail = typeof body.detail === "string" ? body.detail : "";
      requestId = typeof body.request_id === "string" ? body.request_id : "";
    }
  } catch {
    detail = "";
  }
  const hint = STATUS_HINTS[response.status] || `Request failed (${response.status}).`;
  const retryAfter = parseRetryAfter(response.headers.get("Retry-After"));
  return new ApiError(detail || hint, {
    status: response.status,
    detail,
    requestId,
    retryAfter,
  });
}

/**
 * `Retry-After` as milliseconds, or null.
 *
 * Only the delta-seconds form is honoured. The HTTP-date form is legal and this
 * service never sends it, and an unrecognised value has to be null rather than
 * zero: a poller that reads "wait" as "retry immediately" is a hot loop against
 * an endpoint that just asked it to back off.
 */
export function parseRetryAfter(headerValue) {
  if (typeof headerValue !== "string" || headerValue.trim() === "") return null;
  const seconds = Number(headerValue.trim());
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  return Math.round(seconds * 1000);
}

/** The stored API key, or `""`. */
export function getApiKey() {
  try {
    return globalThis.sessionStorage?.getItem(KEY_STORAGE) || "";
  } catch {
    // Storage can throw outright when cookies are blocked. An unusable key store
    // is not a reason to fail the whole page.
    return "";
  }
}

/**
 * Remember the key for this tab.
 *
 * `sessionStorage`, not `localStorage`: this is a credential, and it should not
 * outlive the tab that was given it. It is never put in a URL, never logged, and
 * never sent to an endpoint that did not ask for it.
 */
export function setApiKey(key) {
  try {
    if (key) globalThis.sessionStorage?.setItem(KEY_STORAGE, key);
    else globalThis.sessionStorage?.removeItem(KEY_STORAGE);
  } catch {
    /* see getApiKey */
  }
}

/**
 * One request. Everything else in this module goes through here.
 *
 * `fetchImpl` is a parameter purely so `api.test.js` can drive this with a stub
 * and assert the de-duplication and the error translation without a server.
 */
export async function request(
  path,
  {
    method = "GET",
    body = null,
    headers = {},
    signal = null,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    authenticated = false,
    fetchImpl = globalThis.fetch,
  } = {},
) {
  const requestHeaders = { Accept: "application/json", ...headers };
  if (authenticated) {
    const key = getApiKey();
    // Sent only when present and only on routes that declared they need it, so a
    // key is never volunteered to an unauthenticated report endpoint.
    if (key) requestHeaders["X-API-Key"] = key;
  }

  // `AbortSignal.any` so a caller's own cancellation and the timeout both work;
  // the timeout alone would leak a request whose panel has already been replaced.
  const timeout = AbortSignal.timeout(timeoutMs);
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;

  let response;
  try {
    response = await fetchImpl(path, {
      method,
      body,
      headers: requestHeaders,
      signal: combined,
    });
  } catch (error) {
    if (signal?.aborted) throw error; // The caller cancelled; let it through as-is.
    if (timeout.aborted) {
      throw new ApiError(`The server did not answer within ${timeoutMs / 1000}s.`, { status: 0 });
    }
    // A network-level failure has no status and no body: the server is down, the
    // page is offline, or CORS refused it. Say that rather than "undefined".
    throw new ApiError("Could not reach the server.", { status: 0, detail: String(error) });
  }

  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return null;
  return response.json();
}

/**
 * Cached GETs.
 *
 * `cache` holds settled payloads; `pending` holds in-flight promises so
 * concurrent callers share one request. A rejected request is removed from
 * `pending` and never enters `cache`, so a failure is retried on the next ask
 * rather than remembered forever.
 */
const cache = new Map();
const pending = new Map();

/** The cached payload for a path, or `undefined`. Never triggers a request. */
export function peek(path) {
  return cache.get(path);
}

/**
 * Drop cached payloads.
 *
 * With no argument, everything - which is what a completed retrain needs, since
 * every report on the page now describes the previous run. With a prefix, only
 * matching paths.
 */
export function invalidate(prefix = "") {
  for (const key of [...cache.keys()]) {
    if (!prefix || key.startsWith(prefix)) cache.delete(key);
  }
}

/**
 * GET a path, from cache when possible.
 *
 * `force` bypasses the cache for the poll-and-refresh path, and still populates
 * it, so a later redraw sees the new payload.
 */
export async function getJson(path, { signal = null, force = false, ...rest } = {}) {
  if (!force && cache.has(path)) return cache.get(path);
  const inFlight = pending.get(path);
  if (inFlight && !force) return inFlight;

  const promise = request(path, { signal, ...rest })
    .then((payload) => {
      cache.set(path, payload);
      return payload;
    })
    .finally(() => {
      // Only clear the slot if it is still ours: a `force` request started while
      // an ordinary one was in flight would otherwise delete the other's entry.
      if (pending.get(path) === promise) pending.delete(path);
    });
  pending.set(path, promise);
  return promise;
}

/** Optional `?run_id=` on the report endpoints, without hand-built strings. */
export function reportPath(name, runId = null) {
  return runId ? `${name}?run_id=${encodeURIComponent(runId)}` : name;
}

// --- the endpoints, named ------------------------------------------------------
//
// Thin wrappers rather than callers writing paths inline. The point is that a
// renamed endpoint is one edit here, and that a typo in a path is a missing
// export rather than a 404 at runtime.

export const getModel = (options) => getJson("/api/model", options);
export const getSchema = (options) => getJson("/api/schema", options);
export const getMetrics = (runId, options) => getJson(reportPath("/api/metrics", runId), options);
export const getCalibration = (runId, options) =>
  getJson(reportPath("/api/calibration", runId), options);
export const getThresholdCosts = (runId, options) =>
  getJson(reportPath("/api/threshold-costs", runId), options);
export const getVintages = (runId, options) => getJson(reportPath("/api/vintages", runId), options);
export const getDrift = (runId, options) => getJson(reportPath("/api/drift", runId), options);
export const getShapSummary = (runId, options) =>
  getJson(reportPath("/api/shap-summary", runId), options);
export const getComparison = (options) => getJson("/api/comparison", options);
export const getRuns = (options) => getJson("/api/runs", options);
export const getRun = (runId, options) => getJson(`/api/runs/${encodeURIComponent(runId)}`, options);

/**
 * Score one applicant.
 *
 * Never cached: two identical bodies are two decisions, and a scoring response
 * carries a `latency_ms` that would be a lie if it came from a `Map`.
 */
export async function predict(applicant, { explain = true, topK = 5, signal = null } = {}) {
  const query = new URLSearchParams({ explain: String(explain), top_k: String(topK) });
  return request(`/predict?${query}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(applicant),
    signal,
  });
}

/**
 * Store a CSV to train on later.
 *
 * The `File` goes in as the raw body, not as multipart: the route reads
 * `request.stream()` directly so it can enforce its size cap against bytes that
 * actually arrived. A `FormData` body here would be rejected as CSV containing a
 * multipart boundary.
 *
 * The timeout is raised because this is up to 64 MiB over whatever link the
 * operator has.
 */
export async function uploadDataset(file, { signal = null } = {}) {
  return request("/api/datasets", {
    method: "POST",
    headers: { "Content-Type": "text/csv" },
    body: file,
    signal,
    authenticated: true,
    timeoutMs: 10 * 60_000,
  });
}

/** Start a retrain. Returns the 202 receipt, not a result. */
export async function startRetrain(
  { datasetId, modelType = "logistic_regression", includeLenderPriced = false },
  { signal = null } = {},
) {
  return request("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: datasetId,
      model_type: modelType,
      include_lender_priced: includeLenderPriced,
    }),
    signal,
    authenticated: true,
  });
}

/** One poll of a job. */
export async function getJob(jobId, options = {}) {
  return request(`/api/jobs/${encodeURIComponent(jobId)}`, { ...options, authenticated: true });
}

/** How long to wait before the next poll when the server did not say. */
export const DEFAULT_POLL_MS = 2000;

/**
 * Poll a job until it stops running.
 *
 * `getJob` goes through `request` rather than `getJson`, because a cached job
 * status would report "running" forever. `onUpdate` is called for every poll
 * including the last, so the caller can show progress without its own timer.
 *
 * Terminal statuses are `succeeded`, `failed`, and `timed_out`. An unknown status
 * is treated as terminal too: a client that loops on a status it does not
 * understand never stops.
 */
const RUNNING = "running";

/** Consecutive failed polls tolerated before giving up. */
const POLL_FAILURES_ALLOWED = 3;

export async function waitForJob(
  jobId,
  { signal = null, onUpdate = null, sleep = null, ...rest } = {},
) {
  const pause = sleep || ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
  let failures = 0;
  for (;;) {
    let job;
    try {
      job = await getJob(jobId, { signal, ...rest });
      failures = 0;
    } catch (error) {
      // A single failed poll is not a failed job: the service reloads the new
      // bundle immediately after the child exits, and a poll landing in that
      // window can time out. A 404 is different - the job has been evicted from
      // the bounded history and no amount of waiting will bring it back.
      failures += 1;
      const fatal = error instanceof ApiError && error.status === 404;
      if (fatal || failures > POLL_FAILURES_ALLOWED) throw error;
      await pause(DEFAULT_POLL_MS);
      continue;
    }
    onUpdate?.(job);
    if (job.status !== RUNNING) return job;
    // The service also sends `Retry-After` on a running status, but a header is
    // not worth threading through `request`'s return value for a value that is
    // this one by construction. `ponytail:` fixed interval, read the header if a
    // long fit ever makes a two-second poll wasteful.
    await pause(DEFAULT_POLL_MS);
  }
}

/** Liveness, for the header badge. Never cached, and never throws. */
export async function getHealth({ signal = null } = {}) {
  try {
    return await request("/readyz", { signal, timeoutMs: 4000 });
  } catch (error) {
    if (error instanceof ApiError && error.detail) {
      // /readyz answers 503 with a reason when the bundle failed to load, and that
      // reason is the most useful thing on the page at that moment.
      return { status: "degraded", bundle_loaded: false, reason: error.detail };
    }
    return { status: "degraded", bundle_loaded: false, reason: "The service is not reachable." };
  }
}
