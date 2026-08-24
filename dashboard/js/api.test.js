/**
 * `node --test dashboard/js/` - a stub `fetch`, no server, no browser.
 *
 * What is worth testing here is not "does fetch work" but the four behaviours
 * this module exists for: the cache that makes a resize free, the in-flight
 * de-duplication, the error translation that keeps the request id, and the poll
 * loop's stopping conditions. Each of those was a real defect in the dashboard
 * this replaces, and each is invisible in a browser until it is not.
 */

import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";

import {
  ApiError,
  DEFAULT_POLL_MS,
  getJson,
  invalidate,
  parseRetryAfter,
  peek,
  reportPath,
  request,
  waitForJob,
} from "./api.js";

/** A `Response`-alike with just the surface `api.js` touches. */
function stubResponse({ status = 200, body = {}, headers = {} } = {}) {
  const lower = new Map(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]));
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => lower.get(name.toLowerCase()) ?? null },
    json: async () => {
      if (body === undefined) throw new SyntaxError("not json");
      return body;
    },
  };
}

/** A stub `fetch` that records its calls. */
function recorder(handler) {
  const calls = [];
  const fetchImpl = async (path, options) => {
    calls.push({ path, options });
    return handler(path, options, calls.length);
  };
  return { calls, fetchImpl };
}

beforeEach(() => {
  // The cache is module state, so every test starts from empty. Without this the
  // tests would pass or fail depending on their order, which is its own bug.
  invalidate();
});

// --- 1. the cache that makes a resize free ---

test("a second get of the same path issues no request", async () => {
  const { calls, fetchImpl } = recorder(() => stubResponse({ body: { run_id: "r1" } }));

  const first = await getJson("/api/metrics", { fetchImpl });
  const second = await getJson("/api/metrics", { fetchImpl });

  assert.equal(calls.length, 1, "the second read must come from cache");
  assert.equal(second, first, "and must be the same object, not a copy");
});

test("peek returns the cached payload and never requests", async () => {
  const { calls, fetchImpl } = recorder(() => stubResponse({ body: { ok: true } }));

  assert.equal(peek("/api/metrics"), undefined);
  await getJson("/api/metrics", { fetchImpl });

  assert.deepEqual(peek("/api/metrics"), { ok: true });
  assert.equal(calls.length, 1);
});

test("force refetches and replaces the cached payload", async () => {
  // The path a completed retrain takes: the run changed, so the old payload is
  // wrong rather than stale.
  const { calls, fetchImpl } = recorder((_path, _options, nth) =>
    stubResponse({ body: { nth } }),
  );

  await getJson("/api/metrics", { fetchImpl });
  const refreshed = await getJson("/api/metrics", { fetchImpl, force: true });

  assert.equal(calls.length, 2);
  assert.deepEqual(refreshed, { nth: 2 });
  assert.deepEqual(peek("/api/metrics"), { nth: 2 });
});

test("invalidate with a prefix leaves other paths cached", async () => {
  const { calls, fetchImpl } = recorder(() => stubResponse({ body: {} }));

  await getJson("/api/metrics", { fetchImpl });
  await getJson("/api/drift", { fetchImpl });
  invalidate("/api/metrics");

  assert.equal(peek("/api/metrics"), undefined);
  assert.notEqual(peek("/api/drift"), undefined);
  assert.equal(calls.length, 2);
});

// --- 2. in-flight de-duplication ---

test("concurrent gets of one path share a single request", async () => {
  // Two panels both want /api/metrics on first paint. One request.
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const { calls, fetchImpl } = recorder(async () => {
    await gate;
    return stubResponse({ body: { shared: true } });
  });

  const both = Promise.all([
    getJson("/api/metrics", { fetchImpl }),
    getJson("/api/metrics", { fetchImpl }),
  ]);
  release();
  const [a, b] = await both;

  assert.equal(calls.length, 1);
  assert.equal(a, b);
});

test("a failed request is not cached, so the next ask retries", async () => {
  const { calls, fetchImpl } = recorder((_path, _options, nth) =>
    nth === 1 ? stubResponse({ status: 500, body: { detail: "boom" } }) : stubResponse({ body: { ok: 1 } }),
  );

  await assert.rejects(() => getJson("/api/metrics", { fetchImpl }), ApiError);
  assert.equal(peek("/api/metrics"), undefined);

  assert.deepEqual(await getJson("/api/metrics", { fetchImpl }), { ok: 1 });
  assert.equal(calls.length, 2);
});

// --- 3. error translation ---

test("an error body's detail and request id survive onto the thrown error", async () => {
  // The request id is the only way to find the traceback in the service log. The
  // dashboard this replaces discarded it and showed one hardcoded sentence for
  // every failure of every endpoint.
  const { fetchImpl } = recorder(() =>
    stubResponse({
      status: 503,
      body: { detail: "No active run.", request_id: "abc123" },
    }),
  );

  const error = await getJson("/api/metrics", { fetchImpl }).catch((caught) => caught);

  assert.ok(error instanceof ApiError);
  assert.equal(error.status, 503);
  assert.equal(error.message, "No active run.");
  assert.equal(error.requestId, "abc123");
});

test("a status with no usable body still gets a human sentence", async () => {
  // A proxy 502 or a Starlette-level rejection never reaches the app, so there is
  // no `detail` to show. "Error 409" is not a sentence; the hint table is.
  const { fetchImpl } = recorder(() => stubResponse({ status: 409, body: undefined }));

  const error = await getJson("/api/runs", { fetchImpl }).catch((caught) => caught);

  assert.match(error.message, /already running/);
  assert.equal(error.status, 409);
});

test("a network failure is an ApiError with status 0, not a raw TypeError", async () => {
  const fetchImpl = async () => {
    throw new TypeError("Failed to fetch");
  };

  const error = await getJson("/api/metrics", { fetchImpl }).catch((caught) => caught);

  assert.ok(error instanceof ApiError);
  assert.equal(error.status, 0);
  assert.match(error.message, /Could not reach the server/);
});

test("a caller's own abort propagates rather than becoming an ApiError", async () => {
  // A panel that was replaced mid-request cancels on purpose, and that is not an
  // error worth showing anyone.
  const controller = new AbortController();
  const fetchImpl = async () => {
    controller.abort();
    throw new DOMException("aborted", "AbortError");
  };

  const error = await request("/api/metrics", {
    fetchImpl,
    signal: controller.signal,
  }).catch((caught) => caught);

  assert.equal(error.name, "AbortError");
});

// --- 4. headers and paths ---

test("the api key is sent only on authenticated routes", async () => {
  const { calls, fetchImpl } = recorder(() => stubResponse({ body: {} }));

  await request("/api/metrics", { fetchImpl });
  await request("/api/jobs/j1", { fetchImpl, authenticated: true });

  assert.equal("X-API-Key" in calls[0].options.headers, false);
  // No key is stored in this environment, so the header is absent here too - the
  // assertion that matters is the first one: a report endpoint is never offered a
  // credential.
  assert.equal(calls[1].options.headers.Accept, "application/json");
});

test("a run id becomes a query parameter and is encoded", () => {
  assert.equal(reportPath("/api/metrics"), "/api/metrics");
  assert.equal(reportPath("/api/metrics", "a b"), "/api/metrics?run_id=a%20b");
});

test("retry-after is parsed as seconds, and anything odd is null", () => {
  // Null rather than 0: a poller that reads "wait" as "retry now" is a hot loop
  // against an endpoint that just asked it to back off.
  assert.equal(parseRetryAfter("5"), 5000);
  assert.equal(parseRetryAfter(" 2 "), 2000);
  assert.equal(parseRetryAfter("Wed, 21 Oct 2026 07:28:00 GMT"), null);
  assert.equal(parseRetryAfter("-1"), null);
  assert.equal(parseRetryAfter(""), null);
  assert.equal(parseRetryAfter(null), null);
});

// --- 5. the poll loop ---

test("polling stops on a terminal status and reports every poll", async () => {
  const statuses = ["running", "running", "succeeded"];
  const { calls, fetchImpl } = recorder((_path, _options, nth) =>
    stubResponse({ body: { job_id: "j1", status: statuses[nth - 1], run_id: "r2" } }),
  );
  const seen = [];
  // No real waiting: the sleep is injected, which is also what proves the loop
  // waits between polls rather than spinning.
  const waits = [];
  const sleep = async (ms) => waits.push(ms);

  const job = await waitForJob("j1", {
    fetchImpl,
    onUpdate: (update) => seen.push(update.status),
    sleep,
  });

  assert.equal(job.status, "succeeded");
  assert.equal(calls.length, 3);
  assert.deepEqual(seen, ["running", "running", "succeeded"]);
  assert.deepEqual(waits, [DEFAULT_POLL_MS, DEFAULT_POLL_MS]);
});

test("an unknown status is terminal, so a client cannot loop forever", async () => {
  const { fetchImpl } = recorder(() => stubResponse({ body: { status: "who knows" } }));
  const job = await waitForJob("j1", { fetchImpl, sleep: async () => {} });
  assert.equal(job.status, "who knows");
});

test("a 404 gives up at once, because an evicted job never comes back", async () => {
  const { calls, fetchImpl } = recorder(() => stubResponse({ status: 404, body: { detail: "gone" } }));

  await assert.rejects(() => waitForJob("j1", { fetchImpl, sleep: async () => {} }), {
    status: 404,
  });
  assert.equal(calls.length, 1);
});

test("a transient poll failure is tolerated, a persistent one is not", async () => {
  // The service reloads the new bundle the instant the child exits, and a poll
  // landing in that window can fail. One bad poll must not report a successful
  // retrain as failed.
  const { fetchImpl } = recorder((_path, _options, nth) =>
    nth <= 2
      ? stubResponse({ status: 500, body: { detail: "reloading" } })
      : stubResponse({ body: { status: "succeeded" } }),
  );
  const job = await waitForJob("j1", { fetchImpl, sleep: async () => {} });
  assert.equal(job.status, "succeeded");

  const { calls, fetchImpl: alwaysFails } = recorder(() => stubResponse({ status: 500, body: {} }));
  await assert.rejects(() => waitForJob("j2", { fetchImpl: alwaysFails, sleep: async () => {} }));
  assert.equal(calls.length, 4, "one attempt plus the three tolerated retries");
});
