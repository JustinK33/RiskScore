# `src/risk_score/api/app.py`

## Purpose

Assemble a configured service: the factory, the middleware stack, the error shape, and the model load.

A factory rather than a module-level `app = FastAPI()`, because a module-level app is configured by import side effects.
It reads the environment at import time, so a test cannot build a second one with different settings, and the uvicorn `--reload` worker and its parent disagree about which one is real.
`create_app(settings)` takes its configuration as an argument, which is why the whole test suite can run several differently configured services in one process.

The other reason this file exists is that four cross-cutting concerns have to be ordered relative to each other - request id, body cap, host check, compression - and an ordering spread across four files is an ordering nobody can check.

## Public API

| Name | What it is |
| --- | --- |
| `create_app` | The factory. Returns an app with its bundle already loaded. |
| `load_service` | Load or reload the active bundle into `app.state`. Called at construction and after a retrain. |
| `RequestContextMiddleware` | Request id, per-path body cap, `X-Request-ID` on the way out. Raw ASGI. |
| `BodyTooLarge` | Raised from the wrapped `receive`; handled as a 413. |
| `summarize_validation_errors` | Field names and reasons for a 422, with no submitted values. |
| `GZIP_MINIMUM_BYTES`, `MAX_REPORTED_FIELDS`, `SCOPE_REQUEST_ID`, `UPLOAD_PATH` | The tuning constants. |

## Inputs and outputs

Takes a `Settings`.
Returns a `FastAPI` whose `state` holds `settings`, `service`, `applicant_model`, `load_error`, and `job_runner` - the five things every dependency in `deps.py` reads.

Reads `reports/active_run.json` and the bundle it names, once, at construction.
Nothing on the request path opens a file except the report cache, which is populated from fingerprinted reads and covered in [reports.md](reports.md).

The middleware stack, outermost first:

```
RequestContext   request id, body size cap, X-Request-ID on the way out
TrustedHost      Host header allow-list (DNS rebinding)
GZip             responses over 1 KiB
<routes>
```

`add_middleware` prepends, so the calls in `create_app` read bottom-up relative to that diagram.
The order is not cosmetic: the body cap has to sit outside anything that reads the body, and the request id has to be bound before any handler - including an error handler - can log.

## Invariants and failure modes

**The bundle is loaded before the app is returned, not in a lifespan handler.**
The pydantic model that validates an applicant is generated from the bundle's own `FeatureSpec`, so the bundle has to be in hand at construction time for `/docs` to describe the model that will actually score the request.
A failure to load is *recorded* in `load_error` rather than raised, because a fresh clone has no bundle and should still boot far enough to say so - `/readyz` returns 503 with the reason and `/healthz` still answers.
`require_bundle` inverts that for a container.

**Per-bundle state is replaced together or not at all.**
`load_service` clears the service, the applicant model, the report cache and the cached OpenAPI document in one pass.
A half-swapped state would serve scores from a new model against the old model's input contract, which is the kind of bug that produces a plausible number from the wrong feature.
Clearing `app.openapi_schema` matters because FastAPI caches the document on first request and would otherwise describe the old model forever.

**The body cap sees a stream, which is why this middleware is raw ASGI.**
`BaseHTTPMiddleware` hands a handler a fully buffered body, so the memory has already been spent by the time anything can object - the cap would only ever refuse requests it had already accepted.
A declared `Content-Length` over the limit is refused before the application is called at all, and the body is never read: a client that declared 900 MB gets one response and the connection closes rather than draining.
A chunked body with no declared length is counted as it arrives.

**A malformed `Content-Length` is a 400, not a hang.**
Checked with `isdigit` rather than `int()` in a `try`, because `int(" -1 ")` succeeds and a negative declared length is precisely the input that makes a naive `read(length)` block forever.
The `_MALFORMED` sentinel is distinct from `None` because absent is legal for a chunked body and gets a different answer.

**Two caps, keyed on exact paths.**
One number cannot serve both routes that take a body: a `/predict` body is a few hundred bytes and 1 MiB is already generous, while a dataset upload is tens of megabytes by design.
A per-path map rather than exempting the upload route, because an uncapped route is a full disk.
Exact paths rather than a prefix match, because a prefix is how an exemption ends up applying to something that was never meant to have it.

**Every response carries `X-Request-ID`, errors included.**
The id is only useful to somebody reporting a failure, and a failure is exactly when a service is least likely to have bothered to add it.

**One error shape, and it never echoes the input.**
FastAPI's default 422 body quotes the submitted value back at the caller, and a 422 that repeats an applicant's income puts that value in every proxy log between here and there.
`summarize_validation_errors` reports field paths and reasons only.
It keeps list indices, because "row 3" is the only way a batch caller finds the offending row, and it drops the leading `body` because every one of these is about the body.
Beyond `MAX_REPORTED_FIELDS` it counts rather than lists: a caller posting forty wrong fields is not helped by forty names.

**The 500 handler is the one that keeps a filesystem path out of an error body.**
The traceback goes to the log with the request id; the caller gets the id and an instruction to quote it.
Nothing else - no path, no dataset date range, no column list.

**The dashboard mount goes last.**
A mount at `/` matches any path an earlier route did not, and Starlette resolves routes in order, so mounting before the routers would shadow `/predict` with a filesystem 404.
An absent dashboard directory is skipped with an INFO log rather than failing the boot: a container that ships the API only is a supported deployment.
One server, not two, because a dashboard reading a different process's `reports/` is how a page ends up showing metrics from a run the scoring service is not using.

**The job runner is built whether or not retraining is enabled.**
It costs a lock and two empty containers until a job is submitted.
Building it on first use would let two concurrent first requests each build one, and the single retrain slot would quietly be two.

**The OpenAPI patch is best-effort.**
The predict routes declare a free-form body, because the accepted fields are not known until a bundle is loaded and a retrain can change them; declaring them statically would be a third hand-maintained copy of the column list, which is the thing [schemas.md](schemas.md) exists to avoid.
So the generated model's schema is patched in afterwards, from the same model that validates the request.
A patch failure is logged and ignored, because broken documentation is a much smaller problem than a service that will not answer `/openapi.json`.

**413 and 422 are spelled numerically.**
Starlette renamed both constants - `HTTP_422_UNPROCESSABLE_ENTITY` to `..._CONTENT`, `HTTP_413_REQUEST_ENTITY_TOO_LARGE` to `HTTP_413_CONTENT_TOO_LARGE` - and referring to either by name makes the import version-dependent for no gain.
The numbers have never moved.

## What must NOT live here

- **Route handlers.** They are in `routes_public.py` and `routes_admin.py`; this file wires them up.
- **Scoring.** `ScoringService` is constructed here and called nowhere in this file.
- **Business decisions about *whether* a route is allowed.** The flag-then-key guard order is `routes_admin.py`'s, so the guard and the route it protects are read together.
- **Reading the environment.** `Settings` does that, once.
- **A second app instance, or a module-level one.** Every consumer takes the app or its state as an argument.

## Related tests

`tests/test_api.py` sections 3, 4, 7 and 8, and the fixtures in `tests/conftest.py` that build real apps over the session bundle.

- `test_oversized_declared_body_is_refused_unread` and `test_malformed_content_length_is_400` are the two pre-application refusals; the second is the one that would otherwise hang a worker.
- `test_request_id_is_generated_and_present_on_errors` and `test_error_bodies_carry_only_detail_and_request_id` pin the response contract.
- `test_no_error_body_leaks_a_path` and `test_validation_summary_never_quotes_a_value` are the disclosure guards, and `test_validation_summary_counts_beyond_the_reported_limit` covers the truncation.
- `test_healthz_is_ok_and_readyz_is_503_with_no_bundle` and `test_require_bundle_refuses_to_start_without_one` are the two halves of the missing-bundle policy.
- `test_the_dashboard_does_not_shadow_the_api` is the mount-order regression, and `test_a_missing_dashboard_leaves_the_api_working` is the skip path.
- `test_openapi_describes_the_loaded_model` proves the patch actually reaches `/openapi.json`, and `test_docs_can_be_switched_off` covers the disabled case.
- `tests/test_routes_admin.py::test_a_real_retrain_publishes_a_run_and_the_service_swaps_to_it` exercises `load_service` as a reload rather than as a boot.

## Known limits

- **No CORS at all.** Adding it is one middleware, and it is left out because the dashboard is served from the same origin, so any CORS configuration here would exist only to let something else in.
- **`GZIP_MINIMUM_BYTES` is a single global threshold.** It does not know that a 1.2 KiB JSON payload compresses far better than a 1.2 KiB PNG. The figures are small enough that the difference is noise.
- **The request id is trusted from the header when present.** A caller can supply one, which is what makes it useful for correlation across a proxy, and also means it is not an audit identifier.
- **Reloading is whole-bundle.** There is no way to swap the calibrator or the threshold alone, by design: they are one artifact for exactly that reason ([0007](../decisions/0007-the-scoring-bundle.md)).
- **Single process.** Multiple uvicorn workers would each hold their own bundle and their own retrain slot; see [jobs.md](jobs.md).
