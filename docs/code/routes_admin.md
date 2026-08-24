# `src/risk_score/api/routes_admin.py`

## Purpose

The three routes that change what the server will do next, and nothing else.

The split from `routes_public.py` is the same argument from the other side: the interesting question about this service is what an unauthenticated caller can reach, and that question is answered by reading one file.
This is the other one - the whole mutating surface, in one place, short enough to audit in a sitting.

The feature is genuinely dangerous and is treated that way.
`POST /api/runs` is remote-triggered execution over caller-supplied data that ends by writing a pickle the service will later load.
Four independent things stand between a request and that:

- `RISKSCORE_ALLOW_UPLOAD` and `RISKSCORE_ALLOW_RETRAIN`, both off by default, checked per route;
- `X-API-Key`, and `Settings` refuses to start at all if a mutating route is enabled on a non-loopback bind with no key;
- a `dataset_id` rather than a path, so a caller names content that was already stored rather than any file the process can read;
- the fit itself in a separate, killable process with one slot.

## Public API

| Route | What it does |
| --- | --- |
| `POST /api/datasets` | Stream a CSV to a content-addressed file. 201 with its id. |
| `POST /api/runs` | Start a retrain on a stored dataset. 202 with a job id. |
| `GET /api/jobs/{job_id}` | One retrain's status, including why it failed. |

| Name | What it is |
| --- | --- |
| `router` | The three routes. Imported by `app.py`. |
| `build_runner` | The `JobRunner` `create_app` installs. Here so the wiring lives with the routes. |
| `require_upload`, `require_retrain`, `get_runner` | The dependencies. |
| `UPLOAD_GUARDS`, `RETRAIN_GUARDS` | The guard pairs, applied per route because the order matters. |

## Inputs and outputs

Reads the raw request stream for an upload, a `RetrainIn` body for a retrain, a path parameter for a status.
Writes exactly one thing: a content-addressed CSV under `settings.datasets_dir`.
Everything else it does is ask `JobRunner` to start a process.

The response bodies are `DatasetOut`, `JobOut`, and `JobOut` - all declared in [schemas.md](schemas.md).

## Invariants and failure modes

**The flag is checked before the key.**
`UPLOAD_GUARDS` and `RETRAIN_GUARDS` are applied per route rather than to the whole router precisely because the order is load-bearing: an operator who has not enabled the feature is told that, rather than being asked for a credential that would not help.
`test_the_flag_is_checked_before_the_key` is what keeps the list from being reordered.

**403 for a switched-off route, not 404.**
The routes appear in `/docs` regardless - they are part of the service's contract - so pretending they do not exist would be a lie a client can see through in one request.
The detail names the environment variable, because "not found" for a documented route is the least actionable answer an API can give somebody following the runbook.

**An enabled route with no key configured is 503, not 401.**
That is `require_api_key`'s rule; see [deps.md](deps.md).

**The upload takes the raw request stream, not `UploadFile` and not a multipart form.**
A multipart parser would be a third-party dependency and a second size accounting to get right, for a route whose entire body is one file.
`request.stream()` is already chunked, which is what lets the cap be enforced against bytes that have arrived rather than against a number the caller supplied.

**This route carries its own, larger cap.**
The body-size middleware's limit is 1 MiB, sized for a `/predict` body, so `create_app` maps this exact path to `max_upload_bytes` instead.
A per-path map rather than an exemption, because an uncapped route is a full disk.

**The write happens in a worker thread.**
Hashing, writing and fsyncing a 64 MiB file on the event loop would stall every concurrent `/predict`.
It is also what makes `_SyncStream` legal: `anyio.from_thread.run` schedules each chunk read back onto the loop that owns the request, and it has to be called from off the loop to do that.
Awaiting the request iterator from the thread directly would touch loop state from off the loop, which is undefined behaviour rather than an error anything reports.

**`_SyncStream` exists so `store_dataset` can stay synchronous.**
It hashes, writes and fsyncs - filesystem work an `async def` cannot overlap with anything useful - and duplicating it as a coroutine would mean two implementations of the cap, the digest, and the cleanup.
Its loop treats an empty chunk as end of body, because the ASGI iterator yields one more time before stopping and treating that as "keep going" never terminates.

**A rejected upload is a 422, not a 400.**
The request was well-formed and its body was not what the route accepts, which is the same answer a malformed applicant gets.

**Re-uploading the same bytes is a no-op that says so.**
`existing: true`, reported rather than hidden, because an operator who uploaded twice by accident should be told the second one did nothing rather than be left wondering which copy a run used.

**202, not 200, for a retrain.**
Nothing has been trained yet: the response is a receipt, and `GET /api/jobs/{job_id}` is where the answer arrives.
Returning 200 with the run's metrics would mean holding the connection open for the length of a fit, which is exactly the behaviour this replaces.

**409 with `Retry-After` when a retrain is already running.**
A refusal rather than a queue, for the memory reason [jobs.md](jobs.md) gives.
The header exists so a poller does not have to invent an interval, and the same header appears on a still-running job's status - absent once it finishes, which is itself the signal that there is nothing more to wait for.

**An unknown dataset id is a 404 naming the route that would fix it.**
`resolve_dataset` raises `KeyError` for a malformed id and an absent one alike, and both arrive here as the same answer, so a caller cannot use the response to learn which of its guesses had the right shape.

**Job status is behind the key with the route that starts a job.**
`detail` is the child's exception text - a server-side error message, and the one place this service would otherwise volunteer an internal failure to a caller.

**An unknown job and an evicted one get the same 404.**
Both mean this service cannot tell you, and the run registry is the durable record of what was trained.

**`_job_out` goes through the response model rather than returning `job.as_dict()`.**
So a field added to the dataclass cannot appear in the API without somebody deciding it should.

**`get_runner` answers 503 rather than raising an `AttributeError`.**
That only happens in a partially constructed app - a test building one by hand, in practice - and a 503 is a better report of it than a 500.

**422 is spelled numerically.**
Starlette renamed the constant, and importing it by name from `app.py` would be circular anyway, since `app` imports this module.

## What must NOT live here

- **Anything read-only.** It belongs in `routes_public.py`, so the exposure question keeps its one-file answer.
- **The fit.** `jobs.py` owns the process, the timeout, and the slot; this file submits.
- **The upload's storage rules.** The digest, the cap, the sniff and the atomic rename are `store_dataset`'s, so they are testable without a client - and `tests/test_jobs.py` tests them that way.
- **A second way to name a dataset.** No path parameter, no filename, no CSV body. Content only.
- **Deciding whether the feature is safe to enable.** `Settings` refuses the genuinely unsafe combinations at boot; this file reports the flag's state.

## Related tests

`tests/test_routes_admin.py`, twenty tests, in four sections that follow the guard order.

- `test_a_disabled_feature_is_403_naming_its_variable`, `test_the_flag_is_checked_before_the_key`, `test_an_enabled_route_still_needs_the_key`, and `test_an_enabled_route_with_no_key_configured_is_503` are the four-way gate, and they run before anything else in the file for a reason.
- `test_an_upload_is_stored_by_content` and `test_re_uploading_the_same_bytes_says_so` are the content-addressing contract; `test_a_chunked_upload_is_stored_whole` is the `_SyncStream` adapter, which is the part most likely to be subtly wrong.
- `test_an_upload_over_the_cap_is_413` and `test_a_predict_body_is_still_capped_at_the_smaller_limit` are the two halves of the per-path limit map - the second is the one that fails if the exemption ever becomes a prefix match.
- `test_a_retrain_returns_202_with_a_job_id` and `test_a_second_retrain_is_409_with_a_retry_after` are the receipt and the refusal.
- `test_retrain_offers_every_supported_model` keeps `RetrainIn`'s `Literal` in step with `SUPPORTED_MODEL_TYPES`, which a type annotation cannot do on its own.
- `test_the_admin_routes_are_documented_even_when_disabled` is the 403-not-404 policy, asserted from the OpenAPI document.
- `test_the_job_body_has_exactly_the_documented_keys` is what makes `_job_out`'s indirection worth having.
- `test_a_real_retrain_publishes_a_run_and_the_service_swaps_to_it` is the end-to-end proof: a real child process, a real fit, and a service scoring with the new bundle afterwards.

## Known limits

- **One shared key for all three routes.** A caller permitted to upload is permitted to retrain. See [deps.md](deps.md).
- **No job cancellation.** The timeout is the only way a running fit stops early.
- **`Retry-After` is a fixed 5 seconds.** It does not estimate remaining time, because nothing here knows it - a fit reports no progress ([jobs.md](jobs.md)).
- **Uploads accumulate.** Nothing removes stored datasets, and nothing here limits how many a caller may store beyond the per-file cap.
- **`_SyncStream` concatenates into one buffer.** For a 64 MiB upload read in 1 MiB chunks that is fine because `store_dataset` drains it every chunk; a caller sending a single enormous chunk would hold it whole, which the declared-length check in front already refuses.
- **The retrain uses the training defaults.** `RetrainIn` exposes the model type and the feature tier and nothing else, deliberately: a route that accepted arbitrary hyperparameters would be a much wider execution surface for a much narrower benefit.
