# `src/risk_score/api/routes_public.py`

## Purpose

Every route that only reads: scoring, model identity, the seven reports, run history, and the two health probes.

Read-only in the sense that matters for a threat model - nothing here writes to disk, starts a process, or changes what the next request will see.
That is what makes the file boundary worth having: a reviewer asking "what can an unauthenticated caller reach" reads this file and is done.

Scoring is read-only despite being a POST.
The verb is POST because the request body is an applicant's financial details, and a GET would put them in the query string, which proxies log and browsers keep in history.

## Public API

| Route | What it does |
| --- | --- |
| `POST /predict` | One applicant to a calibrated probability, a decision, and reason codes. |
| `POST /predict/batch` | Up to `max_batch_rows` applicants in one vectorized pass, errors inline. |
| `GET /api/model` | The active run's identity. |
| `GET /api/schema` | The input contract, in a shape a form generator can consume. |
| `GET /api/metrics`, `/api/calibration`, `/api/threshold-costs`, `/api/vintages`, `/api/drift`, `/api/shap-summary` | The six per-run reports. Each takes `?run_id=`. |
| `GET /api/comparison` | LR against XGBoost. The one report with no `?run_id=`. |
| `GET /api/runs`, `/api/runs/{id}`, `/api/runs/{id}/card` | Run history, one manifest, one model card. |
| `GET /artifacts/{run_id}/{name:path}` | One allowlisted file from one run directory. |
| `GET /healthz`, `/readyz` | Liveness and readiness, deliberately different. |

`router` is the only name `app.py` imports.

## Inputs and outputs

Takes validated bodies and query parameters; returns the response models in [schemas.md](schemas.md), or a pre-serialized `Response` for the report routes.

The report and artifact handlers are deliberately thin - each is a name, a docstring, and one call into [reports.md](reports.md).
Caching, ETag revalidation and path containment are the same problem for all of them, and ten copies of that logic is ten places for one to be missing a guard.

## Invariants and failure modes

**The predict bodies are declared free-form and validated inside the handler.**
The accepted fields come from the loaded bundle and a retrain can change them, so they cannot be a static annotation.
`/docs` still lists them, because `app.py` patches the generated schema into the OpenAPI document.
`_validated` re-raises pydantic's `ValidationError` as `RequestValidationError`, so an invalid applicant is reported in the same shape as any other 422 rather than escaping as a 500 from an uncaught error.

**`model_dump` returns field names, not aliases.**
That is the step that turns a caller posting `funded_amnt` into the `loan_amnt` the pipeline declares, and it is why alias support costs nothing downstream.

**`async def` with synchronous work inside is correct here.**
Scoring is single-digit milliseconds of CPU, so handing it to the threadpool would cost more in context switches than it saves - and the GIL makes the threadpool no more parallel than the event loop for numpy-bound work anyway.
A retrain is the opposite case, and it runs in a process pool; see [jobs.md](jobs.md).

**The batch is validated in one pass and scored in another.**
Validating inside the scoring loop would mean 1000 rows make 1000 passes through the sklearn pipeline, which is roughly two orders of magnitude slower than one pass over 1000 rows.
Invalid rows are recorded by index and the valid ones are scored together, then the results are re-interleaved in the caller's order - so row 407 failing does not shift the indices of rows 408 onwards.

**A row's error names the field and never quotes the value.**
`_first_message` reports one problem, because a batch row's error is read in a table and the field plus the reason is enough to fix it.
A batch is the request most likely to be logged whole, which is why the value is never in there.

**Over-large batches are a 422 in the standard shape.**
Raised as a `RequestValidationError` with a synthesized `loc`, rather than an `HTTPException`, so the limit failure looks like every other body failure to a client.

**Batch `latency_ms` is summed from the per-row figures**, which already divide one vectorized call across the rows, so the total is the server-side cost of the request rather than a number multiplied by the row count.

**There is no test-set threshold cost curve, and there is not going to be one.**
A cost curve over test scores is exactly the artifact that would let somebody pick a threshold on test by eye, which is the leak the tri-split exists to prevent ([0003](../decisions/0003-train-validation-test-split.md)).
The route serves validation only and says so.

**`/api/comparison` takes no `run_id`, and 404s until a comparison has been run.**
"Nothing has been compared" and "the two models scored the same" are different answers, and a client cannot tell them apart from `{}`.

**`/api/runs` is one file read, not one per run.**
The registry carries each run's headline metrics as well as its identity, precisely so a history table does not cost twenty manifest reads per page load.
Reversal happens here rather than in `read_registry`, which is append-ordered because that is what an append-only index is - newest-first is a presentation choice and belongs at the presentation edge.

**A manifest is served as its own bytes, not as a model built from it.**
A manifest is the record of what a run did, and re-serializing it through a schema written later is how a field silently stops being reported.

**A model card is served as markdown, not rendered HTML.**
It is a document to read, commit, or attach to a review, and rendering it server-side would put an HTML escaping problem inside a service whose job is arithmetic.

**`{name:path}` accepting a separator is safe *because* the allowlist is exact-match.**
One allowed entry is `figures/calibration_test.png`.
No string outside those dozen filenames reaches the filesystem, there is no directory listing, and `model.joblib` is not on the list.

**A run id is length-bounded in the router**, so an absurd id is refused before any handler runs, and then pattern-checked and containment-checked in `reports.py`.
In the path where it identifies a resource; in the query where it selects a view of the active one.

**`/healthz` and `/readyz` are separate, and `/healthz` is 200 with no model.**
A liveness probe that fails when the model is missing makes an orchestrator restart a process that would have come up in exactly the same state, forever.
`/readyz` is the one the container's `HEALTHCHECK` uses, and it 503s through the shared dependency so a load balancer stops sending traffic to a replica whose bundle failed to load.
`/healthz` still reports `status: "degraded"` and the load error, so the state is visible rather than merely survivable.

**Report responses declare no pydantic model.**
Their columns come from whatever the run wrote, so a response model would be a fourth copy of a schema that changes per run.
`_REPORT_RESPONSES` documents the shape and the 304 in `/docs` instead.

## What must NOT live here

- **Anything that writes.** That is the file boundary, and it is the whole reason a reviewer can read one file to answer the exposure question. Uploads and retrains are in `routes_admin.py`.
- **Caching, ETags, or path resolution.** `reports.py` owns them once.
- **Scoring logic.** `ScoringService` owns it, so it is testable without a client.
- **Reading the environment or the filesystem directly.** Handlers take `SettingsDep`, and only `list_runs` and `healthz` touch app state beyond a dependency.
- **A key check.** If a route here needed one, it would belong in the other file.

## Related tests

`tests/test_api.py`, sixty-six tests, organized in the same order as this file's sections.

- `test_batch_matches_single_scoring` ties the two scoring routes together, so the vectorized path cannot quietly become a second implementation.
- `test_batch_reports_bad_rows_inline` and `test_batch_refuses_more_rows_than_configured` are the batch contract; the first also pins index preservation.
- `test_every_train_report_is_served` is parametrized over `REPORTS`, so adding a report without a test is not possible.
- `test_report_run_id_never_escapes_the_runs_directory`, `test_the_model_pickle_is_not_served`, and `test_an_artifact_name_outside_the_allowlist_is_404` are the exposure guards.
- `test_healthz_is_ok_and_readyz_is_503_with_no_bundle` is the split-probe rule, and the reason both routes exist.
- `test_run_history_is_empty_rather_than_an_error_before_any_run` covers the fresh-clone case that `/api/runs` has to survive.
- `test_manifest_is_served_verbatim` and `test_model_card_is_served_as_markdown` pin the two pass-through routes, including the content type.
- `test_no_error_body_leaks_a_path` covers every route here at once, which is the point of having one error shape.

## Known limits

- **No pagination on `/api/runs`.** The retention ceiling is twenty runs, so the list is bounded by the writer rather than by the reader. A registry that grew would need a cursor.
- **No streaming for a large batch.** The whole response is built in memory before the first byte is sent; `max_batch_rows` is what bounds it.
- **Reason codes on a batch are synchronous and uncapped by anything but the row limit.** `explain=true` on 1000 rows is 1000 explanations in one request.
- **No conditional requests on `/predict`.** Every score is computed fresh, deliberately: a cached decision is a decision that survives a retrain.
- **The health payload exposes the load error to an unauthenticated caller.** It is a short type-and-message string produced by our own loader rather than a traceback, and the alternative - a probe that will not say why it is not ready - costs more in operations than it buys.
