# `src/risk_score/api/reports.py`

## Purpose

Serve the artifacts a run wrote, without reading them per request and without letting a caller name a file the service did not intend to publish.

A published run directory holds a dozen small files: metrics, two calibration tables, the threshold cost curve, the vintage breakdown, two PSI tables, the SHAP summary.
The dashboard needs most of them on every page load.
Reading and parsing them per request would mean a page refresh costs a dozen `read_csv` calls for bytes that have not changed since the run was published.

Three decisions follow from that, and one from the fact that a run directory also contains a pickle.

## Public API

| Name | What it is |
| --- | --- |
| `report_response` | One report, with an ETag, a cache policy, and a 304 when nothing changed. |
| `artifact_response` | One allowlisted file out of one run directory, streamed. |
| `resolve_run_dir` | A run id to a directory, or the active run's, plus whether it was named explicitly. |
| `clear_cache` | Drop every cached payload. Called after a retrain publishes. |
| `REPORTS` | The seven report endpoints and the files each one reads. |
| `ARTIFACT_FILENAMES` | The twelve files that may be fetched by name. `model.joblib` is not one of them. |
| `Report`, `fingerprint`, `MEDIA_TYPES`, `RUN_ID_PATTERN`, `CACHE_ENTRIES`, `IMMUTABLE_CACHE_CONTROL`, `REVALIDATE_CACHE_CONTROL` | The supporting types and constants. |

## Inputs and outputs

Reads files under `settings.runs_dir/<run_id>/`, plus `comparison.json` at the report root.
Returns a `Response` with pre-serialized JSON bytes, or a `FileResponse` streaming one file.

Payloads are **columnar** - one array per column, via `risk_score.reporting.columnar`.
The threshold cost table is 99 rows of six numbers, and a row-per-object encoding repeats every key 99 times.
It is also the shape a chart wants, so the dashboard does no reshaping.

Every report payload except the root-scoped comparison carries a `run_id` key, so a chart can never be mislabelled by a race between two fetches.

## Invariants and failure modes

**Cached by file identity, not by clock.**
The key is `(filename, mtime_ns, size)` per file the report reads.
A republished run is picked up on the next request with no TTL to tune and no stale window.
A time-based cache has to choose between serving old metrics and re-reading files that never change; keying on identity chooses neither.
Size as well as mtime, because a filesystem with coarse timestamps can rewrite a file within one tick, and a report that changed length is exactly the case where that matters.

**Cached as serialized bytes, not as a parsed object.**
The response is the same bytes every time, so `json.dumps` runs once per version of a file rather than once per request - and that is most of the cost of these endpoints.

**The fingerprint is in the cache key, not read inside the function.**
`_serialize`'s `_version` parameter is unused in the body and load-bearing in the signature.
Reading the mtimes inside the function would defeat the entire invalidation mechanism, because `lru_cache` would return the first answer forever.

**Absent optional files change the fingerprint.**
`fingerprint` omits a file it cannot `stat`, so an optional part appearing later produces a different key and invalidates the entry.
Without that, a run whose test calibration curve was written a moment after the metrics would serve `"test": null` until the process restarted.

**A run directory is immutable, the active run is a pointer.**
`?run_id=` gets `max-age=31536000, immutable`; the active run gets `no-cache`.
A stale dashboard showing the previous model's metrics under the new model's name is the one caching bug that matters here, and a retrain moves that pointer.

**The ETag is weak.**
gzip is applied downstream of this response, so two representations of the same payload exist and a strong validator would assert byte equality that does not hold.
`If-None-Match` is compared on the opaque part, because the header may carry a list and a proxy is entitled to have added `W/` or stripped it.

**An allowlist, not "any file in the run directory".**
`ARTIFACT_FILENAMES` is twelve exact strings.
The run directory also holds `model.joblib`, and a pickle offered over HTTP is an invitation to unpickle something a stranger chose.
The run log *is* included, deliberately - it is the run's own record and contains no request data - and so is the calibration figure, because the dashboard renders it.

**Three guards on an artifact path, and the first does almost all the work.**
An exact-match name check, so no input outside those twelve strings reaches the filesystem at all; then the run id's pattern-plus-containment check; then containment again on the joined path.
The third is not redundant: `figures/` means one allowlisted entry legitimately contains a separator, and that is precisely the shape a traversal wants to borrow.

**Two independent guards on a run id.**
`RUN_ID_PATTERN` is conservative - `build_run_id` produces only those characters, and everything a traversal needs (`/`, `\`, `.`) is outside the class - and the resolved path is checked for containment in case a symlink inside `runs/` got there another way.
One guard on a filesystem path is one guard too few.

**A rejected id is a 404, not a 400.**
Distinguishing "malformed" from "absent" tells a scanner which of its guesses were the right shape.
The same reasoning applies to an artifact name outside the allowlist: the allowed list is in the OpenAPI document for anybody entitled to it.

**No active run is a 503; a run with no such report is a 404.**
Different conditions with different fixes - train a model, versus this particular run did not write that file.

**A missing optional part is `null`, a missing required one is a 404.**
A run whose test window held too few positives to bin writes only the validation curve.
That is a partial report, not a broken one.

**The comparison is root-scoped and revalidates.**
It is a statement about several runs, complete only once all of them are published, so no single run directory owns it.
It has no `?run_id=`, and its payload omits the key entirely rather than stamping the currently-served run onto a document about several - that would be a claim the file does not make.
Absent means 404, because "nothing has been compared" and "the two models tied" are different answers.

**`allow_nan=False`.**
There is no JSON spelling of `NaN` that a strict parser accepts, so a payload containing one has to fail here rather than produce a body the dashboard cannot parse.

**Media types are explicit, with charsets on the text types.**
A browser guessing an encoding for a model card is how a card with a non-ASCII feature name renders as mojibake.
By suffix rather than by sniffing, because the allowlist is closed: five kinds of file, and no sixth can appear without this dictionary being edited.

**Filenames come from `artifacts.py`, not from string literals here.**
They were literals, and the copy for `comparison.json` disagreed with the writer's - which made that endpoint a permanent 404 that no test caught until the filenames were centralized.

## What must NOT live here

- **Computing anything.** Every number served was computed by a run and written to a file. This module reads, serializes, and caches.
- **Serving the pickle.** Not by an allowlist entry, not by a special case.
- **Directory listing.** There is no route that enumerates a run directory; `/api/runs` lists run ids from the registry, which is a curated record rather than a filesystem walk.
- **Authorization.** Every route here is public because every payload here is a published metric. Anything requiring a key is in `routes_admin.py`.
- **The route decorators.** They are in `routes_public.py`, so the URL surface reads in one file.

## Related tests

`tests/test_api.py` sections 5 and 6, roughly twenty tests, weighted towards the guards rather than the happy path.

- `test_every_train_report_is_served` is parametrized over `REPORTS`, so a new report cannot be added without a test covering it.
- `test_report_cache_notices_a_rewritten_file` is the invalidation claim, and the reason the fingerprint is in the key.
- `test_report_etag_answers_304`, `test_report_etag_tolerates_a_proxy_rewriting_the_validator`, and `test_report_etags_differ_per_report` cover the validator; `test_active_run_revalidates_and_a_named_run_is_immutable` covers the policy.
- `test_report_run_id_never_escapes_the_runs_directory` and `test_report_refuses_a_bad_run_id` are the traversal guards, parametrized over the shapes worth trying.
- `test_the_model_pickle_is_not_served` and `test_an_artifact_name_outside_the_allowlist_is_404` are the allowlist; `test_every_allowlisted_artifact_is_served` is its other half, so the list cannot drift into denying something the dashboard needs.
- `test_report_tables_are_columnar` and `test_report_payloads_contain_no_nan_token` pin the payload shape.
- `test_comparison_is_read_from_the_report_root`, `test_comparison_ignores_a_run_id`, and `test_comparison_is_404_on_a_plain_train_run` cover the root-scoped case from all three directions.

## Known limits

- **`CACHE_ENTRIES` is a plain LRU on entry count with no byte accounting.** `ponytail:` the artifacts are kilobytes and the retention ceiling is twenty runs, so 256 entries holds a full history. If a future report is megabytes, evict on bytes instead.
- **The cache is per process.** Multiple uvicorn workers each hold their own, which is correct but means the first request to each worker pays the read.
- **`stat` per file per request.** A dozen `stat` calls is microseconds and it is what buys the no-stale-window guarantee, but it is not zero, and it is the reason these endpoints are not literally a dictionary lookup.
- **The ETag covers the fingerprint, not the bytes.** Touching a file without changing it produces a new ETag and a re-transfer. Hashing the payload instead would mean serializing it before deciding to answer 304, which is the work the 304 exists to avoid.
- **No range requests on artifacts.** `FileResponse` supports them; nothing here needs them, since the largest file served is a calibration figure.
- **A report that needs two files from *different* runs cannot be expressed.** `Report` is scoped to one directory or to the root. The comparison is why `root_scoped` exists, and a third shape would need a third mechanism.
