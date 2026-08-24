# `src/risk_score/artifacts.py`

## Purpose

Turn a fitted model into something that can be served, named, listed, rolled back, and thrown away.

The old pipeline called `joblib.dump` on a fitted estimator and nothing ever loaded it.
That file was not merely unused, it was unusable: scoring one applicant needs four things and the pickle held one.

- The **fitted pipeline**, which is preprocessing plus estimator.
- The **calibrator**, without which the probability is not the number the cost matrix was applied to.
- The **decision threshold**, which lived in a separate JSON file and could therefore be paired with the wrong pickle by anyone who loaded them one at a time.
- The **`FeatureSpec`**, without which a request payload cannot be checked against the columns the model was actually fitted on.

`ScoringBundle` holds all four and is the only object this project pickles.

The second problem was the layout around it.
`reports/models/logistic_regression.joblib` was overwritten by the next run, `reports/metrics/metrics.json` was overwritten separately, and the calibration figure was overwritten separately again.
Three writes with no ordering guarantee means a dashboard polling that tree can read new metrics against an old calibration curve and present the pair as one result.
Worse, there was no way to answer "what produced this number" - no dataset digest, no commit, no row counts - and no way back to yesterday's model except retraining.

A run is now an immutable directory under an id that says what it is, published by a single rename.

## Public API

| Name | What it is |
| --- | --- |
| `ScoringBundle` | The frozen four-part artifact: `pipeline`, `calibrator`, `threshold`, `feature_spec`, `shap_background`, `metadata`. |
| `RunMetadata` | Typed identity and provenance: run id, commit, dataset digest, row counts, split windows, embargo counts, cost matrix, library versions. |
| `save_bundle` / `load_bundle` | Write and read the one pickle. `load_bundle` refuses a schema-version mismatch. |
| `read_manifest` | The same metadata as `load_bundle().metadata`, without unpickling anything. |
| `build_run_id` | `20260824T101530123Z-logistic_regression-origination_only-02e3a94`. |
| `feature_tier` | `"origination_only"` or `"with_lender_priced"`, the tier name that appears in ids and manifests. |
| `dataset_fingerprint` | Streaming sha256 prefix of the extract, so a run records which file it read. |
| `git_commit` / `library_versions` | Provenance that survives the machine it was produced on. |
| `staged_run` | Context manager: a scratch directory that becomes a run directory atomically, or is removed. |
| `prune_staging` | Drop scratch directories left behind by crashed runs. |
| `register_run` | Append to `registry.json`, and optionally point `active_run.json` at the new run. |
| `read_registry` / `rebuild_registry` | Read the index; reconstruct it from the manifests when it is lost. |
| `set_active_run` / `read_active_run_id` / `load_active_bundle` | The serving pointer, and the one call the service makes at boot. |
| `prune_runs` | Retention: keep the newest N, never delete the active run. |
| `now_iso` | One spelling of a timestamp, millisecond resolution, `Z` suffix. |
| `HEADLINE_METRICS` | The three metrics lifted into the registry so a run listing renders from one file. |
| `BUNDLE_SCHEMA_VERSION`, `DEFAULT_RETENTION`, the filename constants | The artifact contract, named once. |

## Inputs and outputs

This module owns the layout of `reports/`:

```
reports/
  registry.json                 append-only index of every run
  active_run.json               the only thing the service reads at boot
  runs/<run_id>/
    model.joblib                ONE ScoringBundle
    manifest.json               the same metadata as plain JSON
    metrics.json  calibration_*.csv  threshold_costs_validation.csv
    figures/*.png
    run.log
  runs/.staging-<uuid>/         a run being written; never a complete run
```

It does not know what a metric is, how a model is fitted, or what a good threshold is.
It takes objects and writes bytes.
The only asymmetry is `HEADLINE_METRICS`, which names three metric keys so that a run listing does not have to open N metrics files; the values still arrive from the caller.

The JSON files are a deliberate duplication of what is inside the pickle.
Identity questions - which dataset, which commit, which windows, how many rows - are then answerable with `cat`, from any language, without unpickling and therefore without importing this package or trusting the file's contents.

## Invariants and failure modes

**A run directory is either complete or absent.**
Everything is written into `runs/.staging-<uuid>/`, fsynced, and moved into place with `os.replace`, which is atomic within a filesystem.
Files *and* their parent directories are fsynced, because a rename that survives a crash while the file contents do not produces a run directory full of zero-length files - which is worse than no run at all, since it looks like one.
A crashed run therefore leaves a `.staging-` directory, which sorts and globs apart from real run ids because those start with a digit, and `prune_staging` drops them after an hour.

**A run id is never reused.**
`staged_run` raises `FileExistsError` rather than overwriting, so the immutability claim is enforced rather than assumed.
Run ids carry milliseconds for this reason: with the parquet cache warm, two runs of the same model and tier inside one second is ordinary rather than pathological.

**The threshold travels inside the pickle.**
There is no supported way to load a model without the decision rule it was published with.

**`load_bundle` refuses a version mismatch.**
A bundle written by different code fails at load with a message naming both versions, rather than unpickling into an `AttributeError` three calls later at which point the traceback points at the API and not at the stale artifact.

**Registry updates are serialized across processes.**
`registry.json` and `active_run.json` are shared mutable state, so both are written temp-then-`os.replace` under an `O_CREAT | O_EXCL` lock.
The wait timeout (10s) is strictly shorter than the stale timeout (60s), and that ordering is the whole design: if they were equal, a lock held legitimately for slightly too long would be *stolen* rather than waited out, and the timeout branch would be unreachable.
A stale timeout is nonetheless mandatory, because the process that would have released the lock is exactly the one that died, and without it the first `SIGKILL` wedges every later run.

**A corrupt registry is an empty registry, not an outage.**
`read_registry` returns `[]` for a missing or truncated file rather than raising, because the registry is an *index* - the run directories are the data - and refusing to list runs over one bad JSON file turns a cosmetic problem into a service that will not boot.
`rebuild_registry` reconstructs it from the manifests.

**Retention never deletes the active run.**
The alternative is a retention policy taking the service down.
Deletion is best-effort per directory, so a locked file in one run does not stop the others being cleaned.

**`RunMetadata.from_dict` ignores keys it does not know.**
So a manifest written by a *newer* version of this project can still be listed by `riskscore runs`.
Loading the pickle is where a version mismatch has to be fatal; reading identity out of JSON is not.

## What must NOT live here

- **Fitting, scoring, or metric computation.** This module must be importable by a serving process that never trains anything.
- **Deciding *what* to record.** The pipeline chooses the row counts and the metrics; this module chooses how they are stored.
- **Rendering.** The model card, the comparison table, and the dashboard payloads belong in `reporting.py`. A JSON file written for machines and a Markdown file written for humans have different lifetimes.
- **HTTP.** Path allowlisting and containment checks for `/artifacts/{run_id}/{name}` belong with the route, because they are about what a request may ask for, not about how a run is stored.
- **A second serialization format.** One pickle. Adding "also as ONNX" doubles the number of things that can disagree about what the model is.

## Related tests

`tests/test_artifacts.py`.
The atomicity and locking claims are tested by interruption rather than by inspection: a fit that raises inside `staged_run` must leave no directory, and a lock held by a live holder must produce a `TimeoutError` while one older than the stale timeout must be broken.

Worth knowing about:

- `test_the_run_id_sorts_chronologically_and_names_the_run` pins the id format and asserts the millisecond field is zero-padded, because `...30099Z` sorting after `...30100Z` would quietly break "which run is newest".
- `test_a_run_becomes_visible_only_when_it_is_complete` and `test_a_failed_run_leaves_nothing_that_could_be_mistaken_for_a_result` are the atomicity proof, from both sides.
- `test_a_corrupt_registry_reads_as_empty_rather_than_raising` and `test_the_registry_can_be_rebuilt_from_the_manifests_on_disk` cover the index-versus-data distinction.
- `test_a_stale_lock_is_broken_rather_than_wedging_every_later_run` and `test_a_lock_held_by_someone_else_is_waited_out_and_then_reported` are the two halves of the wait-versus-stale ordering; if the two timeouts were made equal, the second would fail.
- `test_retention_never_deletes_the_active_run` is the one deletion rule that matters.
- `test_scoring_goes_through_the_calibrator_not_the_bare_pipeline` is why the bundle holds both.
- `tests/test_cli.py::test_activate_refuses_a_run_whose_bundle_cannot_be_loaded` is the end-to-end version of the version-check claim: a run that cannot be served is refused when it is activated, not at the service's next boot.

## Known limits

- **The `O_CREAT | O_EXCL` lock is not correct on NFS,** where `O_EXCL` is not reliably atomic. Documented rather than papered over, because the alternative is a lock service this project has no use for. A single-host deployment, which is what the runbook describes, is safe.
- **Pickle couples every bundle to the module path of everything inside it.** Moving `risk_score.transformers` breaks every bundle ever written, silently at import time. That is the standing cost of putting feature engineering inside the pipeline (ADR 0002); the mitigation is a rule, not code, and the rule is that these module paths do not move.
- **Pickle is not a trust boundary.** `load_bundle` executes whatever it is given. The service loads bundles it produced itself, from a path in its own configuration, and never from a request.
- **`dataset_fingerprint` reads the whole file.** For a 1.19 GB extract that is a real cost, which is why the pipeline computes it once and hands the same string to the cache key and to the manifest.
- **The registry grows without bound** while run *directories* are pruned, so an entry can name a run that no longer exists. Deliberate: the index is the history, and losing the record of a run because its figures were cleaned up would defeat the point.
- **No signing or checksums on the bundle itself.** A run directory that is edited in place is indistinguishable from one that is not.
