# Architecture

How data becomes a bundle, and how a bundle becomes a score.

[README.md](README.md) covers *why* the pipeline is shaped this way - the four modelling mistakes it is organized around.
This page covers what actually runs, in what order, and what depends on what.
Per-file detail is in [code/](code/), one page per source file.

## The two processes

There are exactly two, and they share no state except a directory.

```
                      ┌──────────────────────────────────────┐
  data/raw/*.csv ───► │  riskscore train | compare           │  (batch, minutes)
                      │  pipeline.py orchestrates            │
                      └──────────────────┬───────────────────┘
                                         │ publishes, atomically
                                         ▼
                            reports/runs/<run_id>/
                              model.joblib   ← the ScoringBundle
                              manifest.json  metrics.json  *.csv  figures/
                            reports/registry.json
                            reports/active_run.json
                                         │ read once, at boot
                                         ▼
                      ┌──────────────────────────────────────┐
   HTTP ────────────► │  riskscore serve                     │  (online, milliseconds)
                      │  FastAPI, uvicorn, api/app.py        │
                      └──────────────────────────────────────┘
```

**The bundle directory is the only interface between them.**
Training never imports the API; the API imports the pipeline in exactly one place, and that place runs it in a child process ([code/jobs.md](code/jobs.md)).
So a broken model is a bad file rather than a crashed service, and rolling back is `riskscore activate <run_id>` plus a restart.

**`active_run.json` is the only file the service reads to decide what to serve.**
Training writes a run and does *not* activate it - `riskscore compare` deliberately leaves the pointer alone, because choosing what to serve in production is a decision and not a side effect of measuring.

## The training path

`riskscore train` and `riskscore compare` both funnel into `_execute_run` in [`pipeline.py`](code/pipeline.md), which is pure orchestration in six numbered stages.
The order is the whole point: every stage below can only see what the stages above it have already decided.

```
1. admissible rows
     read (column-subset, declared dtypes)      data_loading.py
     alias-resolve, quarantine bad dates        schema.py
     filter to closed statuses                  data_loading.py
     OUTCOME-MATURITY EMBARGO                   data_loading.py
     derive the binary target                   data_loading.py
     select the feature tier                    features.py
     audit for unknown / leaky columns           leakage_check.py
     tri-split by issue_d                       data_loading.py
           │
           ├── TRAIN ──► fit preprocessor + estimator     transformers.py, modeling.py
           │
2.         ├── VALIDATION ──► fit the calibrator          calibration.py
3.         │                  select the threshold        evaluation.py
           │
4.         └── TEST ──► score once. Report only.          evaluation.py
5. drift: PSI on the score and per feature, metrics by vintage   drift.py
6. publish: bundle, manifest, artifacts, card, registry          artifacts.py, reporting.py
```

**The embargo runs before the split**, so all three partitions share one definition of "outcome known".
Applying it per-partition would give the test window a different survivorship profile than train, which is the bug it exists to prevent wearing a different hat.

**Validation is the only partition the calibrator and the threshold ever see.**
Enforced structurally rather than by convention: both functions accept a `ValidationScores` wrapper and nothing else, so handing them test scores is a mypy error *and* a runtime error, and a regression test monkeypatches both to assert the test arrays never appear in their arguments.
See [decisions/0003-train-validation-test-split.md](decisions/0003-train-validation-test-split.md) and [decisions/0006-calibration-on-validation.md](decisions/0006-calibration-on-validation.md).

**Test is scored exactly once**, with the frozen pipeline, the frozen calibrator, and the already-chosen threshold.
There is no code path that scores test, looks at the number, and changes anything.

## Feature engineering lives inside the estimator

The single most consequential structural decision here.
Every transformation - alias renaming, `'13.56%'` -> `0.1356`, `' 36 months'` -> `36.0`, the six engineered features, dropping the raw strings they replace - is a step in the sklearn `Pipeline` that gets pickled with the model.

```
Pipeline([
    ("canonicalize", CanonicalizeFrame(feature_spec)),   # transformers.py
    ("engineer",     EngineerFeatures(feature_spec)),    # transformers.py
    ("preprocess",   ColumnTransformer(...)),            # modeling.py
    ("estimator",    LogisticRegression | XGBClassifier),
])
```

Three things follow, and each of them replaces a class of bug with an impossibility:

**Train and `/predict` cannot diverge.**
There is no second implementation of the parsing to keep in step, because the request goes through the same transformer instances the training data did.
This is why the API accepts either dialect - `" 36 months"` or `36` - with no per-endpoint parsing code.

**No string column can reach `OneHotEncoder`.**
The `ColumnTransformer` is built from the explicit `feature_spec.numeric_features` and `.categorical_features` lists, never from `select_dtypes`.
Routing by runtime dtype is what turned `earliest_cr_line` (655 distinct values) into 655 dense one-hot columns; declared lists make the width of the transformed matrix a property of the spec.

**SHAP gets real feature names for free**, out of `get_feature_names_out`, which is what makes a reason code say `credit_utilization` instead of `x[17]`.

The cost is real and is paid deliberately: a pickle is coupled to the module path forever.
`transformers.py` may not move or be renamed, and `BUNDLE_SCHEMA_VERSION` is checked on load so an incompatible bundle is a named refusal rather than an `AttributeError` deep inside joblib.
See [decisions/0002-feature-engineering-inside-the-pipeline.md](decisions/0002-feature-engineering-inside-the-pipeline.md).

## The bundle

One frozen dataclass, one `joblib.dump`, in [`artifacts.py`](code/artifacts.md).

| Field | Why it is in the same file as the model |
| --- | --- |
| `pipeline` | The fitted estimator, transformers included. |
| `calibrator` | A fitted object. A model served with somebody else's calibrator is miscalibrated. |
| `threshold` | **The decision rule cannot drift apart from the model that it was chosen for.** |
| `feature_spec` | What the model expects, so `/api/schema` is generated rather than maintained. |
| `shap_background` | A k-means summary of at most 200 transformed training rows, so a pre-fit explainer needs no training data at serve time. |
| `metadata` | Run id, git commit, dataset hash, row counts at every filter stage, split windows, embargo rule, target definition, tier, cost matrix, library versions, schema version. |

Everything a reader needs is *also* written as plain JSON and CSV beside it, so identity and metrics are legible without unpickling anything.
That is not redundancy: `manifest.json` is what makes a two-year-old run auditable on a machine where the pickle no longer loads.

**Publication is atomic.**
Everything is written into `reports/runs/.staging-<uuid>/`, fsynced, then moved with `os.replace`.
A run directory is either absent or complete, so a concurrent `GET` can never see new metrics beside an old calibration curve.
`registry.json` and `active_run.json` are updated by temp-write plus `os.replace` under an `O_CREAT|O_EXCL` lock with a stale timeout - no new dependency, and a crash leaves a droppable `.staging-` directory rather than a plausible-looking run.
See [decisions/0007-the-scoring-bundle.md](decisions/0007-the-scoring-bundle.md).

## The serving path

`create_app(settings)` builds the app; a `lifespan` handler does all the expensive work once.

```
boot:     active_run.json -> load_bundle() -> app.state.bundle
                          -> build the SHAP explainer from shap_background
                          -> warm the mtime-keyed report cache

request:  POST /predict
            dict -> one-row DataFrame with pre-declared dtypes
                 -> pipeline.transform -> calibrator -> probability
                 -> threshold -> decision
                 -> LinearExplainer / TreeExplainer -> reason codes
                 -> response
```

**Zero disk IO on the hot path.**
No unpickling, no explainer construction, no report file read: every one of those happens at boot or on a cache miss keyed by file mtime.

That is what the latency budget rests on: **p50 <= 12 ms / p99 <= 25 ms without reason codes, p50 <= 25 ms / p99 <= 50 ms with them**, for a warm process against a logistic-regression bundle.
Measured on the development machine: 7.6/8.1 ms and 14.9/15.7 ms.
`riskscore bench` prints the figures; `tests/test_bench.py` asserts a deliberately generous 250 ms p99 under `-m slow`, so CI catches a catastrophic regression without being flaky on a shared runner.

**Preprocessing is the request.**
Measured, not assumed: about 5.5 ms in the two pandas transformers, 1.5 ms in the `ColumnTransformer`, and under 0.1 ms in the model itself.
So the optimizations that mattered were pandas-pass reductions in `feature_engineering.py` and `transformers.py`, which sped up training by the same mechanism, and the 13 ms that reason codes cost is a *second* pass through the preprocessor - the calibrator wraps the whole pipeline, so a calibrated probability needs the raw frame while the explainer needs the transformed matrix.
Batches do not pay it per row: 200 applicants cost about 8.8 ms in total, because the cost is per pass.

The route surface splits along a security line rather than a topical one:

| Module | Routes | Available |
| --- | --- | --- |
| [`routes_public.py`](code/routes_public.md) | `/predict`, `/predict/batch`, `/api/*` reports, run history, `/artifacts/{run_id}/{name}`, `/healthz`, `/readyz` | Always. All read-only. |
| [`routes_admin.py`](code/routes_admin.md) | `POST /api/datasets`, `POST /api/runs`, `GET /api/jobs/{id}` | **Off** unless `RISKSCORE_ALLOW_UPLOAD` / `RISKSCORE_ALLOW_RETRAIN` |

`/artifacts/` serves from an explicit twelve-filename allowlist with resolved-path containment and no directory listing.
`model.joblib` exists in every run directory and is not on the list, so it 404s - the smoke test asserts both halves, because "the allowlist is closed" is a claim until something asks for the file it excludes.

**Retraining runs in a child process**, not a thread: matplotlib's pyplot global state is not thread-safe, a fit holding the GIL would stall `/predict`, and a process gives a hard kill for timeouts.
One job at a time or `409`.
`/predict` keeps serving the old bundle for the whole fit and the new one after, so no request ever sees a half-built model.
See [decisions/0008-retraining-in-a-child-process.md](decisions/0008-retraining-in-a-child-process.md).

## Module dependencies

The graph is acyclic and layered, and it stays that way because nothing below imports anything above it.

```
L0  features.py   evaluation.py   logging_setup.py   sample_data.py   api/settings.py
        │              │
L1  schema.py  feature_engineering.py  leakage_check.py      config.py
        │              │
L2  transformers.py                  data_loading.py
        │                                  │
L3  modeling.py    artifacts.py         cache.py
        │              │
L4  calibration.py  explain.py
                       │
L5                 drift.py
                       │
L6                 reporting.py
        ┌──────────────┘
L7         pipeline.py
                │  (imported by api/ only here)
L8         api/jobs.py

    api/settings.py ─► api/scoring.py ─► api/deps.py ─┐
    api/reports.py ──────────────────────────────────┼─► api/routes_public.py ─┐
    api/schemas.py ──────────────────────────────────┘   api/routes_admin.py ──┴─► api/app.py

L9  cli.py   (the only module that imports both pipeline and api)
```

Three properties of this graph are load-bearing:

**`features.py` has no internal imports.**
The column registry is the bottom of the stack, which is what lets every other module treat it as the single source of truth about what a column is - see [data-dictionary.md](data-dictionary.md).

**`explain.py` sits below both `drift.py` and `api/scoring.py`.**
The global SHAP summary written at training time and the per-applicant reason codes served at request time come from one implementation, so a bar chart on the dashboard and a reason code in a response cannot disagree about what a feature contributed.

**Only `api/jobs.py` reaches into `pipeline.py`.**
Every other API module is read-only with respect to the run tree.
That single edge is the entire retraining blast radius, and it is the one that runs in a subprocess behind an off-by-default flag.

## The command surface

Nine subcommands, one entry point ([`cli.py`](code/cli.md)).

| Command | What it does |
| --- | --- |
| `make-sample-data` | Write a synthetic raw extract, so the demo needs no 1.19 GB download. |
| `train` | One model, one tier, one run directory. |
| `compare` | Two or more models and/or both tiers on **one identical split**, plus `comparison.json`. |
| `explain` | Reason codes for one row of a CSV, from a published bundle, without a server. |
| `card` | Re-render a run's model card. |
| `runs` | List the registry. `--rebuild` reconstructs it from the run directories. |
| `activate` | Point `active_run.json` at a run. The documented rollback path. |
| `serve` | The FastAPI service plus the dashboard, from one process. |
| `bench` | p50/p90/p99 for `/predict`, with and without reason codes. |

`compare` exists because a model comparison across two different splits measures the splits.
It fits every variant against one set of row indices and reports them side by side, which is also how the cost of admitting the `LENDER_PRICED` tier gets *quantified* rather than asserted.

## The dashboard

Vanilla ES modules, no npm, no bundler, served by the same process as the API.

```
dashboard/index.html
         styles/tokens.css   colours, spacing, type scale, both themes
         styles/app.css
         js/api.js           fetch wrapper: AbortController, ETag, in-memory cache
         js/dom.js           the handful of element helpers everything else shares
         js/format.js        Intl.NumberFormat; missing renders "-", never "0.000"
         js/charts.js        nice ticks, measureText labels, axes, legends
         js/panels.js        metrics, comparison, vintages, PSI, SHAP, run history
         js/score.js         the score-an-applicant form, generated from /api/schema
         js/retrain.js       upload and job polling; mounts only if the routes are on
         js/main.js          wiring
         js/*.test.js        node --test on the pure helpers - 104 tests, no npm install
```

Fetched data is cached in memory, so a resize redraws from cache and issues **zero** network requests - which removes the double fetch, the out-of-order stale draw, and the swallowed `.catch(() => {})` in one move.
Chart colours are read from CSS custom properties, so dark mode follows automatically from one source of truth.
`dashboard/package.json` exists for `"type": "module"` and one script; there is nothing to `npm ci`, which is why the CI job has no cache key and the container copies the directory as-is.
A build step would add more surface than it removes and would break the clone-and-run story: see [decisions/0009-vanilla-dashboard.md](decisions/0009-vanilla-dashboard.md).

## What proves any of this

| Claim | Checked by |
| --- | --- |
| Every module behaves as documented | `pytest -q` - 686 tests, value-exact against scipy/sklearn oracles |
| The properties a chosen example cannot cover | `tests/test_properties.py` - hypothesis on the cost table, KS, the parsers, PSI, threshold monotonicity |
| The artifact contract, end to end | [`scripts/smoke_e2e.py`](code/smoke_e2e.md) - real CLI, real service, on synthetic data |
| The dashboard renders and reflows | `scripts/probe_dashboard.mjs` - real Chrome, both themes, six widths |
| Every code file is documented | [`scripts/check_docs.py`](code/check_docs.md) - in CI, before the install |
| The image boots and serves | CI only. Docker is not available on the machine this was built on. |
