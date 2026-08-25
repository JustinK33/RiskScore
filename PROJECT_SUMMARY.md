# RiskScore Project Summary

## Executive Summary

RiskScore is a credit default risk system with two halves: a training pipeline that publishes versioned, self-describing model bundles, and a FastAPI service that loads one of those bundles and scores a single applicant with reason codes in single-digit to low-double-digit milliseconds.

The training half reads a Lending Club style extract, canonicalizes its column names against a declared registry, filters to terminal loan outcomes, applies an outcome-maturity embargo, builds a binary default target, selects a feature tier, splits by time into train/validation/test, fits the model, calibrates it and picks a decision threshold on validation only, scores the test partition exactly once, and writes the whole run into its own directory under `reports/runs/`.

The serving half loads one published bundle at boot, holds the model, the calibrator, the threshold and the explainer in memory, and does no disk IO on the hot path.

The dashboard is a static frontend served by that same process, and it reads only the service's JSON endpoints.

The honest headline is that removing the lender's own pricing signal and enforcing the embargo costs accuracy and buys trust: test AUC is 0.6448 on origination-only features against 0.6902 with the lender-priced tier added back, and both variants are reported side by side rather than one being quoted alone.

## High-Level Workflow

1. A user provides a raw loan dataset as CSV or parquet, or generates a synthetic one with `riskscore make-sample-data`.
2. `CanonicalizeFrame` resolves aliases to canonical names, reindexes to the declared column list, and parses each column by its declared kind rather than by guessing from its runtime dtype.
3. Rows are filtered to closed loan outcomes.
4. The outcome-maturity embargo drops every loan whose term had not finished by the snapshot date, so a loan that simply has not had time to default yet is not counted as a survivor.
5. `create_default_target` maps the remaining outcomes into `default_flag`.
6. A feature tier is selected, which is what excludes post-origination and lender-priced columns by construction instead of by a deny list.
7. The frame is split by issue date into train, validation and test, with a gap between partitions.
8. The preprocessor and the estimator are fit on train, and nothing else touches train.
9. The calibrator is fit and the cost-aware threshold is chosen on validation.
10. The test partition is scored once with the frozen model, calibrator and threshold, and is report-only.
11. Metrics, calibration curves, threshold costs, per-vintage breakdowns, PSI drift, a SHAP summary, figures, a model card and a manifest are staged and then atomically published as `reports/runs/<run_id>/`.
12. `riskscore activate <run_id>` points `reports/active_run.json` at a run, and that file is the only thing the service reads to decide what it serves.

## Repository Infrastructure

The project is a Python package under `src/risk_score`, with the service under `src/risk_score/api`.

The single entrypoint is `src/risk_score/cli.py`, installed as `riskscore`, with nine subcommands: `train`, `compare`, `explain`, `card`, `runs`, `activate`, `serve`, `bench` and `make-sample-data`.

There is no second server.

The hand-rolled `scripts/serve_dashboard.py` is gone, along with the unauthenticated POST that used to retrain the model on the request thread and overwrite the pickle the server was reading.

Static dashboard assets live under `dashboard/`, and the service mounts them.

Run configuration lives in `configs/run.yaml`, which is the single place split windows, the cost matrix, the feature tier and column aliases are specified.

Raw datasets live under `data/raw/`, and a parquet read cache keyed by input hash lives under `data/processed/`.

Every published run lives under `reports/runs/<run_id>/`, indexed by `reports/registry.json`, with `reports/active_run.json` naming the served one.

Tests live under `tests/`, and the dashboard's own unit tests live beside the modules they cover as `dashboard/js/*.test.js`.

Raw data, processed data, reports, virtual environments and caches are ignored by git, and only `.gitkeep` placeholders are tracked in the artifact directories.

## Dataset Compatibility Layer

`src/risk_score/features.py` holds `COLUMN_REGISTRY`, which declares 50 columns.

Each spec carries the canonical name, its aliases, its tier, whether it is required, and the parse kind to apply.

There are seven parse kinds: `NUMERIC`, `PERCENT`, `TERM_MONTHS`, `EMP_LENGTH_YEARS`, `MONTH_DATE`, `CATEGORY` and `TEXT`.

The parse rule is declared per column, never inferred, which is what stops `int_rate` arriving as the string `'13.56%'` and being one-hot encoded into 600 columns.

Only five columns are required: `loan_amnt`, `term`, `annual_inc`, `issue_d` and `loan_status`.

Everything else is optional and is filled with NA when absent, so both public Lending Club extracts run without edits.

Alias resolution is priority-ordered, and losing source columns are dropped and recorded rather than renamed on top of a canonical column that already exists.

That is the fix for the duplicate-column bug where a frame containing both `funded_amnt` and `loan_amnt` came back with two columns named `loan_amnt`.

Aliases have a three-character minimum, which is what removed the single-letter `n` alias that used to hijack the target column.

Dates are parsed with an ordered list of explicit whole-column formats, so an ambiguous file raises instead of silently mixing March and April across rows.

`fico_range_low` and `fico_range_high` are absent from both real extracts, so the FICO features are optional in practice and the pipeline runs without them.

`docs/data-dictionary.md` documents all of this one table per tier, with every column's aliases and parse rule checked against both real extracts.

## Data Loading

`src/risk_score/data_loading.py` reads `.csv`, `.parquet` and `.pq`.

It reads only the columns the registry declares, with declared dtypes, instead of all 145 columns in the raw file.

`src/risk_score/cache.py` writes a parquet copy of the read keyed by a hash of the input, so a second run over the same 1.1 GB CSV starts from parquet.

The loader requires a canonical `loan_status` and filters to closed statuses only.

The closed-status set is immutable and covers `Fully Paid`, `Charged Off`, `Default` and the two `Does not meet the credit policy` variants.

## Target Creation and the Embargo

`create_default_target` writes `default_flag`, mapping paid loans to `0`, charged-off and defaulted loans to `1`, and leaving anything ambiguous missing.

Filtering to closed loans alone is not enough, and this is the correction that matters most in the project.

Without an embargo, measured default rate by vintage is 2013 15.6%, 2014 18.5%, 2015 20.2%, 2016 24.3% and 2018 14.7%.

The 2018 figure is not a good year, it is survivorship: the only 2018 loans that had closed by the snapshot are the ones that defaulted early or prepaid.

Requiring `issue_d + term <= snapshot` flattens it to 2013 15.6%, 2014 13.7% and 2015 14.9%.

On the real extract the embargo removes 630,269 of 1,306,387 closed loans, and it runs before the split so all three partitions share one outcome definition.

## Leakage Controls

Leakage is handled by tiering columns in the registry, not by a deny list applied after the fact.

The seven tiers and their counts are `BORROWER` 18, `POST_ORIGINATION` 17, `IDENTIFIER` 6, `LENDER_PRICED` 4, `LOAN_REQUEST` 3, `TARGET_SOURCE` 1 and `TIMELINE` 1.

Only `BORROWER` and `LOAN_REQUEST` reach the model by default.

`POST_ORIGINATION` covers payment totals, recoveries, last payment dates, outstanding principal and later credit pulls, and none of it is available at underwriting time.

`IDENTIFIER` covers ids, URLs and free text, plus `zip_code`, which is excluded as a fair-lending matter rather than a leakage one.

`LENDER_PRICED` is the interesting tier: `int_rate`, `grade`, `sub_grade` and `installment` are all available at origination, but every one of them is the output of somebody else's risk model.

Training on them measures how well the model can recover Lending Club's own grading, not how well it can underwrite.

So they are excluded by default, available behind `riskscore train --include-lender-priced`, and `riskscore compare --tiers both` quantifies what they are worth instead of asserting it.

See `docs/decisions/0005-lender-priced-feature-tier.md`.

## Feature Engineering

Feature engineering runs inside the sklearn pipeline, not before it, which is the decision that makes train and serve share one code path.

`src/risk_score/transformers.py` holds two estimators: `CanonicalizeFrame`, which does aliasing, reindexing and declared-dtype parsing, and `EngineerFeatures`, which calls the pure builders in `feature_engineering.py` and drops the raw source columns each engineered feature replaces.

`ENGINEERED_FEATURES` declares six features, each with what it `requires`, what it `requires_any`, and what it `consumes`.

The `consumes` field is what makes the 35 GB one-hot blow-up unrepresentable: a raw string column that has been replaced cannot survive to reach the encoder.

The features are cleaned DTI, credit utilization parsed from either `'47.5%'` or a float, loan-to-income ratio with zero income treated as missing, credit history months derived from `earliest_cr_line`, and the two FICO features when both range columns exist.

The `ColumnTransformer` is built from the declared numeric and categorical lists, never from `select_dtypes`, and the one-hot encoder uses `handle_unknown="infrequent_if_exist"`, `min_frequency=0.005` and sparse output.

See `docs/decisions/0002-feature-engineering-inside-the-pipeline.md`.

## Modeling

`src/risk_score/modeling.py` builds the four-step pipeline and fits either a logistic regression or an XGBoost classifier.

`class_weight="balanced"` was removed from the logistic regression default, because it was the direct cause of the miscalibration the old Brier score and calibration plot were describing.

XGBoost gets `scale_pos_weight` only when the logistic regression is weighted, so a comparison between them is apples to apples.

Early stopping cannot go through `Pipeline.fit`, so `fit_with_validation_monitoring` fits the preprocessing prefix on train, `transform`s validation through it, fits the estimator with `eval_set`, and reassembles a `Pipeline` from the already-fitted steps.

The `transform`-not-`fit_transform` line is where the whole no-leakage claim lives, and a test asserts the reassembled pipeline's predictions equal the manual two-step predictions.

`ensure_model_available` preflights every requested model before any data is read, so an XGBoost wheel that cannot load says so in one import rather than after a 1.19 GB read, and a four-variant comparison cannot publish three runs and then abort before writing `comparison.json`.

## Train, Validation and Test

The split is three-way, and who sees what is the point of it.

Train fits the preprocessor and the estimator, and nothing else.

Validation drives XGBoost early stopping, fits the calibrator, and selects the threshold.

Test is scored exactly once with the frozen model, calibrator and threshold, and is report-only.

This is enforced structurally, not by convention: `fit_calibrator` and `select_threshold` accept only a partition-tagged `ValidationScores` wrapper, so handing them test scores is both a mypy error and a runtime error, and a regression test monkeypatches both to assert the test arrays never appear in their arguments.

The defaults are measured rather than picked: snapshot `2018-12-01`, 36-month terms only, train `2013-01-01` to `2014-09-30`, validation `2014-10-01` to `2015-03-31`, test `2015-04-01` to `2015-12-31`.

The 36-month restriction is not arbitrary either, because with the embargo applied 60-month loans only exist through 2013Q4, which would put 13.9% 60-month loans in train against 0.0% in validation and test.

Unparseable dates are quarantined and counted instead of aborting the run.

See `docs/decisions/0003-train-validation-test-split.md` and `docs/decisions/0004-outcome-maturity-embargo.md`.

## Evaluation, Calibration and Threshold

`src/risk_score/evaluation.py` reports AUC ROC, average precision, the KS statistic, Brier score, ECE with per-bin counts, and a cost-aware threshold.

The KS statistic is tie-correct and direction-aware, aggregating ties before the cumulative sum.

The old artifact that reported `auc_roc: 0.0696` next to `ks_statistic: 0.9304` is now impossible to produce, and there is a test that pins the exact value against `scipy.stats.ks_2samp`.

Threshold selection replaced 99 full-array `confusion_matrix` calls with one argsort and two cumulative sums, carries a minimum approval-rate guard, and breaks ties toward the higher threshold, which is what stopped the cost minimizer from recommending "decline everyone at 0.01".

Calibration is fit on validation through `CalibratedClassifierCV(FrozenEstimator(model))`, because the `cv="prefit"` argument the old dead code used was removed in scikit-learn 1.9, and it falls back to sigmoid when positives are few.

Every metric helper is compared to a scipy or sklearn oracle or a hand-computed constant, and `assert 0 <= result <= 1` style assertions are banned, since that assertion style is exactly why the inverted KS survived.

See `docs/decisions/0006-calibration-on-validation.md`.

## The Scoring Bundle and Run Registry

`src/risk_score/artifacts.py` owns `ScoringBundle`, a frozen dataclass and the only object joblib ever writes.

It carries the pipeline, the calibrator, the threshold, the feature spec, a SHAP background summary and a metadata block, plus a `bundle_schema_version`.

The threshold is in the bundle rather than in a config file so that the model and the decision rule cannot drift apart.

The metadata records the run id, creation time, git commit, dataset path and hash prefix, row counts at every filter stage, the split windows, the embargo rule and how many rows it removed, the target definition, the feature tier, the cost matrix and the library versions.

A run is staged into `reports/runs/.staging-<uuid>/`, written in full, fsynced and then `os.replace`d, so a partially written run is never reachable by name and a concurrent read can never see new metrics beside an old calibration curve.

The registry and `active_run.json` are updated the same way under an `O_CREAT|O_EXCL` lock with a stale timeout, which needs no new dependency.

`run_id` is `<UTC basic timestamp>-<model>-<tier>-<git short sha>`, and retention keeps the most recent 20 runs.

See `docs/decisions/0007-the-scoring-bundle.md`.

## Explainability, Drift and Comparison

`src/risk_score/explain.py` builds a pre-fit explainer from the bundle alone, so nothing needs the training data at serve time.

Reason codes are log-odds contributions with human labels, and one-hot families collapse back to the source feature, so a reason reads `home_ownership = RENT` rather than `cat__home_ownership_RENT`.

`shap` itself is not a dependency, because it would drag in a numba and llvmlite toolchain for something the logistic regression coefficients and XGBoost's built-in TreeSHAP already provide.

`src/risk_score/drift.py` computes PSI for the score and for every feature, with the conventional 0.10 and 0.25 bands, plus a per-vintage metric breakdown.

`riskscore compare` fits several variants on an identical split and writes `comparison.json` and a table, and it is what turns the leakage argument into a number.

`src/risk_score/reporting.py` renders the model card from a template, and CI asserts no `$placeholder` survives in the rendered card.

## The Service

`create_app(settings)` has a lifespan handler that reads `active_run.json`, loads the bundle, builds the explainer once and warms an mtime-keyed report cache.

There are 19 routes.

`POST /predict` takes the applicant as the bare JSON body, with `explain` and `top_k` as query parameters, and returns the calibrated default probability, the decision, the threshold, the baseline and total log odds, the reason codes, the model identity and `latency_ms`.

`POST /predict/batch` handles up to 1000 rows in one preprocessing pass with per-row errors inline.

`GET /api/model`, `/api/schema`, `/api/metrics`, `/api/calibration`, `/api/threshold-costs`, `/api/vintages`, `/api/drift`, `/api/shap-summary` and `/api/comparison` serve the report payloads columnar rather than as repeated-key row objects, with gzip, ETag and `Cache-Control`.

`GET /api/runs`, `/api/runs/{run_id}` and `/api/runs/{run_id}/card` serve the run history, and `GET /artifacts/{run_id}/{name}` serves files against an explicit filename allowlist with resolved-path containment and no directory listing.

`POST /api/datasets`, `POST /api/runs` and `GET /api/jobs/{job_id}` are the retrain flow, and `/healthz` and `/readyz` are the probes.

Retraining runs in a child process rather than a thread, because matplotlib's pyplot state is not thread-safe, a fit holding the GIL would stall `/predict`, and a process gives a hard kill on timeout.

It is single-flight, answers `202` with a `job_id`, and `/predict` keeps serving the old bundle for the whole fit.

Both request models set `extra="forbid"`, so a typo in a field name is a 422 rather than a silently ignored input.

Security posture: bound to `127.0.0.1` by default and refusing `0.0.0.0` without an explicit flag, `TrustedHostMiddleware` against DNS rebinding, CORS off, a body-size limit, `X-API-Key` on every mutating route, uploads off unless `RISKSCORE_ALLOW_UPLOAD=1`, and a global exception handler that logs the traceback against a request id and returns a body containing neither a filesystem path nor the dataset's date range.

See `docs/decisions/0008-retraining-in-a-child-process.md`.

## Dashboard

The dashboard is vanilla ES modules with no npm, no bundler and no framework, loaded with `<script type="module">`, which is natively deferred.

The modules are `api.js`, `charts.js`, `dom.js`, `format.js`, `main.js`, `panels.js`, `retrain.js` and `score.js`, with `node --test` unit tests beside the four that hold pure logic.

Styling is a two-file token layer, `styles/tokens.css` for colors, spacing, type scale and radii in both themes, and `styles/app.css` for layout.

Chart colors are read from CSS custom properties via `getComputedStyle`, so dark mode follows automatically from one source of truth.

Fetched data is cached in memory, so a resize redraws from cache and issues zero network requests, which removed the double fetch, the out-of-order stale draw and the swallowed `.catch(() => {})` in one change.

All chart text is positioned with `measureText` and `textAlign`, replacing hardcoded half-widths that were already wrong because `Inter` was requested in five places and never loaded.

The panels are score-an-applicant, metrics, calibration, threshold costs, model comparison, per-vintage default rates, a PSI table with the standard bands, a global SHAP bar chart, and a run-history selector.

A header strip shows the run id, generated-at, dataset, tier and per-split row counts, and a sanity banner appears when the test partition has fewer than 1000 rows or fewer than 50 positives, because the old dashboard rendered an n=116 run with one positive as confidently as a real result.

See `docs/decisions/0009-vanilla-dashboard.md`.

## Documentation

`docs/README.md` is the index.

`docs/architecture.md` covers the two processes, the training path, the bundle, the serving path and a layered module dependency graph.

`docs/data-dictionary.md` covers every canonical column, its aliases, its tier and its parse rule.

`docs/runbook.md` covers running, Docker, retraining, rolling a bundle back and reading a PSI alarm.

`docs/code/` has one page per code file, 44 pages, each with the same seven required headings: Purpose, Public API, Inputs and outputs, Invariants and failure modes, What must NOT live here, Related tests, Known limits.

`docs/decisions/` holds eight ADRs.

`scripts/check_docs.py` runs in CI and fails if a code file has no page or a page is missing a heading, so the docs cannot silently rot.

Inline comments follow one rule: the docstring is the contract, and an inline `#` explains why, never what.

They are mandatory at guard clauses, non-obvious pandas idioms, magic numbers with their provenance, leakage-relevant choices, order-dependent steps, and every fixed bug, tagged with an audit id that greps to exactly two places, the fix and the test that proves it.

## Command-Line Usage

Generate a synthetic extract and fit on it, which is the no-download path.

```bash
riskscore make-sample-data data/sample/loans.csv
riskscore train data/sample/loans.csv
```

Fit against a real extract with a run configuration.

```bash
riskscore train data/raw/1/loan.csv --config configs/run.yaml
```

Quantify what the lender-priced tier is worth.

```bash
riskscore compare data/raw/1/loan.csv --config configs/run.yaml --tiers both
```

List the published runs, choose what is served, and start the service.

```bash
riskscore runs
riskscore activate 20260825T025646315Z-logistic_regression-origination_only-91b97bb
riskscore serve
```

Score one applicant.

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "loan_amnt": 12000, "term": " 36 months", "annual_inc": 62000,
  "purpose": "debt_consolidation", "home_ownership": "RENT",
  "dti": 18.4, "revol_util": "62.5%", "issue_d": "Jun-2015",
  "earliest_cr_line": "Aug-2003", "open_acc": 9, "total_acc": 22
}'
```

Measure the latency of that path, and explain one row of a CSV from the terminal.

```bash
riskscore bench
riskscore explain data/sample/loans.csv --row 0 --top-k 5
```

Run the checks.

```bash
ruff check . && ruff format --check . && mypy src tests scripts && pytest -q
python scripts/check_docs.py
cd dashboard && node --test
```

## Measured Results

Measured on `data/raw/1/loan.csv`, 1.1 GB, with `riskscore compare --tiers both`.

The read produced 2,260,668 rows, of which 1,306,387 were closed.

The embargo at snapshot `2018-12-01` kept 676,118 of those and removed 630,269 as immature.

The split produced train 212,972, validation 106,588 and test 226,285, dropping 75,004 rows into the inter-partition gaps and 0 to unparseable dates.

Test-partition results, n = 226,285, scored once:

| Variant | AUC ROC | KS | Brier | ECE | Approval rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `origination_only` | 0.6448 | 0.2084 | 0.1225 | 0.0055 | 0.6502 |
| `with_lender_priced` | 0.6902 | 0.2801 | 0.1195 | 0.0076 | 0.6516 |

The lender-priced tier is worth 0.0454 AUC, and that is the number the leakage discussion is about.

Train-to-test drift is stable in both variants, score PSI 0.0042 and 0.0157.

The worst individual feature differs between them, and informatively so: `verification_status` at 0.0795 without the lender-priced tier, `int_rate` at 0.2816 with it, which is Lending Club repricing its own book across 2013 to 2015 showing up as feature drift.

Resource cost for both variants in one command was 18.41 s wall, 3.32 GB peak resident set and 3.64 GB peak footprint, with the second variant reading from the parquet cache.

`POST /predict` latency, warm, logistic regression bundle, one applicant, n = 2000 in-process calls:

| Path | p50 | p90 | p99 | Budget p50 / p99 |
| --- | ---: | ---: | ---: | ---: |
| Score only | 8.06 ms | 8.43 ms | 11.19 ms | 12 / 25 ms |
| With reason codes | 15.76 ms | 17.02 ms | 32.16 ms | 25 / 50 ms |

Over real HTTP the warm p50 is 16.57 ms with reason codes and 9.17 ms without.

Roughly 5.5 ms of a request is the two pandas transformers, 1.5 ms is the `ColumnTransformer` and under 0.1 ms is the model itself, so preprocessing is the request.

Reason codes cost a second preprocessing pass because the calibrator wraps the whole pipeline.

That per-pass cost is also why a 200-row batch is about 8.8 ms in total rather than 200 times a single call.

## How the Old Numbers Compare

The previous summary quoted 0.7007 AUC ROC, 0.3611 average precision and 0.2925 KS as a full-dataset baseline.

That figure is not comparable to 0.6448 and should not be read as a regression, for four reasons.

It included the lender-priced columns, it had no outcome-maturity embargo, its decision threshold was selected on the test set it was reporting, and it trained with balanced class weights that inflated the predicted probabilities its Brier score and calibration plot were describing.

The current numbers are lower and mean what they say.

## Validation Performed

`pytest -q` passes 687 tests with 3 skipped.

The three skips are XGBoost paths, skipped because `libomp` is not installed on this machine; the Linux CI job runs them.

`node --test` from `dashboard/` passes 101 tests.

`ruff check .`, `ruff format --check .` over 122 files, and `mypy src tests scripts` over 59 files are all clean.

`python scripts/check_docs.py` reports 44 pages, all present and complete, and there are 0 broken relative links across `docs/`.

CI has four jobs.

Lint, format, strict mypy and per-file documentation coverage run on 3.12.

The test suite runs on 3.12 and 3.13 with `libgomp1` installed and an explicit step that asserts XGBoost imports, so the three tests skipped locally genuinely execute somewhere.

`node --test` runs the dashboard helpers on Node 22.

The last job runs `make-sample-data`, `compare`, `activate` and `card` on synthetic data, boots the service, hits `/predict` and the report endpoints, asserts every documented artifact exists, and then repeats those checks against the built container.

That smoke job is the only automated proof of the artifact contract, because `reports/` is gitignored.

The Docker image is multi-stage on `python:3.13-slim`, installs the serving requirements only, runs as a fixed non-root uid 10001 so a bind-mounted `reports/` has stable ownership, and has a `HEALTHCHECK` on `/readyz`.

It deliberately does not install `libgomp1`, because XGBoost is in the `[train]` extra and a serving image that cannot fit a model does not need the OpenMP runtime.

CI builds the image, runs the smoke run inside it, waits on that `HEALTHCHECK`, and repeats the service checks against the container, which is the only place the image is exercised, because Docker is not available on this development machine.

## Known Gaps

Neither real extract contains `fico_range_low` or `fico_range_high`, so the two FICO features are exercised only by synthetic data and by unit tests.

The XGBoost comparison is unverified on this machine until `libomp` is installed, and is covered by the Linux CI job in the meantime.

`riskscore compare` fits variants sequentially, so a four-variant comparison on the real extract costs four fits of wall clock.

The `/predict` reason-code path pays a second preprocessing pass, and collapsing it would mean calibrating the estimator separately from the pipeline.

The retrain-over-HTTP flow is inherently the most dangerous surface in the project, since it is remote-triggered execution over supplied data that ends in a pickle write, so it stays off unless explicitly enabled and is documented as a local-demo feature.

## Resume Bullets

- Built a leakage-aware credit default risk system in Python: a training pipeline that publishes versioned, self-describing model bundles and a FastAPI service that scores a single applicant with reason codes at a measured p50 of 15.8 ms including explanations and 8.1 ms without.
- Found and fixed a one-hot encoding blow-up that made the pipeline non-functional on the real 1.1 GB dataset, by declaring each column's parse rule and feature tier in a registry instead of routing columns to transformers by runtime dtype, reducing a projected roughly 35 GB dense matrix to a sparse one and bringing peak memory to 3.3 GB.
- Identified survivorship bias in the published default rates, where the 2018 vintage read 14.7% against 2016's 24.3%, and corrected it with an outcome-maturity embargo that removed 630,269 immature loans and flattened the series to 15.6%, 13.7% and 14.9%.
- Replaced a test-set threshold selection with a three-way time split, enforcing the boundary in the type system through a partition-tagged wrapper so that fitting calibration or a threshold on test data fails both type checking and at runtime.
- Quantified the cost of excluding lender-priced features rather than asserting it, reporting 0.6448 test AUC on origination-only features against 0.6902 with the lender's own grade and rate added back, on an identical 226,285-row test partition.
- Moved feature engineering inside the sklearn pipeline so training and inference share one code path, which made an inference-time schema mismatch a named error instead of a silent wrong answer.
- Fixed an alias resolver that produced duplicate column names on the standard Lending Club extract, where a frame containing both `funded_amnt` and `loan_amnt` came back with two columns of the same name.
- Made the KS statistic tie-correct and direction-aware, which is what made the committed artifact reporting 0.070 AUC beside 0.930 KS, the signature of inverted labels, impossible to produce again.
- Replaced an unauthenticated retrain endpoint that ran on the request thread and overwrote the model the server was reading with a single-flight child-process job, atomically staged run directories, and an API-key-gated route set bound to localhost by default.
- Built per-applicant reason codes, PSI drift monitoring, per-vintage metric breakdowns, an auto-generated model card and a run registry with rollback, with no SHAP dependency, using logistic regression coefficients and XGBoost's built-in TreeSHAP.
- Rebuilt the dashboard as vanilla ES modules with a design token layer, cached redraws that issue zero network requests on resize, measured text layout, and sanity banners for statistically meaningless runs, with no build step.
- Documented every code file on a page with seven required headings, 44 pages plus eight architecture decision records, and enforced the coverage in CI so the docs cannot silently rot.
- Verified with 687 Python tests, 101 JavaScript tests, strict mypy, ruff, and a CI end-to-end smoke job that boots the service and asserts every documented artifact exists.
