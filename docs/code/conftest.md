# `tests/conftest.py`

## Purpose

The fixtures every suite shares, and one decision behind all of them: **tests read the same synthetic generator the demo does.**

Three near-identical 8-row dataframes used to be pasted into `tests/test_pipeline.py`. That cost twice. A schema change needed three edits, and none of the three exercised realistic data - no percent strings, no `' 36 months'`, no missingness, no immature loans for the embargo to remove, and no signal, so a test could not tell a fitted model from a coin flip. Everything now comes from `risk_score.sample_data`, which is also what `riskscore make-sample-data` writes, so the fixtures and the clone-and-run demo cannot drift.

The second decision is that the expensive thing is fitted **once**. `trained_run` is session-scoped and produces a real published run - a real pipeline, real coefficients, a real design matrix, a real SHAP background - because explainability, drift, reporting, and every API endpoint need a bundle a stub estimator cannot impersonate.

## Public API

Fixtures, plus one marker.

| Name | Scope | What it gives |
| --- | --- | --- |
| `raw_loans` | function | A raw Lending Club-shaped frame, `SMALL_ROWS` rows. |
| `raw_loans_factory` | function | `make_synthetic_loans` itself, for non-default options. |
| `raw_csv` | function | `raw_loans` written to `tmp_path`. |
| `raw_csv_factory` | function | `**kwargs -> Path`, numbering each file so two calls in one test do not collide. |
| `trained_run` | **session** | One `RunResult` from a real `train_run`. Read-only. |
| `api_settings` | function | `Settings` pointing at the session run's reports tree. |
| `client` | function | `TestClient` over a real app with that bundle loaded. |
| `applicant` | function | One valid applicant in the extract's own string dialect. |

| Name | What it is |
| --- | --- |
| `requires_xgboost` | `pytest.mark.skipif` that checks XGBoost *loads*, not that it imports. |
| `SMALL_ROWS = 900` | The row count every synthetic fixture uses. |
| `DASHBOARD_DIR` | The real `dashboard/`, resolved from this file. |

## Inputs and outputs

Imports `risk_score.sample_data`, `risk_score.pipeline.train_run`, and `risk_score.api`. Nothing else in the suite imports `create_app` directly.

Writes only under pytest's `tmp_path` / `tmp_path_factory`. No test touches the repository's `reports/`, and none reads the real `data/`.

Reads no environment variable, which is deliberate and stated below.

## Invariants and failure modes

**`requires_xgboost` checks that XGBoost *loads*, not that it is installed.**
`pytest.importorskip("xgboost")` is not enough. On macOS without the OpenMP runtime the package is present and importing it raises `XGBoostError` from a failed `dlopen` - which is not an `ImportError`, so `importorskip` lets it through and the test **fails** instead of skipping. The guard catches bare `Exception` for exactly that reason, and the skip reason names the fix: `brew install libomp`.

**The stub-based XGBoost tests deliberately carry no marker.**
The fit-transform-reassemble logic they cover - fit the preprocessor, transform train and validation, fit the estimator with an `eval_set`, then assemble a `Pipeline` from the already-fitted steps - is this project's code, not the library's, and it must be verified on every machine including one without `libomp`. Marking them would let the only machine that can run the real fit be the only machine that checks the assembly.

**`api_settings` is built explicitly, never from the environment.**
`Settings()` reads `RISKSCORE_*` and `.env`. A suite that let it default would pass or fail depending on the developer's shell, and the failure would be a security test passing on a machine that had `RISKSCORE_ALLOW_UPLOAD=1` exported. `log_level="WARNING"` keeps the run log out of the test output.

**`DASHBOARD_DIR` is resolved from `__file__`, not from the working directory.**
`Settings.dashboard_dir` defaults to a relative path, so a suite that let it default would mount the dashboard when run from the repo root and not when run from anywhere else - a difference nobody would look for, and one that turns the static-mount tests into a function of how pytest was invoked.

**`client` passes `raise_server_exceptions=False`.**
The default re-raises an unhandled error into the test, so the response body is never seen - and the response body is the thing worth asserting about, because the error-shape and no-filesystem-paths-in-errors guarantees are about what a real client receives. With the flag, an unhandled error is delivered as the 500 it would be in production.

**`client` is used as a context manager.**
That is what runs the app's `lifespan`, which is where the bundle is loaded, the explainer is built, and the report cache is warmed. A `TestClient` constructed without the `with` block gets an app whose `state.bundle` was never set, and every endpoint answers 503 - which reads as a service bug rather than a test bug.

**`trained_run` is session-scoped and read-only.**
Refitting per test would add a couple of seconds each to a suite that runs in about a minute. The contract is that nothing mutates the run directory; a test that needs to mutate one fits its own. Nothing enforces it, which is the honest limitation - a test that wrote into it would corrupt every later test in an order-dependent way.

**`SMALL_ROWS = 900` is a balance, and both sides matter.**
Small enough to keep the default run fast, large enough that a tri-split still leaves positives in every partition. Below roughly that, the validation partition can end up with too few positives for the calibrator's isotonic path, and the failure surfaces as a metric assertion rather than as "the fixture is too small".

**`applicant` is in the extract's own string dialect, on purpose.**
`" 36 months"`, `"5 years"`, `"62.5%"`, `"Jun-2015"`, `"Aug-2003"` - strings rather than numbers, because that is the harder case and the one a client copying values out of a Lending Club CSV actually sends. A fixture of clean floats would leave every parser in `CanonicalizeFrame` unexercised through the API.

This is also the fixture that documents the difference from the dashboard's `EXAMPLE_APPLICANT`, which is in the *control's* dialect - `36`, `62.5`, `"2015-06"` - because a `<select>` and a `type="month"` input cannot hold these strings. Two dialects, both valid, and [score.md](score.md) says why.

**`raw_csv_factory` numbers its files.**
`loans_1.csv`, `loans_2.csv`. Two calls in one test would otherwise overwrite one path, and the second read would silently get the first frame's successor - a test comparing two generator configurations that actually compared one to itself.

**Every fixture's docstring says why it exists, not what it returns.**
The return type is in the annotation. The docstrings here carry the decisions, which is the project's comment convention applied to test scaffolding.

## What must NOT live here

- **Assertions.** This file builds inputs. A fixture that asserts turns a data problem into a mysterious error in whatever test happened to request it.
- **Hand-written dataframes.** The generator is the one source. A literal frame here is the copy-paste this file was written to delete.
- **`os.environ` reads or writes.** Settings are constructed explicitly; a test needing a specific env var sets it with `monkeypatch` in its own module, where the scope is visible.
- **Anything the whole suite does not share.** A fixture used by one module belongs in that module. This file is on the import path of every test, so growth here is a cost everything pays.
- **Network access, or the real `data/` directory.** Nothing here reads the 1.19 GB extract; the `-m slow` tests that do take their path as an argument.
- **`pytest.importorskip("xgboost")`.** See above. `requires_xgboost` is the correct guard and the wrong one silently converts a skip into a failure.

## Related tests

Every test file, which is the point - `test_features.py`, `test_schema.py`, `test_data_loading.py`, `test_feature_engineering.py`, `test_leakage_check.py`, `test_transformers.py`, `test_config.py`, `test_modeling.py`, `test_evaluation.py`, `test_calibration.py`, `test_artifacts.py`, `test_pipeline.py`, `test_cache.py`, `test_logging_setup.py`, `test_cli.py`, `test_explain.py`, `test_drift.py`, `test_reporting.py`, `test_api.py`, `test_routes_admin.py`, `test_jobs.py`, and `test_bench.py`.

The generator these fixtures wrap has its own suite, `tests/test_sample_data.py`, which is where the properties that make the fixtures trustworthy are asserted: that the raw column names and the string dialects are what the real extracts contain, that both the `'13.56%'` and float `int_rate` forms are produced, that the status vocabulary is the real one, that the signal is monotone so a fitted model beats chance, and that immature loans exist for the embargo to remove.

Current state: **676 passed, 3 skipped in 58.80s**. The three skips are `requires_xgboost`, because `libomp` is not installed on this machine - so the real gradient-boosted fits, the `TreeExplainer` path, and early stopping are locally unverified. The ubuntu CI job exists so they run somewhere.

## Known limits

- **`trained_run`'s read-only contract is unenforced.** A test that wrote into the session run directory would break every later test, and the failure would look like a bug in whatever ran next. A copy-per-test fixture would remove the hazard and remove the reason the session scope exists.
- **`SMALL_ROWS = 900` is close to the floor.** A change to the split windows or to the embargo could leave the validation partition without enough positives, and the resulting failure would name a metric rather than the fixture. Raising it costs suite time linearly and has not been needed.
- **One `applicant`, not a set.** No fixture for an applicant that should be rejected, one with every optional field absent, or one at a boundary. Each test that needs a variant builds it by spreading this dict, which works and means the variants are not enumerated anywhere.
- **`raw_loans` is regenerated per test.** `make_synthetic_loans` at 900 rows is fast enough that this has never mattered, and a session-scoped frame would have to be treated as read-only by every consumer that currently mutates a copy freely.
- **No fixture for a bundle-less service.** The tests that need one build their own `Settings` with an empty reports directory. Three modules do it, slightly differently.
- **The three skips are invisible in a green run.** `pytest -q` reports `3 skipped` and nothing says which capability is unverified. A `-W error` style gate, or a CI job that fails on any skip, would be the enforcement.
