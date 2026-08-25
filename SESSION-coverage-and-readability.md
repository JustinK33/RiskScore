# Coverage and Readability

<!-- 2026-08-25 -->

## TL;DR

Closed out the test-coverage and inline-comment work carried into this session, ratcheted the CI coverage floor from 80 to 95, then verified the documented quickstart end to end so the user could run the project themselves.
Two real defects surfaced during that verification and were fixed: reason codes printed 17-digit floats in the CLI, and the boot log claimed to be serving before the socket was bound.
The suite is green at 704 passed / 3 skipped, 96.6% coverage, zero warnings.
The 3 skips are the XGBoost tests, still blocked on `libomp` not being installed locally.

## Changes

| File | What it does now |
| ---- | ---------------- |
| `src/risk_score/explain.py` | `Contribution.display_value` formats a reason for a human - four places, `__missing__` as "not provided". `sentence()` uses it; the JSON paths keep the raw value. |
| `src/risk_score/cli.py` | The human-facing `explain` output uses `display_value`; `--json` is unchanged. |
| `src/risk_score/api/app.py` | Boot log says "will serve on", since it runs in `create_app` before uvicorn binds. |
| `.github/workflows/ci.yml` | Coverage floor at 95, ratcheted in two steps with the reasoning for the slack recorded. |
| `src/risk_score/data_loading.py` | Comments on the four ingestion lines a reader stops at - the period-offset maturity arithmetic and why `fillna(0)` is a cast requirement, not a maturity decision. |
| `tests/test_explain.py` | Argument guards, the unattributable-column branch, and `display_value` pinned to the same cases as the dashboard's `reasonValue`. |
| `tests/test_modeling.py`, `tests/test_jobs.py`, `tests/test_api.py` | Cover the deliberate error paths, the retrain child in-process, and the scoring degradation path. |
| `tests/test_properties.py` | The KS oracle no longer emits a divide-by-zero warning on a one-row-per-class draw. |
| `tests/test_data_loading.py` | Pins that the two outcome vocabularies are disjoint and cover `CLOSED_LOAN_STATUSES` exactly. |
| `src/risk_score/{features,reporting,modeling}.py`, `api/{deps,bench,routes_public}.py` | Nine comments on idioms that read as arbitrary; `modeling.py`'s unreachable guard marked with why it stays. |
| `docs/code/{explain,jobs,scoring,data_loading}.md`, `docs/runbook.md`, `PROJECT_SUMMARY.md` | Doc pages track the new tests and API; the runbook covers the retrain panel, not just the routes. |

## Metrics

| Measurement | Before | After |
| ----------- | ------ | ----- |
| Line + branch coverage | 96.03% (122 missed) | 96.62% (91 missed) |
| CI `--cov-fail-under` | 80 | 95 |
| pytest warnings | 1 | 0 |
| `modeling.py` / `api/jobs.py` / `explain.py` / `api/scoring.py` | 85 / 91 / 89 / 89 | 91 / 96 / 95 / 99 |

## Notes

- Measured on the active real-data run: `/predict` p50 7.10 ms scoring only, 13.78 ms with reason codes (in-process); 15-21 ms warm over HTTP.
- The comment-density metric that motivated the inline sweep was misleading - `api/deps.py` scored 0/92 because its reasoning is in docstrings and `#:` attribute docs, which a `#`-count cannot see. Nine comments were warranted, not ~80.
- Deferred by decision: `routes_admin.py`'s last 7 uncovered lines are ASGI read-shim branches, awkward to reach for diminishing return.
