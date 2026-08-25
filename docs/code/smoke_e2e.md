# `scripts/smoke_e2e.py`

## Purpose

Walk the documented path with nothing but a synthetic extract, and assert the contract at every step.

```
make-sample-data -> compare -> activate -> card -> serve -> predict
```

This is the **only automated proof that the artifact contract holds**. `reports/` is gitignored, so nothing in the pytest suite ever sees a run tree published by the real CLI - the suite calls `train_run` into a `tmp_path` and asserts on the returned objects. That leaves a specific and expensive gap:

- A filename constant renamed in `artifacts.py` and not in `reports.py`'s allowlist, so an artifact link 404s.
- A report the dashboard fetches that the pipeline quietly stopped writing.
- A card template variable nobody substituted, so the card says `$auc_roc`.
- A `serve` extra that is not actually enough to boot the service.
- A default that offers the mutating routes.

Each is invisible to a unit test and fatal in a clone-and-run demo, which is the first thing anybody does with this repository.

It is also the check that the two clone-and-run commands in the README actually work, on a machine with no dataset and no configuration.

## Public API

A script. Its four module constants are the specification and are meant to be read.

```
python scripts/smoke_e2e.py [--workdir DIR] [--rows N]
python scripts/smoke_e2e.py --against http://127.0.0.1:8000
```

| Name | What it is |
| --- | --- |
| `EXPECTED_ARTIFACTS` | The twelve files the service allowlists, spelled literally. |
| `BUNDLE` | `model.joblib`. Must exist on disk and must never be served. |
| `REPORT_PATHS` | The twelve GETs the dashboard makes. |
| `APPLICANT` | One valid applicant in the raw extract's string dialect. |
| `check(condition, message)` | Record a failure without stopping; return whether it passed. |

`--workdir` keeps the tree instead of using a tempdir it deletes, which is what CI passes so a failed run has something to upload. `--rows` defaults to 4000.

`--against URL` trains nothing and publishes nothing: it runs only the service checks, against a service this script did not start. That is how the container is checked, using the tree the full pass just published - so the image gets `/predict`, the reason-code sum, both 422 shapes, the batch, every allowlisted artifact, the pickle 404, and the dashboard's own files, rather than a `curl /readyz` and a shrug. The run id is read from `/api/model` instead of being asserted against a known one, which makes "is it serving the run I activated?" the single check this mode gives up.

Exits `0` with `end-to-end smoke clean`, or `1` with one `FAIL` line per problem.

## Inputs and outputs

Runs the installed `riskscore` CLI as a subprocess - the real entry point, not an imported function - and speaks to the service over HTTP with `urllib`.

Stdlib only. No pytest, no requests, no httpx. It has to run in an environment installed with `[train,serve]` and nothing else, because a `[dev]` install would make a missing serve dependency undetectable.

Writes into a temporary directory: the synthetic CSV, the parquet cache, the `reports/` tree, and `serve.log`. Nothing in the repository is touched, and no environment variable is read for configuration.

Removes `RISKSCORE_ALLOW_UPLOAD` and `RISKSCORE_ALLOW_RETRAIN` from the child's environment before booting, so the run measures the defaults rather than the developer's shell.

## Invariants and failure modes

**`EXPECTED_ARTIFACTS` is spelled literally, not imported from `risk_score.artifacts`.**
Importing the constants would make this script agree with the code *by construction*, and agreeing with the code is precisely what is not being tested. A rename has to be made here too, and that second edit is the check. The same argument applies to `REPORT_PATHS`.

**Every check runs; nothing stops at the first failure.**
Four broken artifacts should produce four lines in one run. `check` returns a bool only for the few cases whose successors would be meaningless - there is nothing to say about a payload that never arrived.

**`compare`, not `train`.**
It is the only path that publishes `comparison.json` at the report root, and a comparison the service cannot find renders as "nothing has been compared" on a tree where something has. It also gives more than one run, which is what makes the next assertion possible.

**Every variant's artifacts are checked, not just the activated one.**
A report written only for the baseline is a run picker whose other options render as dashes. This is the check that would catch a per-run artifact accidentally written to the root.

**That `compare` leaves `active_run.json` *absent* is asserted, not assumed.**
Nothing is activated by measuring - choosing what to serve is a decision. If that ever regressed the service would boot with an unexpected bundle and every check below would fail for the wrong reason, so the assertion is here where it names the cause.

**`riskscore activate` is a step, and its output file is checked.**
It is also the documented rollback path, so this is the only automated exercise of it.

**The model card is checked for four markers: `$`, `{{`, `None`, and `nan`.**
A template variable that reached the output means the card claims a fact it does not have, which is worse than omitting it. `None` and `nan` are the Python-leaked forms of the same failure. Plus a length floor and the run id, because an empty card also contains no `$`.

**The pickle must exist and must 404.**
Both halves. It is the one file in a run directory that is not on the allowlist, and "the allowlist is closed" is only a claim until something asks for the file it excludes.

**Every allowlisted artifact is fetched over HTTP as well as checked on disk.**
The disk check catches a pipeline that stopped writing; the HTTP check catches an allowlist that stopped serving. They fail for different reasons and neither implies the other.

**`/predict` takes the applicant as the body, with `explain` and `top_k` as query parameters.**
So a raw extract row scores with no wrapper object, which is what makes the endpoint usable from `curl`. Getting this wrong is how the first version of this script failed, with a 422 naming every field.

**The reason codes are asserted to sum back to the model's log-odds.**
`baseline_log_odds + sum(log_odds) == total_log_odds`, within `1e-6`. This is the property that separates exact SHAP from a plausible-looking bar chart, and it is the one thing about an explanation a reader cannot verify by looking. Asked again at `top_k=50` because the five-reason response is truncated on purpose, and only asserted when the response came back *under* the cap - at the cap it would be truncated too, and the check would be asserting something false.

**Both error shapes are checked, and they are different failures.**
A missing required field must be a 422 **naming the field**, because the dashboard's whole error story is the field name being in the response. An unknown field must also be a 422, because a typo silently dropped is a score computed from an imputed value with nothing anywhere to show it.

**`/predict/batch` is checked for one score and one inline error.**
Failing all rows because one is malformed is a worse answer than one score and one message, and the caller cannot tell which row was at fault from a 422 about the body.

**`mutating_routes == "none"` on a default service.**
This is the field the dashboard's retrain panel mounts on, so a default that offered the routes would be both a security regression and a visible UI change. Checked from the response rather than inferred from the flags.

**The dashboard's own files are fetched from the same process.**
`/`, both stylesheets, and `/js/main.js`. That is the sentence in the page's footer and the reason there is no second server.

**`wait_for_ready` polls `/readyz` and checks the child is alive on every pass.**
A service that refused to boot otherwise produces a 60-second timeout instead of the reason it refused. On either failure the server's own log is written to stderr, because the log is the diagnosis.

**A connection error is status `0`, not an exception.**
`wait_for_ready` polls before the listener exists, and a refused connection is what "not yet" looks like. The first version of this script crashed there.

**The response body is decoded with `errors="replace"`.**
One allowlisted artifact is a PNG. Nothing here inspects a figure's bytes, only that it arrived, and a `UnicodeDecodeError` would abort the run on a *passing* check.

**The service is terminated in a `finally`, and a `SIGTERM` it ignores is itself a reported failure.**
A hung child would hang CI.

**A failing subprocess writes its own stdout and stderr through.**
Swallowing them and raising "command failed" is how a CI log becomes useless.

## What must NOT live here

- **Any dependency.** Stdlib only. A `requests` import would make the `[serve]`-only install untestable, which is half the point.
- **`import risk_score`.** It drives the CLI and the HTTP API - the two surfaces a user touches. Importing the library would test a third thing that already has 676 tests.
- **Assertions about whether a number is *right*.** The AUC being correct is `tests/test_evaluation.py`. This checks that it exists, is finite, and reached the reader.
- **A pytest wrapper.** It boots a server, takes about a minute, and wants its output to be a list of failures rather than an assertion diff. Marking it `-m slow` would put it in a suite people run with `-m "not slow"`.
- **Anything that writes into the repository.** Everything goes into a tempdir.
- **XGBoost.** Deliberately `--model logistic_regression` only: the fit is a minute of CI time for a path the test matrix already covers on ubuntu with `libgomp1` installed.

## Related tests

It is a test. Nothing tests it, on the same grounds as [check_docs.md](check_docs.md): a test for a test is where the regress stops.

What gives confidence that it works is that it found four real problems while being built, each on a run rather than on a reading - a `--reports-dir` flag that does not exist (`serve` takes `--output-dir`), a `/predict` body shape that is flat rather than wrapped, six response field names that were guesses, and a PNG that a strict UTF-8 decode aborted on. Every one of those is a class of drift it exists to catch.

The unit-level coverage of everything it walks: `tests/test_cli.py` for each command, `tests/test_artifacts.py` for the staging and the registry, `tests/test_api.py` for the endpoints and the allowlist, `tests/test_explain.py` for the exactness the log-odds sum re-checks over HTTP, `tests/test_reporting.py` for the card template.

`scripts/probe_dashboard.mjs` is the other end-to-end check and covers what this one cannot: layout, both themes, and the generated form in a real browser. This script asserts the dashboard's files are *served*; the probe asserts they *work*.

## Known limits

- **Logistic regression only.** No XGBoost variant is fitted, so the `TreeExplainer` reason-code path and the early-stopping assembly are proven by the pytest matrix and not end to end. Adding `--model xgboost` is one flag and about a minute.
- **The mutating routes are only checked in the off direction.** It never starts a second service with both flags and drives an upload-and-retrain, which is the highest-risk feature in the project. `tests/test_routes_admin.py` and `tests/test_jobs.py` cover it at the unit level, and the full browser path was driven by hand.
- **No latency assertion.** `riskscore bench` is not run, so a catastrophic `/predict` regression would pass. The `-m slow` p99 guard in the suite is the backstop.
- **`APPLICANT` duplicates the `applicant` fixture in `tests/conftest.py`**, kept in step by hand. Importing it would need the test package on the path, which would defeat the stdlib-only rule.
- **The card marker check is a substring scan.** `"None"` would false-positive on a legitimate sentence containing the word, and `"$"` on a dollar amount. Neither appears in the template today, and the failure mode is a false alarm rather than a missed bug.
- **One applicant.** No boundary case, no all-optionals-absent case, no applicant that should be declined - so the `decision` check only ever sees one branch of the threshold.
- **`PORT = 8399` is fixed.** A local run collides with anything already on it, and the failure reads as "the service never became ready".
- **It does not check `/predict` against a second run.** Activating another variant and re-scoring would prove the rollback path changes what is served, which is the claim `activate`'s help text makes.
