# Runbook

Operating this thing: install it, train a model, serve it, roll it back, and read the reports when they say something is wrong.

Every command here is real and copy-pasteable.
Nothing in this file needs the 1.19 GB Lending Club download - `riskscore make-sample-data` produces a synthetic extract with the same shape and the same survivorship bias, which is what the demo and CI both use.

## Install

Three extras, because a training box and a serving box need different things.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'          # everything, for working on it
pip install -e '.[train]'        # xgboost, matplotlib, pyarrow - fits and reports
pip install -e '.[serve]'        # fastapi, uvicorn, pydantic-settings - scores
```

`[dev]` includes both of the others.
The split is not cosmetic: `[serve]` pulls in no matplotlib and no xgboost, and `[train]` pulls in no web framework, which is why `cli.py` imports `risk_score.api` inside the `serve` and `bench` handlers rather than at the top of the file.

**XGBoost needs OpenMP.**
On macOS that is `brew install libomp`; without it `import xgboost` fails on a missing `libomp.dylib` and every XGBoost test skips.
The Linux wheel bundles what it needs, so CI does not have this problem.

## Train a model

```bash
riskscore make-sample-data                        # data/sample/loans.csv, 8000 rows, seeded
riskscore train data/sample/loans.csv
```

That publishes a run under `reports/runs/<run_id>/`, appends it to `reports/registry.json`, and points `reports/active_run.json` at it.
The summary on stdout leads with row counts at every filter stage, before any metric:

```
rows   raw=2000  closed=1551  mature=1193  in_scope_terms=1072  labelled=1072  train=629  validation=193  test=250
```

Read that line first, every time.
A 0.65 AUC on 250 test rows and a 0.65 AUC on 250,000 are not the same claim, and the metric alone does not say which one you are looking at.
`mature` is the outcome-maturity embargo doing its work: `closed - mature` is the loans dropped for not having had time to default.

Useful flags:

| Flag | When |
| --- | --- |
| `--model xgboost` | The other estimator. Needs OpenMP. |
| `--no-activate` | Publish and register without repointing the service. What a comparison run wants. |
| `--include-lender-priced` | Admit `int_rate`/`grade`/`sub_grade`/`installment`. See [decisions/0005](decisions/0005-lender-priced-feature-tier.md) before you do. |
| `--config configs/run.yaml` | Change the windows, the snapshot date, or the cost matrix. The only way to change them. |
| `--no-cache` | Re-parse the CSV instead of reading `data/cache`. |
| `--keep 20` | Retention. The active run is never pruned regardless of age. |

Training on the real extract is the same command with a different path:

```bash
riskscore train data/raw/1/loan.csv
```

The first run parses 1.19 GB and caches the canonicalized frame as parquet keyed by the input's content hash, so the second run starts in seconds.
Every cache failure falls back to reading the CSV, so a corrupt cache is slow rather than fatal.

## Serve it

```bash
riskscore serve                      # 127.0.0.1:8000
open http://127.0.0.1:8000/          # the dashboard
open http://127.0.0.1:8000/docs      # the OpenAPI browser
```

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "loan_amnt": 12000, "term": " 36 months", "annual_inc": 65000, "dti": 18.2,
  "emp_length": "10+ years", "home_ownership": "MORTGAGE", "purpose": "debt_consolidation",
  "addr_state": "CA", "open_acc": 9, "revol_util": "42.3%", "pub_rec": 0,
  "delinq_2yrs": 0, "inq_last_6mths": 1, "earliest_cr_line": "Aug-2003",
  "issue_d": "Jan-2015", "verification_status": "Verified"
}' | jq
```

`GET /api/schema` is the authoritative list of accepted fields for the *loaded bundle* - it is generated from the bundle's own `FeatureSpec`, so it changes when the model does.
Unknown fields are refused rather than ignored, because a silently dropped typo returns a different applicant's risk.

Configuration is `RISKSCORE_*` environment variables and `.env`; the two things `serve` takes on the command line (`--host`, `--port`) default to `None`, so an unmentioned flag leaves the environment in charge.
A misspelled `RISKSCORE_` variable is a startup error, not a silently ignored line.

| Variable | Default | Notes |
| --- | --- | --- |
| `RISKSCORE_HOST` | `127.0.0.1` | Anything else needs `RISKSCORE_ALLOW_PUBLIC_BIND=1` as well. |
| `RISKSCORE_PORT` | `8000` | |
| `RISKSCORE_ALLOW_PUBLIC_BIND` | `0` | Two decisions to reach the network, not one typo. |
| `RISKSCORE_ALLOWED_HOSTS` | `localhost,127.0.0.1,testserver` | `Host` header allow-list. Guards DNS rebinding. Comma-separated. |
| `RISKSCORE_API_KEY` | unset | Required in `X-API-Key` on every mutating route. Unset means those routes answer 503. |
| `RISKSCORE_ALLOW_UPLOAD` | `0` | Enables `POST /api/datasets`. |
| `RISKSCORE_ALLOW_RETRAIN` | `0` | Enables `POST /api/runs`. |
| `RISKSCORE_REPORTS_DIR` | `reports` | The run tree to serve from. |
| `RISKSCORE_REQUIRE_BUNDLE` | `0` | Refuse to start without a loadable bundle. On in a container. |
| `RISKSCORE_DOCS_ENABLED` | `1` | Worth turning off in a public deployment. |
| `RISKSCORE_MAX_BODY_BYTES` | `1048576` | 1 MiB. `/api/datasets` has its own larger cap. |
| `RISKSCORE_MAX_UPLOAD_BYTES` | `67108864` | 64 MiB. |
| `RISKSCORE_MAX_BATCH_ROWS` | `1000` | |
| `RISKSCORE_JOB_TIMEOUT_SECONDS` | `1200` | After this a retrain's process is killed and the slot released. |
| `RISKSCORE_LOG_LEVEL` / `RISKSCORE_LOG_JSON` | `INFO` / `0` | |

**Put the key in the environment, never in a file that gets committed.**
`.env` is gitignored; that is a safety net, not the plan.

### Exposing it

```bash
RISKSCORE_HOST=0.0.0.0 \
RISKSCORE_ALLOW_PUBLIC_BIND=1 \
RISKSCORE_ALLOWED_HOSTS=risk.internal.example \
RISKSCORE_API_KEY="$(openssl rand -hex 32)" \
RISKSCORE_DOCS_ENABLED=0 \
riskscore serve
```

`Settings` is constructed before uvicorn is imported, so a refused combination exits 3 with a readable message and never opens a socket.
Enabling a mutating route on a non-loopback bind with no key is one of the combinations it refuses.

### In a container

```bash
docker build -t riskscore .
docker run --rm -p 8000:8000 -v "$PWD/reports:/app/reports:ro" riskscore
```

The image serves and does nothing else.
It installs the `[serve]` extra from `requirements-serve.txt`, which pins exact versions so two builds of one commit are the same artifact - `pyproject.toml` keeps floors, because a library that pins is a library nobody can install alongside anything.
Regenerate the pins with the recipe in the file's own header after changing the extra.

**The reports tree is a mount, not a layer.**
A model bundle is generated output that changes every retrain; baking one in would make the image the thing you rebuild to roll a model back, and `riskscore activate` is that thing.
Read-only is correct: a scoring service writes nothing.

Four defaults are inverted inside the image, and the reasons matter:

| Setting | In the image | Why |
| --- | --- | --- |
| `RISKSCORE_HOST` | `0.0.0.0` + `ALLOW_PUBLIC_BIND=1` | A container that binds loopback is a container nothing can reach. Safe here only because the mutating routes stay off. |
| `RISKSCORE_REQUIRE_BUNDLE` | `1` | A fresh clone should boot and be told to train. A container that cannot score should fail its health check and be replaced, not answer `/healthz` while every `/predict` returns 503. |
| `RISKSCORE_LOG_JSON` | `1` | The audience is a log aggregator, not a terminal. |
| `RISKSCORE_ALLOWED_HOSTS` | unchanged | Left at `localhost,127.0.0.1` so `docker run -p 8000:8000` works from a browser. **Behind a proxy or a real hostname, set it to that hostname** - the `Host` allow-list is what stops DNS rebinding, and `*` gives it up. |

The `HEALTHCHECK` polls `/readyz` and parses the body for `bundle_loaded` rather than trusting a 200, because a process serving no bundle is alive and useless.
It runs `python`, not `curl`, so the image needs no extra package, and it reads `RISKSCORE_PORT` itself so overriding the port does not silently break the check.

```bash
docker inspect -f '{{.State.Health.Status}}' <container>
```

**The image cannot serve an XGBoost bundle.**
Unpickling one imports xgboost, which is the `[train]` extra and is not installed - stated plainly because the failure is a `ModuleNotFoundError` at startup, not a wrong answer.
Serving a compared XGBoost run means adding `xgboost` to `requirements-serve.txt` and `libgomp1` to an `apt-get` line in the runtime stage, at roughly double the image size.
The default bundle is logistic regression.

It runs as uid 10001, so a bind-mounted `reports/` has to be world-readable - which it is by default, since the pipeline writes 0644 files.
A Docker *named* volume works because the Dockerfile chowns `/app/reports` before the volume covers it; a volume inheriting root ownership is otherwise the first thing that goes wrong.

The image is built and exercised in CI, in the same job that publishes a synthetic run: it starts the container against that tree, waits on the image's own health check, and then runs `scripts/smoke_e2e.py --against` so the container faces the same assertions the local process does.
That is the only place it is built - see [code/smoke_e2e.md](code/smoke_e2e.md).

## Is it healthy

| Probe | Means | On a missing bundle |
| --- | --- | --- |
| `GET /healthz` | The process is alive. | `200`, `status: "degraded"`, and the load error. |
| `GET /readyz` | A bundle is loaded and it can score. | `503`. |

They are deliberately different.
A liveness probe that fails when the model is missing makes an orchestrator restart a process that would have come up in exactly the same state, forever.
`/readyz` is the one a load balancer and a container `HEALTHCHECK` should use.

```bash
curl -s localhost:8000/healthz | jq       # status, run_id, error
curl -si localhost:8000/readyz | head -1
```

Logs go to stderr with a `run=`/`req=` context on every record; `RISKSCORE_LOG_JSON=1` if something downstream is parsing them.
`X-Request-ID` is on every response including errors, which is how "it returned 500" is joined to the traceback in the log.
Error bodies are two fields, `detail` and `request_id` - no paths, no date ranges, no tracebacks.

## Roll back a bad model

Runs are immutable, so reverting is repointing rather than retraining.

```bash
riskscore runs                       # newest first, active marked with *
riskscore activate 20260824T183320563Z-logistic_regression-origination_only-02e3a94
```

`activate` loads the bundle and throws it away before repointing, so a run whose pickle is unreadable is refused here - on your terminal - rather than at the service's next restart, where it is an outage.
A failed activation leaves serving exactly where it was.

The service picks up the new pointer on restart, or immediately if the change came from a retrain through the API.
`riskscore serve` reads `active_run.json` once at startup.

If `registry.json` is lost or truncated, the manifests on disk are the truth:

```bash
riskscore runs --rebuild
```

That reconstructs the index by walking `reports/runs/*/manifest.json`, which also handles runs copied in from another machine.

## Compare models, and quantify the leakage cost

```bash
riskscore compare data/sample/loans.csv --tiers both
```

Fits every model x tier combination on one identical split, publishes each as an ordinary run, and writes `comparison.json` at the report root.
Nothing is activated: choosing what to serve is a decision, not a side effect of measuring.

`--tiers both` is the measurement that makes the tier policy defensible rather than asserted - it reports what admitting the lender's own price adds to the AUC.
Expect it to add a lot.
That is the point: a model that needs `int_rate` cannot score an applicant nobody has priced yet.

## Explain one decision

```bash
riskscore explain data/sample/loans.csv --row 0 --top-k 5
riskscore explain data/sample/loans.csv --row 0 --json | jq
```

The contributions are exact SHAP values in log-odds and they sum to the model's own score, which is what makes them usable in an adverse action notice rather than merely suggestive.
One-hot families are collapsed back to their source feature, so a reason code names `home_ownership` rather than `cat__home_ownership_RENT`.

`POST /predict` returns the same thing per request; `?explain=false` is the cheaper path, and roughly a third of the latency.

## Reading a PSI table

`GET /api/drift`, or `psi_features.csv` and `psi_score.csv` in the run directory.

**Reference is train, comparison is test.**
The question is whether the rows the model is *used* on still look like the rows it was fitted on.

| PSI | Band | What to do |
| --- | --- | --- |
| `< 0.10` | `stable` | Nothing. At these sample sizes this is noise. |
| `0.10` - `0.25` | `moderate` | Look at the per-bin table and find which end moved. |
| `> 0.25` | `significant` | The reference population is not the population being scored. Retrain on a window that includes it, or explain why the shift is expected. |

The bands are industry convention, not a derivation, which is why they are reported as bands: 0.099 and 0.101 are the same finding and a pass/fail column would present them as different ones.

Two columns to read alongside the number.
`buckets` is how many bins the feature got - a 0.02 over four bins is a weaker test than a 0.02 over ten.
`missing_rate_reference` and `missing_rate_comparison` are separate columns because missingness is itself a bin: a feature that stopped being collected shows up here as a large PSI driven entirely by the `__missing__` bucket.

When the score PSI is the one that moved, `psi_score.csv` is the per-bin working - a scalar is a sum, and a sum does not say which end of the distribution shifted.

**The known one:** `term` reads as significant on the default configuration, and that is documented rather than a bug.
With the embargo applied, 60-month loans only exist through 2013Q4, so a train window that includes them and a test window that cannot is a real term-mix cliff.
It is why the default config restricts to 36-month loans, and the PSI table is where that restriction stays a measured decision instead of a magic number.

## Retrain through the API

Off by default, and it should usually stay that way.
It is remote-triggered execution over caller-supplied data that ends by writing a pickle the service will later load; see [decisions/0008](decisions/0008-retraining-in-a-child-process.md).

```bash
export RISKSCORE_API_KEY="$(openssl rand -hex 32)"
RISKSCORE_ALLOW_UPLOAD=1 RISKSCORE_ALLOW_RETRAIN=1 riskscore serve
```

```bash
# 1. store the extract. content-addressed, so the id is its hash.
DATASET=$(curl -s -X POST localhost:8000/api/datasets \
  -H "X-API-Key: $RISKSCORE_API_KEY" --data-binary @data/sample/loans.csv | jq -r .dataset_id)

# 2. start the fit. 202 immediately; the fit runs in a child process.
JOB=$(curl -s -X POST localhost:8000/api/runs \
  -H "X-API-Key: $RISKSCORE_API_KEY" -H 'content-type: application/json' \
  -d "{\"dataset_id\": \"$DATASET\", \"model_type\": \"logistic_regression\"}" | jq -r .job_id)

# 3. poll.
curl -s localhost:8000/api/jobs/$JOB -H "X-API-Key: $RISKSCORE_API_KEY" | jq
```

`/predict` keeps serving the old bundle throughout and swaps to the new one only after the child exits successfully.

| You see | It means |
| --- | --- |
| `403` naming a variable | The feature is switched off. The flag is checked before the key, deliberately. |
| `503` on a mutating route | The feature is on but `RISKSCORE_API_KEY` is unset. |
| `401` | Wrong or missing `X-API-Key`. |
| `409` with `Retry-After: 5` | A retrain is already running. One slot, and a refusal rather than a queue. |
| `status: "failed"` with a `detail` | The fit raised. `detail` is the child's exception text. |
| `status: "failed"`, `"exited without reporting"`, `exit_code` set | The child died before it could report. An OOM kill looks exactly like this. |
| `status: "timed_out"` | It hit `RISKSCORE_JOB_TIMEOUT_SECONDS` and was killed. |
| `404` on a job id | Unknown, or evicted from the 32-entry history. `riskscore runs` is the durable record. |

### The same thing from the dashboard

The retrain panel drives those three routes, so the flags above are what makes it appear at all.

It requires **both** `RISKSCORE_ALLOW_UPLOAD=1` and `RISKSCORE_ALLOW_RETRAIN=1`, not either.
The browser has no server-side `dataset_id` to name, so uploading is the only way it can produce one, and a retrain route with uploads off is reachable from the CLI and not from here.
A form whose every submission returns 403 reads as a broken feature rather than a switched-off one, so with only one flag set the panel stays hidden.

Then, in the page:

1. Paste the API key into the key field and submit it once. It goes to `sessionStorage`, covers the upload, the start and every poll, and does not outlive the tab.
2. Choose a CSV and a model. The model list comes from the service, not from the markup, so it cannot offer a model the service would reject.
3. Submit. The panel polls `/api/jobs/{id}` and reports the stage; the page reloads onto the new run when the child exits successfully.

Everything the panel can tell you is in the table above - it renders the same `detail` string.
If the panel is absent while the routes work from `curl`, check `/readyz`: `mutating_routes` has to read `both`.

Uploads accumulate under `RISKSCORE_DATASETS_DIR` (`data/uploads`) and nothing removes them.
Delete them by hand; the content-addressed name means a re-upload just recreates the one you need.

## Measure the latency

```bash
riskscore bench                 # active run, 2000 calls per configuration
riskscore bench --json | jq
```

In-process: no HTTP, no JSON serialization, no socket, so what is reported is the part this project controls.
The budget lives in `scoring.py` and this command checks it - p50 <= 12 ms / p99 <= 25 ms without reason codes, p50 <= 25 ms / p99 <= 50 ms with them.
Preprocessing is the request; the model itself is under 0.1 ms.

If the numbers regressed, the thing to suspect is a transformer, not the estimator.

## When something is wrong

| Symptom | Cause | Fix |
| --- | --- | --- |
| `riskscore serve` exits 3, "is not loopback" | `RISKSCORE_HOST` is public without the flag | Set `RISKSCORE_ALLOW_PUBLIC_BIND=1` deliberately, and set a key first. |
| `riskscore serve` exits 3, "extra inputs are not permitted" | A misspelled `RISKSCORE_` variable | Fix the name. This is the check working. |
| `/readyz` is 503 and `/healthz` says `degraded` | No active run, or its pickle will not load | `riskscore runs`, then `riskscore activate` a good one, or `riskscore train`. |
| Every `/api/...` report is 503 | Same cause. The detail names `riskscore train`. | As above. |
| `/api/comparison` is 404 | No `comparison.json` at the report root | `riskscore compare`. A plain `train` run does not write one. |
| `ImportError` mentioning `libomp` / `libxgboost` | OpenMP is missing | `brew install libomp`, or use `--model logistic_regression`. |
| `ValueError` naming a version in `load_bundle` | The bundle was written by a different `BUNDLE_SCHEMA_VERSION` | Retrain. Bundles are not migrated. |
| `ModuleNotFoundError: risk_score.transformers` on load | The module was renamed or moved | Put it back. Pickle couples every bundle to these paths forever; see [decisions/0007](decisions/0007-the-scoring-bundle.md). |
| A `.staging-*` directory in `reports/runs/` | A run crashed mid-publish | Safe to delete. It was never reachable by name. |
| Retrain always 409s and the job it names finished long ago | Would mean the watcher thread died with the slot held | Restart. The broad handler in `_watch` exists to make this unreachable; if it happens, that is a bug worth a traceback. |
| A run reports an implausible metric on a tiny `test=` count | Too few rows to measure anything | Read the row-count line, or `rows` in `manifest.json`. An AUC well below 0.5 next to a near-perfect KS is the signature of inverted labels on a handful of rows. |

## Verifying a change

```bash
ruff format --check . && ruff check . && mypy src tests scripts && pytest -q
python scripts/check_docs.py                     # every code file has a page, every page has all seven headings
(cd dashboard && node --test)                    # the JS helpers
python scripts/smoke_e2e.py                      # the whole path on synthetic data, ~1 minute
```

The first two lines are the gate every commit passes.
The last two are what CI adds, and both run locally with no arguments.

`pytest -m slow` additionally runs the latency guard, which is excluded from the default run because a percentile measured on a loaded machine is flaky by construction.
XGBoost tests skip rather than fail when OpenMP is absent, which is why the Linux CI job is the one that proves those paths work.

Two checks need something this repository cannot assume.
`node scripts/probe_dashboard.mjs http://127.0.0.1:8000` measures layout at six widths in two themes and needs Chrome, so it is run by hand - see [code/probe_dashboard.md](code/probe_dashboard.md).
The image build needs Docker and only ever runs in CI.
