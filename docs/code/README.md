# Per-file reference

One page per code file.
Every page uses the same seven headings, so you can skim to the one you need:

**Purpose** - why the file exists, and what went wrong without it.
**Public API** - what other modules may call.
**Inputs and outputs** - what it reads and returns, and what it deliberately does not touch.
**Invariants and failure modes** - the guarantees, and the audit bugs each one closes.
**What must NOT live here** - the boundary, stated so a future change lands in the right file.
**Related tests** - where the proof is, including the named audit regressions.
**Known limits** - what it does not do, and what would have to change.

`scripts/check_docs.py` enforces in CI that every in-scope code file has a page and every page carries all seven headings.
Documentation that can rot silently does.

## The data spine

Read these in order; each one hands its output to the next.

| File | Page | One line |
| --- | --- | --- |
| `src/risk_score/features.py` | [features.md](features.md) | The canonical column registry. One `ColumnSpec` per column: aliases, parse rule, feature tier. |
| `src/risk_score/schema.py` | [schema.md](schema.md) | Raw column names and date formats to canonical ones, with a report. |
| `src/risk_score/data_loading.py` | [data_loading.md](data_loading.md) | Projected reads, the closed-status filter, the outcome-maturity embargo, the label. |
| `src/risk_score/feature_engineering.py` | [feature_engineering.md](feature_engineering.md) | Declared-kind parsers and the derived features, as pure Series functions. |
| `src/risk_score/leakage_check.py` | [leakage_check.md](leakage_check.md) | The allow-list audit: what reached the model, and why nothing else did. |
| `src/risk_score/transformers.py` | [transformers.md](transformers.md) | `FeatureSpec` and the two in-Pipeline steps that enforce it. No dtype inference. |

## The model

| File | Page | One line |
| --- | --- | --- |
| `src/risk_score/config.py` | [config.md](config.md) | One validated run configuration. Unknown keys raise instead of becoming defaults. |
| `src/risk_score/modeling.py` | [modeling.md](modeling.md) | The tri-split, the declared preprocessor, and the trainers. No dtype inference, no fit on validation. |

## Measuring and deciding

| File | Page | One line |
| --- | --- | --- |
| `src/risk_score/evaluation.py` | [evaluation.md](evaluation.md) | The reported metrics, and the cost-sensitive threshold search behind a partition guard. |
| `src/risk_score/calibration.py` | [calibration.md](calibration.md) | The reliability curve with sample counts, the ECE, and the correction that is now actually applied. |

## Artifacts and orchestration

| File | Page | One line |
| --- | --- | --- |
| `src/risk_score/artifacts.py` | [artifacts.md](artifacts.md) | The `ScoringBundle`, atomic run directories, the registry, and the active pointer. |
| `src/risk_score/pipeline.py` | [pipeline.md](pipeline.md) | The sequence, and nothing else. Where the four hardest bugs lived. |
| `src/risk_score/cache.py` | [cache.md](cache.md) | The canonicalized extract as content-addressed parquet. Every failure falls back to the CSV. |
| `src/risk_score/logging_setup.py` | [logging_setup.md](logging_setup.md) | One `dictConfig`, UTC timestamps, and run/request ids in contextvars. |
| `src/risk_score/cli.py` | [cli.md](cli.md) | The `riskscore` commands. Tracebacks propagate; exit codes distinguish a typo from a bug. |

## Explaining and monitoring

| File | Page | One line |
| --- | --- | --- |
| `src/risk_score/explain.py` | [explain.md](explain.md) | Exact SHAP without the `shap` package. Reason codes that sum back to the model's own score. |
| `src/risk_score/drift.py` | [drift.md](drift.md) | PSI on the score and on every feature, and the per-vintage breakdown. Missingness is a bin. |
| `src/risk_score/reporting.py` | [reporting.md](reporting.md) | The model card, the variant comparison, and the shapes the dashboard reads. Computes nothing. |

## The service

`api -> risk_score`, and never back.
Every file here may import the library; nothing in the library imports these, which is what keeps a training-only install working with none of the serve dependencies present.

| File | Page | One line |
| --- | --- | --- |
| `src/risk_score/api/settings.py` | [settings.md](settings.md) | One validated settings object. Every default is the safe one, and an unsafe combination refuses to boot. |
| `src/risk_score/api/app.py` | [app.md](app.md) | The factory, the middleware order, the one error shape, and the bundle load. |
| `src/risk_score/api/schemas.py` | [schemas.md](schemas.md) | Every request and response body. The applicant model is generated from the bundle's own `FeatureSpec`. |
| `src/risk_score/api/deps.py` | [deps.md](deps.md) | The four things a handler may ask for, and the constant-time key check. |
| `src/risk_score/api/scoring.py` | [scoring.md](scoring.md) | One applicant to a probability and a reason. Where the latency budget is declared. |
| `src/risk_score/api/reports.py` | [reports.md](reports.md) | Fingerprint-keyed payload cache, ETags, and the artifact allowlist that excludes the pickle. |
| `src/risk_score/api/routes_public.py` | [routes_public.md](routes_public.md) | Everything an unauthenticated caller can reach, in one readable file. |
| `src/risk_score/api/routes_admin.py` | [routes_admin.md](routes_admin.md) | The whole mutating surface, off by default and behind a key. |
| `src/risk_score/api/jobs.py` | [jobs.md](jobs.md) | A retrain in a killable child process, one slot, and content-addressed uploads. |
| `src/risk_score/api/bench.py` | [bench.md](bench.md) | Nearest-rank latency percentiles, measured in process, checked against the budget. |

`src/risk_score/api/__init__.py` has no page, on the same grounds as `src/risk_score/__init__.py`: it states the dependency direction above and re-exports two names.
Its one load-bearing consequence - that importing it pulls fastapi in transitively, so `cli.py` must import it inside a handler - is documented where it constrains code, in [cli.md](cli.md) and [bench.md](bench.md).

## The dashboard

No npm, no bundler, no framework, no charting library - see [ADR 0009](../decisions/0009-vanilla-dashboard.md).
The layering is enforced by import direction and readable from the import lines: `format` and `charts` and `dom` import nothing, `panels` imports those three, `main` imports everything and is the only file that fetches.

| File | Page | One line |
| --- | --- | --- |
| `dashboard/js/format.js` | [format.md](format.md) | Absent is not zero. Every displayed number goes through here. |
| `dashboard/js/dom.js` | [dom.md](dom.md) | The node primitives and the one table renderer. No `innerHTML` anywhere. |
| `dashboard/js/charts.js` | [charts.md](charts.md) | Ticks, scales, measured text, and the theme reaching the canvas. |
| `dashboard/js/api.js` | [api.md](api.md) | One error shape, one cache, one timeout. Why a resize costs nothing. |

The `*.test.js` files have no pages of their own; each module's page names its tests under **Related tests**.

## Fixtures

| File | Page | One line |
| --- | --- | --- |
| `src/risk_score/sample_data.py` | [sample_data.md](sample_data.md) | Synthetic raw-format extract that reproduces survivorship bias by the real mechanism. |

## Reading the audit IDs

Bugs found in the audit that prompted this work are numbered `B01`-`B32` (correctness), `S01`-`S06` (security), and `P01`-`P05` (performance).
Bugs found later continue the same sequence; `B33` is the first of those.

Each ID appears at the code that fixes it - a `# Fix B12: ...` comment, or an `(audit B12)` note in the module docstring where the file's whole shape is the fix - and in a test named `test_b12_*` that fails if the fix is reverted.
So `rg B12` shows you the fix and its proof and nothing else.
See [../conventions.md](../conventions.md#audit-id-tags).

The pages above name the IDs each file closes, under **Invariants and failure modes**.
