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

## Fixtures

| File | Page | One line |
| --- | --- | --- |
| `src/risk_score/sample_data.py` | [sample_data.md](sample_data.md) | Synthetic raw-format extract that reproduces survivorship bias by the real mechanism. |

## Reading the audit IDs

Bugs found in the audit that prompted this work are numbered `B01`-`B32` (correctness), `S01`-`S06` (security), and `P01`-`P05` (performance).

Each ID appears at the code that fixes it - a `# Fix B12: ...` comment, or an `(audit B12)` note in the module docstring where the file's whole shape is the fix - and in a test named `test_b12_*` that fails if the fix is reverted.
So `rg B12` shows you the fix and its proof and nothing else.
See [../conventions.md](../conventions.md#audit-id-tags).

The pages above name the IDs each file closes, under **Invariants and failure modes**.
