# `src/risk_score/modeling.py`

## Purpose

Cut the data into time partitions, and assemble the estimator that consumes them.

Two things live here because they are the same decision seen from two ends: *who is allowed to see which rows*, and *what the model is allowed to learn from them*.

The file replaced a two-way `time_based_train_test_json` split and a `build_preprocessor` that inferred its own column routing.
Both were wrong in ways that made the run look healthier than it was.

**The split had no validation partition.**
So the decision threshold was chosen on the same rows the headline metrics came from (audit B04).
That does not make the metrics slightly optimistic; it makes them a description of the selection procedure.
Choosing the cheapest of 99 thresholds on a set and then reporting that set's cost is reporting a minimum, not a performance.

**The split deleted the date column.**
`issue_d` was dropped from both partitions, which reads as tidy hygiene and was load-bearing damage: it is the input `credit_history_months` is derived from.
Keeping the calendar away from the estimator is the `FeatureSpec`'s job - the date is simply not a declared feature - not the split's.
So the split hands the date through and lets the declared contract decide what the model sees.
This is the difference between procedural leakage control (someone remembered to delete a column) and structural leakage control (the column is not in the contract, so there is no path by which it arrives).

**The preprocessor asked the data what each column was.**
`select_dtypes(include=["number", "bool"])` means "categorical" is defined as "whatever pandas failed to parse".
See [transformers.md](transformers.md) for the full account and the 35 GB arithmetic; the consequence here is that `build_preprocessor` takes a `FeatureSpec` rather than a DataFrame, and therefore cannot see the data at all.

Ordering inside the assembled pipeline is the whole design:

```
CanonicalizeFrame -> EngineerFeatures -> ColumnTransformer -> estimator
|------------- stateless -------------| |----- fitted on train only -----|
```

Everything left of the bar learns nothing, so it may run on any partition without carrying information between them.
Everything right of the bar is fitted, and is fitted exactly once, on train.

The two decisions behind this file are recorded separately: [0002 - feature engineering inside the Pipeline](../decisions/0002-feature-engineering-inside-the-pipeline.md) and [0003 - train / validation / test](../decisions/0003-train-validation-test-split.md).

## Public API

| Name | What it does |
| --- | --- |
| `TimeWindow` | A named, inclusive date range. `parse` expands partial dates; `mask` selects rows; `label` prints it. |
| `TimeSplit` | The three partitions plus a full account of every row that reached none of them. |
| `split_by_time(features, target, *, train, validation, test, date_column)` | Partition by origination date. |
| `build_preprocessor(spec, *, min_category_frequency)` | The `ColumnTransformer`, built from declared column lists. |
| `build_model_pipeline(spec, estimator, *, min_category_frequency)` | The four-step `Pipeline`: raw frame in, probability out. |
| `train_logistic_regression(x_train, y_train, *, spec, config, ...)` | Fit the baseline on training rows only. |
| `fit_with_validation_monitoring(spec, estimator, x_train, y_train, x_validation, y_validation, ...)` | Fit an estimator that watches validation, and return one `Pipeline`. |
| `train_xgboost_model(x_train, y_train, *, spec, x_validation=None, y_validation=None, ...)` | Fit the boosted model, with early stopping when validation is supplied. |
| `train_model(model_type, split, *, spec, config, ...)` | Dispatch: give each model type what it may legitimately use. |
| `SUPPORTED_MODEL_TYPES` | The valid `model_type` values, as names rather than functions, so a caller can validate before doing work. |
| `DEFAULT_LOGISTIC_PARAMS`, `DEFAULT_XGBOOST_PARAMS`, `DEFAULT_MIN_CATEGORY_FREQUENCY`, `DEFAULT_EARLY_STOPPING_ROUNDS` | The tested defaults, each with its provenance in a comment. |

## Inputs and outputs

`split_by_time` takes the whole canonicalized frame - not a feature-selected one - and an aligned target, and returns a `TimeSplit`.
`TimeSplit` carries `x_train`/`x_validation`/`x_test`, the three matching target Series, the three resolved `TimeWindow`s, and three counts: `rows_in`, `rows_unparseable_date`, `rows_outside_windows`.
`rows_out` and `summary()` are derived from those.

The trainers take DataFrames and return a fitted `sklearn.pipeline.Pipeline`.
That pipeline accepts a *raw* frame - source column names, percent strings, `' 36 months'` - because its first two steps do the canonicalization.
This is what makes the persisted artifact self-contained: `POST /predict` hands it a one-row frame built from a JSON body and gets the same preprocessing the fit used, because it is the same objects.

Nothing here reads a file, writes a file, or logs.
The counts exist so the caller can put them in a manifest; this module does not decide where they go.

## Invariants and failure modes

### Windows are inclusive at both ends, and the end expands to the period it names

`"2014-09"` means through 30 September, not 1 September.
Writing a window as `2013-01` to `2014-09` is how anyone actually thinks about loan vintages, and the previous half-open interval silently dropped a month of them.

`pd.Period` does the expansion: `'2014'` becomes 31 December, `'2014-09'` becomes 30 September, a full date becomes the end of that day.
Hand-rolled month arithmetic here is where off-by-one-month split bugs come from.

One consequence worth knowing before writing a test: `pd.Period(...).end_time` has **microsecond** resolution, so `end` is `2014-09-30 23:59:59.999999999`.
Assert window membership by containment, never by literal timestamp equality.

### Overlapping windows are rejected

Overlap is the one split mistake that never surfaces as an error.
The model simply scores rows it was fitted on, every metric improves, and the run looks like a success.
So `_reject_overlapping_windows` requires the three windows to be disjoint *and* chronological, and the message names both windows and both boundaries.

### Rows that reach no partition are counted, not discarded in silence

A run that quietly dropped a third of its rows into a gap between windows produces metrics indistinguishable from one that dropped none (audit B09).
`rows_outside_windows` and `rows_unparseable_date` make that visible, and `summary()` puts both in one line for the log and the manifest.

### The date column must arrive already parsed

`split_by_time` raises `TypeError` if it is not datetime, rather than parsing it.
Parsing here would apply a second, different date-format policy to the one `schema.py` already applied (audit B28), and the two would disagree on exactly the ambiguous cases that matter.

### No helper column is created

The old version assigned a `_split_date` column and dropped it afterwards, which destroyed any real column of that name (audit B32).
Masks are computed straight off the date Series; not creating a column at all is both shorter and impossible to get wrong.

### Rows are ordered by date within each partition

`selected.sort_values(kind="stable").index` on the *masked* dates, not `argsort` on the whole column.
`Series.argsort` returns `-1` for `NaT`, which would place unparseable rows at position -1 - the last row - rather than excluding them.
Sorting the masked subset keeps `NaT` out of the ordering entirely.

Ordering matters because XGBoost's `eval_set` and any per-vintage breakdown should see rows in the order they were originated.

### An empty partition is an error that explains itself

The message states what was requested, what the data actually covers, and the row count of each partition.
`_describe_range` handles the all-`NaT` case, because the old error handler called `.min().date()` unconditionally and `NaT.date()` raises - so a run whose dates all failed to parse, by far the likeliest reason for an empty partition, died *inside* the message that was supposed to explain it (audit B08).

### `class_weight="balanced"` is gone

It was the root cause of the miscalibration (audit B05).
Balancing multiplies the minority-class weight by roughly 1/base-rate and therefore inflates every predicted probability - and this project's headline outputs are a Brier score and a calibration curve.
The old artifact's calibration plot was not measuring a miscalibrated model, it was measuring a deliberately reweighted one.

Rebalancing changes the intercept, not the ranking, so AUC and KS are unaffected.
Class imbalance is handled by the cost-sensitive threshold search instead, which is the mechanism that can actually express "a missed default costs five times a declined good loan".

`scale_pos_weight` is absent from the XGBoost defaults for the same reason, and additionally because an LR-versus-XGBoost comparison is only meaningful if both models are weighted identically.

### Validation monitoring transforms, and never fits, the validation partition

`Pipeline.fit` has nowhere to carry a transformed eval set: passing raw validation rows through `classifier__eval_set` would hand XGBoost a DataFrame of strings.
So `fit_with_validation_monitoring` does it in the open:

1. Build the full pipeline, take `steps[:-1]` as the preprocessing prefix.
2. `fit_transform` the prefix on train.
3. **`transform`** - not `fit_transform` - validation through it. This is the line the whole no-leakage claim rests on, which is why it sits one line below the one above.
4. Fit the estimator on both matrices with `eval_set=`.
5. Reassemble a `Pipeline` from the already-fitted steps.

Step 5 is sound because sklearn's `Pipeline` does not clone the steps it is given: the objects in the returned pipeline are the ones just fitted.
A test asserts the assembled pipeline's predictions equal the manual two-step predictions exactly, because "sound" is an argument and that is a measurement.

`early_stopping_rounds` is set only when an eval set is present, since XGBoost raises if asked to stop early with nothing to measure.

### Dispatch is per model type, deliberately not uniform

`train_model` branches rather than calling a common trainer signature.
A shared `(x_train, y_train, x_validation, y_validation)` signature would make the logistic baseline accept a validation partition it then ignores, which reads like an oversight and invites someone to "fix" it by fitting on it.
Only the model that has a use for validation is handed it.

### The XGBoost import error is rewritten

`except Exception`, not `except ImportError`.
An installed-but-unloadable wheel raises `XGBoostError` from inside the import - a forty-line `dlopen` dump whose actual instruction, `brew install libomp`, is buried in the middle of it.
Catching only `ImportError` let that reach the terminal unedited.

The same distinction bites in tests: `pytest.importorskip("xgboost")` does not catch `XGBoostError`, so a test using it *fails* on a machine without OpenMP instead of skipping.
`tests/conftest.py` provides a `requires_xgboost` marker that tries the import and checks the result.

### Preprocessor details that are decisions, not defaults

- `SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)`. The indicator because missingness in credit data is informative - a borrower with no `revol_util` has no revolving account, which is not a borrower at the median. `keep_empty_features` because otherwise a column entirely missing in train is silently dropped and the design matrix is narrower than the spec declares, which surfaces only at serving time as a width mismatch with no column name in the message.
- `StandardScaler` after the imputer, which standardizes the missing indicators too. Harmless for both estimators, and it keeps the stage to one declaration.
- `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=True)`. Unseen categories join the infrequent bucket rather than raising or encoding as an all-zero row, because a new `addr_state` in a serving request is ordinary and `handle_unknown="ignore"` would silently mean "no state". `min_frequency=0.005` caps `addr_state` at roughly the 30 states with enough volume to estimate a coefficient from.
- `remainder="drop"`, with no passthrough: `EngineerFeatures` already reduced the frame to exactly the declared features, and passthrough would quietly readmit whatever a future change forgot to drop.
- `verbose_feature_names_out=False`, so a reason code says `loan_amnt` rather than `numeric__loan_amnt`. Names collide only if a numeric column is named like a one-hot level, and `ColumnTransformer` raises rather than colliding silently.
- `sparse_threshold=0.3` is sklearn's default, stated because it decides the memory layout. The stacked result is CSR only when the *combined* density falls below it, so a wide one-hot block gives sparse output and a narrow one gives dense - and dense is genuinely cheaper there. Declared routing is what made both regimes acceptable; with 2400 inferred one-hot columns the dense branch was the 35 GB failure.

## What must NOT live here

- **Row filtering.** No status filter, no embargo, no deduplication. Those run before the split, on the whole frame, so all three partitions share one outcome definition. This module only *partitions* what it is given.
- **Which columns are features.** That is `features.py` and `transformers.py`. `build_preprocessor` takes a spec precisely so it has no opinion.
- **Anything fitted on validation or test other than the estimator's own early-stopping monitor.** The calibrator and the threshold are fitted on validation, and they live in `calibration.py` and `evaluation.py` where the partition-tagged guard type can enforce it.
- **Metric computation.** `evaluation.py`.
- **Persistence.** No `joblib.dump`, no manifest, no paths. The pipeline object is returned and the caller decides what becomes of it.
- **Reading configuration.** Windows and hyperparameters arrive as arguments. `config.py` turns a YAML file into those arguments; this module never opens one.

## Related tests

`tests/test_modeling.py`.

Named audit regressions: `test_b05_the_logistic_baseline_is_not_class_weighted`, `test_b08_an_empty_partition_reports_the_observed_date_range`, `test_b08_an_all_missing_date_column_explains_itself_instead_of_crashing`, `test_b09_rows_falling_between_windows_are_counted_not_silently_dropped`, `test_b09_an_unparseable_date_is_quarantined_rather_than_aborting_the_run`, `test_b22_the_one_hot_block_is_sparse_and_the_width_is_the_declared_one`, `test_b32_a_column_named_split_date_is_left_alone`.

Audit B28 - re-parsing dates under a second policy - is covered by `test_an_unparsed_date_column_is_refused_rather_than_reparsed_here`.

The validation-monitoring tests are the ones to read first, because they verify an assembly this project invented rather than a library behaviour:

- `test_the_estimator_receives_a_transformed_validation_matrix` - the eval set is a matrix of the same width as the training matrix, and `verbose` is `False`.
- `test_the_assembled_pipeline_predicts_what_the_manual_two_step_path_predicts` - exact equality, which is the claim that reassembly is not a re-fit.
- `test_the_validation_partition_never_reaches_the_preprocessor_fit` - the fitted scaler's mean is the *train* mean (15,000 on the fixture); it would be 43,333 if validation had leaked in.

Those three carry **no** skip marker, on purpose: the logic they cover is this project's and must be verified on every machine.
Only `test_xgboost_stops_early_on_the_validation_partition` and `test_xgboost_without_a_validation_partition_uses_its_whole_budget` are gated on `requires_xgboost`, because only they need a real booster.

`tests/test_pipeline.py` covers the split end to end, including `test_the_three_partitions_are_disjoint_and_chronological` and `test_b04_the_threshold_is_selected_on_validation_not_on_test`.

## Known limits

- **XGBoost is unverifiable without OpenMP.** On a machine where the wheel cannot load, two tests skip and the real early-stopping path has never executed. The assembly around it is stub-verified; XGBoost's own parameter handling is not.
- **No cross-validation.** A single time-ordered split is the right shape for a vintage-structured dataset, but it means every reported number carries the sampling noise of one test period. Per-vintage breakdowns (Phase 5) are the intended partial answer.
- **`min_category_frequency` is one number for every categorical column.** 0.5% is defensible for `addr_state` and arbitrary for `home_ownership`, which has four levels and needs no pooling at all.
- **The split does not check that each partition contains both classes.** An empty partition raises; a partition with zero defaults does not, and the failure surfaces later as an sklearn error about a single class. The sanity banner in the dashboard covers the "technically fits, means nothing" case.
- **`fit_with_validation_monitoring` assumes the estimator accepts `eval_set` and `verbose`.** That is XGBoost's signature, not a sklearn protocol; LightGBM would need different keyword names.
- **Boosting rounds are not recorded in the artifact yet.** `best_iteration` is on the fitted estimator, and until the bundle manifest lands there is nowhere durable that says how many rounds a given run actually used.
