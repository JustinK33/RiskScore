# `src/risk_score/transformers.py`

## Purpose

Declare which columns the model sees and how each one is typed, then enforce that declaration as two steps *inside* the scikit-learn `Pipeline`.

This file exists because the previous design inferred the answer.
`build_preprocessor` chose a transformer per column by inspecting its runtime dtype:

```python
numeric_columns = features.select_dtypes(include=["number", "bool"]).columns
```

So anything pandas had failed to parse was, by definition, categorical.
`earliest_cr_line` is a date string with 655 distinct values in the real extract.
`int_rate` and `revol_util` arrive as percent strings in the `loans_full_schema` extract with roughly 600 and 1100 distinct values.
All three went to `OneHotEncoder`, and with `sparse_output=False` that is about 2400 dense float columns over 1.8M rows - roughly 35 GB.
The pipeline was not slow on the real dataset; it could not run on it at all, and the monotone rate information was destroyed on the way (audit B01, B22, B23).

The fix is not a better dtype check.
It is to delete the inference, so that a string column reaching `OneHotEncoder` unintentionally stops being a bug that has been fixed and becomes a state the code cannot represent.

Why these steps live inside the `Pipeline` rather than ahead of it - and what that costs - is [decisions/0002-feature-engineering-inside-the-pipeline.md](../decisions/0002-feature-engineering-inside-the-pipeline.md).

## Public API

| Name | What it does |
| --- | --- |
| `FeatureSpec` | Frozen contract: the numeric list, the categorical list, the raw inputs, the derived features, the tier setting, and the pinned date formats. |
| `build_feature_spec(available, ...)` | Derive that contract from the columns one extract actually supplies. |
| `CanonicalizeFrame(spec)` | Raw extract or request payload to the declared inputs, correctly typed. |
| `EngineerFeatures(spec)` | The declared inputs to exactly `spec.model_features`. |
| `SPEC_VERSION`, `MISSING_CATEGORY` | Bundle compatibility marker; the level missing categoricals become. |

`FeatureSpec` members worth knowing: `model_features` (numeric then categorical, which is the frame layout `EngineerFeatures` emits), `engineered_specs`, `parse_kinds()`, `summary()`.

## Inputs and outputs

`build_feature_spec` takes an iterable of column names - not a frame, so it is testable without one - and returns a `FeatureSpec`.

`CanonicalizeFrame.transform` takes a DataFrame with *source* names in *source* formats and returns one with canonical names, `spec.raw_inputs` as its exact column list and order, `float64` for every numeric kind, `datetime64` for every date, and `string` for every category.

`EngineerFeatures.transform` takes that and returns exactly `spec.model_features`: `float64` numerics and `str` categoricals with no missing values, because missing became `MISSING_CATEGORY`.

Neither step writes anything, reads anything from disk, or logs.

## Invariants and failure modes

### Nothing is fitted, and that is the leakage argument

Every function these steps call is a stateless transformation of one row's own values - no means, no medians, no category vocabularies.
That is what makes it safe to put them ahead of the `ColumnTransformer` and therefore ahead of any partition boundary: a transformer with no fitted state cannot carry information from train into validation.
Anything fitted belongs in the `ColumnTransformer`, and the boundary is stated again under **What must NOT live here**.

`fitted_` is set in `fit` for one mechanical reason: `check_is_fitted` treats an estimator with no trailing-underscore attribute as unfitted, so a `Pipeline` ending in such a step reports itself unfitted after `fit`.
Nothing reads it.

### The spec is data-dependent at exactly one point

`revol_util` exists in one real extract and not the other.
`fico_range_low` and `fico_range_high` are absent from **both** - and `configs/feature_config.yaml` listed `fico_range_low` as a model feature while being loaded by nothing at all, which is how that went unnoticed.

So a hard-coded feature list either names columns that do not exist or omits ones that do.
`build_feature_spec` resolves it once, against the columns in front of it.
Everything downstream treats the result as fixed, and the bundle persists it, so serving cannot re-derive a different answer from a one-row request that happens to be missing a column.

### One resolver for which derived features exist

`resolvable_engineered_features` in `features.py` does the walk, and both `build_feature_spec` and `build_feature_matrix` call it.
Two separate walks would eventually disagree, and the symptom would be a served row preprocessed into a different feature set from the one the model was fitted on - which shows up as bad predictions, not as an error.

### Audit B20, again: parsing has exactly one owner

The first version of `CanonicalizeFrame` parsed `revol_util` by its declared `PERCENT` kind, and then `build_credit_utilization` parsed it *again*.
`54.3%` became `0.00543`.

`PERCENT` is the one non-idempotent parse rule, so the rule is now explicit: `parse_declared_columns` in `feature_engineering.py` owns the conversion, runs exactly once, and every builder downstream assumes its inputs are already in declared units.
`build_credit_utilization` carries a backstop for the other direction - a `revol_util` column reaching 20 is still in percentage units, and it raises rather than clipping, because clipping would silently record every such borrower at the winsorization ceiling.

This was caught by the `parse_percent` sanity check that already existed, which is the argument for guards that raise instead of guards that branch.

### Reindex before parse

`CanonicalizeFrame` reindexes to `spec.raw_inputs` *before* casting anything.
Parsing 145 columns in order to discard 115 of them is most of the work for none of the result.

`reindex` rather than `frame[cols]` because it creates an absent optional column as all-NA, which `[]` would raise on - and an all-NA column then parses to `float64` NaN or `NaT` without a special case.

### Missing categoricals become a level

Not an imputed mode.
In credit data missingness is informative: an applicant who did not state `emp_length` is not an applicant with the most common `emp_length`.
It also removes the categorical imputer from the preprocessor entirely, and gives the reason codes something readable to name.

### A column cannot be both numeric and categorical

`FeatureSpec.__post_init__` rejects it, because such a column is transformed twice and appears twice in the design matrix, quietly doubling its weight.
The same check rejects a spec with no features at all, a `required_raw_inputs` entry outside `raw_inputs`, and an engineered feature nobody declared.

### Errors name the column

A missing required input lists the names and the twenty columns that *were* present after alias resolution.
A non-DataFrame input says so explicitly rather than failing later on `.columns`, because these steps route by name and a bare array has none.

## What must NOT live here

- **Anything fitted.** Means, medians, category vocabularies, scalers, encoders. They belong in the `ColumnTransformer`, where the partition boundary is enforced. This is the invariant the whole file rests on.
- **The column lists themselves.** They come from `features.py`. This module holds the *shape* of the contract, not its membership.
- **Date-format policy.** `schema.py` owns it. `FeatureSpec.date_formats` pins the formats into the bundle; it does not choose them.
- **Row filtering.** No embargo, no status filter, no split. Those run before the pipeline, on the whole frame, and are recorded in the manifest.
- **This file must not move.** A pickled bundle stores the import path of every object inside it. Renaming this module or either class breaks every bundle ever written. `SPEC_VERSION` covers changes to the *meaning* of a field; nothing covers a moved class.

## Related tests

`tests/test_transformers.py`.

Named audit regressions: `test_b01_no_date_column_is_ever_a_model_feature`, `test_b01_canonicalize_drops_columns_outside_the_declared_inputs`, `test_b01_engineer_drops_the_raw_sources_its_outputs_replace`, `test_b01_no_object_dtype_survives_into_the_numeric_features`, `test_b02_canonicalize_resolves_a_duplicate_source_by_alias_priority`, `test_b22_the_declared_routing_collapses_the_one_hot_width`, `test_b23_every_declared_categorical_is_low_cardinality`.

`test_b22_*` is the one that measures rather than asserts: on 400 synthetic rows the old `select_dtypes` path would have produced thousands of one-hot columns against fewer than 100 now, and the test fails if that ratio collapses.

`test_one_row_scored_through_the_frame_steps_matches_the_batch` is the serving guarantee: row 5 transformed alone must equal row 5 of the batch, which is the property that would break first if anything here became sample-dependent.

## Known limits

- **Pickle couples bundles to this module path forever.** That is the price of putting feature engineering inside the `Pipeline` and it is not mitigable, only managed - see the never-move rule above.
- `build_feature_spec` decides from column *names*. An extract that supplies `revol_util` as an entirely empty column produces a `credit_utilization` feature that is always NaN, and only the imputer notices.
- `MISSING_CATEGORY` is a magic string. An extract containing the literal value `__missing__` would merge genuine and missing levels. No real extract does, and the alternative - a sentinel object - does not survive one-hot encoding.
- `CanonicalizeFrame` reads the registry at transform time, not from the spec, for the parse kind of each column. Editing a `ColumnSpec.parse` therefore changes how an *existing* bundle canonicalizes. `SPEC_VERSION` does not currently catch that; pinning parse kinds into the spec would, and is the obvious next hardening step.
- Neither transformer reports what it dropped. The counts that matter (rows removed, columns refused) are produced upstream by `SchemaReport` and `LeakageAudit`; a column silently absent from a serving request is caught only if it is required.
