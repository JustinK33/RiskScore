# `src/risk_score/feature_engineering.py`

## Purpose

Parse raw values according to their declared `ParseKind`, and build the derived features.

Every function is a pure Series function: it takes a frame, reads the columns it needs, and returns one Series.
`build_feature_matrix` is the only function that returns a frame, and it does exactly one `assign` and one `drop`.

## Public API

Parsers:

| Name | Handles |
| --- | --- |
| `coerce_numeric(series)` | Any numeric column. Strips `,` `$` `%`, drops sentinels and infinities, returns plain `float64`. |
| `parse_percent(series, ...)` | `'13.56%'` or `54.3` to `0.1356` / `0.543`. |
| `parse_term_months(series)` | `' 36 months'` to `36.0`. |
| `parse_employment_years(series)` | `'10+ years'` to 10, `'< 1 year'` to 0. |
| `parse_column(series, kind, ...)` | Dispatch on the registry's declared kind. |

Builders, one per entry in `ENGINEERED_FEATURES`:

`build_dti_clean`, `build_credit_utilization`, `build_loan_to_income_ratio`, `build_credit_history_months`, `build_fico_midpoint`, `build_fico_band`.

Registry and driver: `FEATURE_BUILDERS`, `build_feature_matrix(loans, require_all=False)`.

Constants: `MISSING_SENTINELS`, `MAX_CREDIT_UTILIZATION`, `MAX_PLAUSIBLE_DTI`, `MIN_ROWS_FOR_SANITY_CHECK`.

## Inputs and outputs

In: a canonicalized frame - names already resolved, date columns already `datetime64`.

Out: Series of `float64`, except `build_fico_band` which returns an ordered `Categorical` because it exists to be read by a human.

`build_feature_matrix` returns a frame with the derived columns added and the consumed raw columns removed.

## Invariants and failure modes

### Parsing is by declaration, never by sample

The registry says a column is `PERCENT`; the divisor is 100, always.
Inferring the scale from the data would let a batch of low-utilization applicants be read as already-fractional and scored a hundred times too low, and worse, it would let a one-row `/predict` request - which has no distribution to inspect - disagree with training on the same applicant.

Where a sanity check genuinely needs a sample it *raises* rather than switching behaviour: if a percent column's maximum after dividing is under 2%, the source was already fractional and the error says to change the column's `ParseKind`.
That check is skipped below `MIN_ROWS_FOR_SANITY_CHECK` (100 rows), because firing on a single unusual applicant would reject a valid request.

### Audit B01: derived features drop what they replace

Every old `add_*` function appended a cleaned column and left the raw source in place, so `dti` and `dti_clean` both reached the preprocessor - which routed by runtime dtype, sending the raw percent string to `OneHotEncoder`.

`ENGINEERED_FEATURES` declares `consumes` per feature and `build_feature_matrix` honours it.
`loan_to_income_ratio` deliberately consumes nothing: loan size and income each carry signal beyond their ratio.

`credit_history_months` is the other half of B01.
It derives whole months from `earliest_cr_line` to `issue_d`, replacing a date string that was previously one-hot encoded into 655 columns - discarding an ordered quantity to produce noise.

### Audit P02: one copy, not seven

Seven full copies of the frame produced four columns: each `add_*` copied, and `build_feature_matrix` copied on entry.
Builders now return Series and `build_feature_matrix` leaves the input frame untouched.

Order inside `build_feature_matrix` is load-bearing and commented as such: features are built into a dict first, and the drop happens only after every builder has run, so a feature can consume a column a later feature also reads.
The frame does grow as it goes, because `fico_band` reads the `fico_midpoint` built in the same pass - that is the one place the declared build order matters.

### Audit B24: plain `float64` only

`Int64` and `Float64` propagate through arithmetic into scikit-learn, which casts them to `object` and then fails deep inside a transformer with a message that does not name the column.

### Audit B20: utilization is a fraction, and it is bounded

The old code's two branches disagreed on units: `revol_util` was divided by 100 and `total_credit_utilized / total_credit_limit` was not, so the feature's meaning depended on which extract was loaded.
Both paths now produce a fraction.
Negative utilization is impossible and becomes missing; genuine over-limit values (real `revol_util` runs past 800%) are winsorized at `MAX_CREDIT_UTILIZATION` = 2.0 rather than dropped, because the risk signal saturates long before that and a single 8.9 dominates a scaled coefficient.
A zero credit limit is "no revolving account", not infinite utilization.

### Audit B21: negative income, not just zero

The old guard was `annual_income.replace(0, np.nan)`.
The extract contains negative incomes, which produced a negative ratio the model read as a very low risk of default.
The guard is now `annual_income.where(annual_income > 0)`.

### Skipped features are named

The two real extracts carry different columns - `fico_range_*` exists in neither, `revol_util` in only one - so a feature whose inputs are absent is skipped rather than raising.
Skipped names are logged, because a run that built four of six features must not look identical in the metrics to one that built all six.
`require_all=True` turns the skip into a `KeyError`, which is what the training pipeline wants once the extract is known.

### Import-time consistency

`FEATURE_BUILDERS` and `ENGINEERED_FEATURES` must have identical keys, checked at import.
A declared feature with no builder, or a builder nobody declared, fails immediately.

## What must NOT live here

- **Fitted state.** No means, no medians, no category vocabularies. Every function here is a stateless transformation of one row's own values, which is what makes train and serve identical and makes leakage impossible by construction. Anything fitted belongs in the `ColumnTransformer`.
- **Date parsing.** `schema.py` owns it, with one explicit whole-column format. `parse_column` raises rather than accepting `MONTH_DATE`.
- **Imputation.** Guards here turn impossible values into `NaN`; the pipeline's imputer decides what to do about it.
- **Feature selection.** `leakage_check.py` decides what is admitted.

## Related tests

`tests/test_feature_engineering.py`.

Named audit regressions: `test_b01_build_feature_matrix_drops_the_raw_columns_it_replaced`, `test_b20_credit_utilization_is_a_fraction_from_either_source_column`, `test_b20_credit_utilization_is_bounded`, `test_b21_loan_to_income_ratio_rejects_zero_and_negative_income`, `test_b24_coerce_numeric_returns_plain_float64_not_a_nullable_dtype`, `test_p02_build_feature_matrix_leaves_the_input_frame_untouched`.

## Known limits

- `MAX_CREDIT_UTILIZATION = 2.0` and `MAX_PLAUSIBLE_DTI = 1000.0` are judgements about where signal stops and outliers start. Both are named constants with their reasoning in a comment, so they are arguable rather than hidden, but neither is derived from a measurement.
- `MISSING_SENTINELS` is applied to every numeric column. A column where -1 is a legitimate value would lose it. None of the currently registered columns has one, and a real counterexample should move the sentinel set onto `ColumnSpec` rather than widen this constant.
- `parse_employment_years` maps `'10+ years'` to 10, which is censored in the source. Nothing can recover the true value; the ceiling belongs to the data.
- `build_fico_band` bins a continuous score, which throws away information a linear model can already use. It exists to make a reason code readable, not to improve accuracy, and it is declared `numeric=False` so nothing treats it as a measurement.
