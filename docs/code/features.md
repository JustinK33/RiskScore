# `src/risk_score/features.py`

## Purpose

The canonical column registry.
One `ColumnSpec` per column, carrying its aliases, its parsing rule, and its feature tier, plus the declaration of every engineered feature and what raw column each one replaces.

This file exists because column knowledge used to live in three places that disagreed.
`schema.py` held aliases with no types and no policy.
`leakage_check.py` held policy with no aliases and no types.
`configs/feature_config.yaml` held numeric and categorical lists and was loaded by nothing at all - it listed `fico_range_low` as a model feature, and neither real extract has that column.

Because nothing declared a type, the preprocessor inferred one at runtime from whatever pandas happened to parse.
That is how `earliest_cr_line`, a date string with 655 distinct values, was routed to `OneHotEncoder`.

## Public API

| Name | What it gives you |
| --- | --- |
| `FeatureTier` | Why a column is or is not allowed to be a feature. `BORROWER`, `LOAN_REQUEST`, `LENDER_PRICED`, `POST_ORIGINATION`, `IDENTIFIER`, `TARGET_SOURCE`, `TIMELINE`. |
| `ParseKind` | How a raw value becomes a usable one. `NUMERIC`, `PERCENT`, `TERM_MONTHS`, `EMP_LENGTH_YEARS`, `MONTH_DATE`, `CATEGORY`, `TEXT`. |
| `ColumnSpec` | Frozen record for one column. `match_keys()` returns its lookup keys, canonical name first. |
| `COLUMN_REGISTRY` | `dict[str, ColumnSpec]`, validated at import. |
| `column_spec(name)` | One spec, with an error that lists near misses. |
| `alias_lookup(extra_aliases=None)` | Lowercase source name to canonical name. Config aliases are validated to the same standard as built-in ones. |
| `alias_priority(canonical)` | The lookup keys in priority order. This is what settles a duplicate-column contest in `schema.py`. |
| `model_feature_columns(include_lender_priced=False)` | Raw columns eligible to be features. |
| `columns_in_tier(tier)` | Every column in one tier. |
| `required_columns()` | Columns whose absence aborts a run. |
| `columns_to_read(...)` | The projection list for `read_csv`. |
| `EngineeredFeature`, `ENGINEERED_FEATURES`, `ENGINEERED_BY_NAME` | Derived features in build order, with `requires` and `consumes`. |

## Inputs and outputs

No I/O, no pandas, no configuration file reads.
It is a pure data structure plus lookups over it, which is why it can be imported from anywhere including the serving path without pulling in scikit-learn.

The only external input is the optional `extra_aliases` mapping, which comes from `configs/dataset_schema.yaml` and lets a new extract be onboarded without a code change.

## Invariants and failure modes

Enforced at import, so a bad edit fails on the first import rather than in a training run:

- No two specs share a canonical name.
- No lookup key is claimed by two canonical columns. An ambiguous registry would otherwise resolve by dict insertion order.
- No alias is shorter than `MIN_ALIAS_LENGTH` (3). This is audit B03: `"n"` was an alias for `loan_status`, so any column named `n` silently became the target.

The length floor applies to aliases only, never to canonical names.
A canonical name is the column's literal name in the source extract - `id` is genuinely two characters - whereas an alias is a guess about what some other extract might have called the same thing, and a two-character guess will eventually match something unrelated.

Checked by tests rather than at import, because they are properties of the declarations taken together:

- Every `EngineeredFeature.requires` names a real registry column or an earlier engineered feature.
- `ENGINEERED_FEATURES` is in build order, so a feature never requires one declared after it.
- `FEATURE_BUILDERS` in `feature_engineering.py` has exactly the same keys - checked at import there.
- No `MONTH_DATE` column appears in `model_feature_columns()` at any tier setting (audit B01).

Failure modes are all `ValueError` or `KeyError` at import or lookup time.
Nothing here can fail at runtime on data.

## What must NOT live here

- **Parsing implementations.** This file says a column is `PERCENT`; `feature_engineering.py` knows what to do about it. Keeping them apart is what lets the registry stay importable with no pandas work.
- **Anything sample-dependent.** A tier or a parse kind that depended on the data would let a training run and a single-applicant scoring request disagree.
- **Column *values*.** No thresholds, no bin edges, no cost matrices.
- **Feature selection results.** Which features a specific run used belongs in that run's manifest.

## Related tests

`tests/test_features.py`.

Named audit regressions: `test_b01_date_columns_are_never_model_features`, `test_b03_short_aliases_are_rejected`, `test_p01_read_projection_is_far_smaller_than_the_raw_extract`.

## Known limits

- `fico_range_low` and `fico_range_high` are registered but absent from both real extracts, so `fico_midpoint` and `fico_band` never build on real data. They are kept because the synthetic generator produces them and the serving schema documents them, but a reader should not expect FICO in a real feature-importance chart.
- The tier of a column is a single global judgement. A column that is post-origination in one extract and origination-time in another cannot be expressed; onboarding such an extract needs a code change, which is deliberate.
- `columns_to_read` ignores its `include_lender_priced` argument on purpose - lender-priced columns are always read so both variants can be fitted from one pass. The argument is retained so callers read symmetrically with `model_feature_columns`.
