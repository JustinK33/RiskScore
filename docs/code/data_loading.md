# `src/risk_score/data_loading.py`

## Purpose

Raw ingestion.
Read the extract, restrict rows to loans whose outcome is genuinely known, and build the binary label.

Everything here happens before a single feature exists.
The row-restriction job is the subtle one, and it is where this project's most defensible correction lives - see [decisions/0004-outcome-maturity-embargo.md](../decisions/0004-outcome-maturity-embargo.md).

## Public API

| Name | What it does |
| --- | --- |
| `read_raw_loans(path, ...)` | Read a CSV or parquet file with the read projected to registered columns, then canonicalize. Returns `(frame, SchemaReport)`. |
| `filter_to_closed_loans(loans, ...)` | Keep terminal statuses. Necessary, not sufficient. |
| `apply_outcome_maturity_embargo(loans, *, snapshot, ...)` | Keep loans whose full term elapsed by the snapshot. Returns an `EmbargoResult`. |
| `filter_to_terms(loans, *, terms, ...)` | Keep only the listed term lengths. An empty `terms` keeps all. |
| `term_months(loans, ...)` | `' 36 months'` to a nullable `36`, for the two functions above. |
| `create_default_target(loans, ...)` | Status to `Int64` label. Returns a Series. |
| `load_lending_club_data(path, ...)` | Read plus status filter, for callers that do not need the report. |
| `EmbargoResult` | The kept frame plus per-vintage rates before and after, and removed-row counts. |
| `CLOSED_LOAN_STATUSES`, `DEFAULT_STATUSES`, `PAID_STATUSES` | `frozenset`s. |

## Inputs and outputs

In: a path to a Lending Club CSV or a parquet file this project wrote, and a snapshot date for the embargo.

Out: DataFrames and a label Series, plus two report objects (`SchemaReport`, `EmbargoResult`) whose contents are destined for the run manifest.

Nothing here has an opinion about the snapshot's value.
`apply_outcome_maturity_embargo` requires it as a keyword argument with no default, because it cannot be inferred from the data - the latest `issue_d` is a lower bound on the snapshot, not the snapshot itself.
The shipped value lives in `config.DEFAULT_SNAPSHOT` and reaches this module as an argument, which is the same separation every other tunable in the project has.

## Invariants and failure modes

### Audit P01: the read is projected

`pd.read_csv(path, low_memory=False)` read all 145 columns of a 1.8M-row file to use about 27 of them, which projects to roughly five times the peak memory - the difference between a run that finishes and one the kernel kills.

`read_raw_loans` reads the header alone with `nrows=0`, resolves it against the registry, and then reads only the matching columns.
The header read is what makes a dtype map possible: the map has to be keyed by the extract's own spelling, which is not knowable in advance.

Text-ish columns are declared as `string` so the C parser never guesses and a percent column cannot be read as float in one chunk and object in the next.
Genuinely numeric columns are deliberately left to inference: declaring `float64` would make the entire read *fail* on a single `n/a` cell, whereas `to_numeric` downstream handles that per value.

A file where nothing matched raises, listing the first ten columns found and pointing at `configs/run.yaml`.
`project=False` reads everything, which is only useful for auditing an unfamiliar extract.

### Audit P02: the label is a Series

`create_default_target` used to copy the entire frame to add one column.
On 1.8M rows that is a gigabyte of churn for one `Int64`.
It now returns a Series and leaves the caller's frame untouched, so the caller decides when to pay for a copy.

### The label maps non-terminal statuses to NA, not 0

`Current` and `Late (31-120 days)` become `pd.NA`.
Calling a still-performing loan "did not default" is the same survivorship mistake the embargo exists to prevent, one level down, and it concentrates the resulting label noise in exactly the recent vintages a time-based test set is made of.

### The embargo

`issue_d + term_months <= snapshot`, evaluated on periods rather than with per-row `DateOffset` arithmetic, which would not be vectorized.

Guards, each of which produces a specific error rather than a pandas internal one:

- `issue_d` must already be `datetime64`. Re-parsing here would mean two different date-format policies in one pipeline, which is how B28 got in. The error names `normalize_credit_schema`.
- The snapshot must be timezone-naive, because issue dates are. Comparing them otherwise raises inside pandas with a message that never mentions the snapshot argument.
- Removing every row raises, and the message states the snapshot and the observed date span. The old split code crashed *inside* its own error handler by calling `.min().date()` on an all-`NaT` column; an error about dates has to survive long enough to name the dates.

Rows whose date or term did not parse are removed and counted as `rows_unknown_maturity`, separately from `rows_immature`.
"Corrupt" and "still running" are different problems with different fixes, and a single combined count would hide whichever is smaller.

### The term filter exists because of the embargo

`filter_to_terms` is not an independent option, and reading it as one leads to turning it off.
Once immature loans are gone, 60-month loans survive only in the earliest vintages - 28% of the 2013 rows in the synthetic extract, 0% of 2015's - so an unrestricted run trains on a term mix that validation and test do not contain.

Two details are decisions rather than defaults:

- An empty `terms` returns the frame **unchanged**, not empty. `term_months_in: []` in a config reads naturally as "no restriction", and an empty allow-list interpreted as a filter would silently empty the run.
- Rows whose term does not parse are dropped, matching the embargo. A loan that cannot be shown to be in scope is not in scope.

### Constants are immutable

`CLOSED_LOAN_STATUSES` is a `frozenset` because it is a default argument value.
As a `set`, a caller who mutated it would have changed the filter for every other caller in the process.

## What must NOT live here

- **Feature construction.** No derived columns, no imputation, no scaling.
- **Column name or date-format decisions.** `schema.py` owns those.
- **The split.** The embargo runs before it and knows nothing about it.
- **Caching.** The parquet cache keyed on input hash belongs in the pipeline, so this module stays a pure function of a path.

## Related tests

`tests/test_data_loading.py`.

Named audit regressions: `test_b02_a_frame_with_both_loan_amnt_and_funded_amnt_yields_one_column`, `test_p01_the_read_is_projected_to_registered_columns`, `test_p02_create_default_target_returns_a_series_not_a_frame_copy`.

`test_the_two_outcome_vocabularies_are_disjoint` pins the relationship between the three status constants rather than any one of them: the two outcome sets do not overlap, both are lowercased for the lookup, and together they cover `CLOSED_LOAN_STATUSES` exactly.
That last equality is the one worth having.
A terminal status in neither outcome set passes the closed-loan filter and is then dropped as an NA label, so the rows disappear between two functions that each count only their own removals.

The embargo's load-bearing test is `test_embargo_removes_the_survivorship_bias_in_a_closed_loan_filter`, which runs against the synthetic generator specifically because the generator reproduces the bias by censoring default timing against a snapshot - the same mechanism as the real data, rather than a hard-coded biased label distribution.

Whether a *run* applies the embargo is tested one level up, in `tests/test_pipeline.py`, because for the whole of this project's history until now the answer was no: the function was written, documented, and tested, and `pipeline.py` did not call it.
`test_the_maturity_embargo_runs_and_flattens_the_vintage_default_rate` and `test_immature_loans_never_reach_any_partition` are the tests that would have caught that.

## Known limits

- The embargo discards censored loans rather than modelling them. A loan 18 months into a 36-month term carries real information that survival analysis could use. That is the right next step and it changes the deliverable from a probability to a hazard function, so it is out of scope here.
- The parquet branch does not project columns. It is only used for this project's own cache files, which are already projected.
- `read_raw_loans` opens the CSV twice. The header read is negligible, but it does mean the file must be seekable - not a stream.
- `term` is parsed here by digit extraction *and* in `feature_engineering.parse_term_months`. The duplication is deliberate: the embargo runs before feature engineering and must not depend on it. Within this module there is one parser, `term_months`, shared by the embargo and the term filter. If a third appears, they should be consolidated.
- **A wrong snapshot is detectable but not detected.** Setting it later than the true pull date readmits immature loans, and the signature is a residual climb in `EmbargoResult.default_rate_after`. Nothing here inspects that; the numbers are published and a human reads them.
