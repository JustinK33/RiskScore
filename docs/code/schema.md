# `src/risk_score/schema.py`

## Purpose

Turn a raw extract's column names and date formats into canonical ones, and report what it took.

This is the first code that touches the data, so a mistake here is invisible everywhere downstream: it shows up as a crash three modules later, or as a metric that is quietly wrong.
Two audit bugs lived here.

## Public API

| Name | What it does |
| --- | --- |
| `SchemaReport` | Frozen record of the normalization: renames, dropped duplicates, unknown columns, missing required columns, date formats used, unparseable date counts. Goes into the run manifest. |
| `resolve_column_names(columns, ...)` | Pure name resolution. Returns `(rename_map, dropped, unknown)`. Takes an iterable of names, so it can be tested without a DataFrame. |
| `normalize_column_names(loans, ...)` | Applies the resolution. `drop_unknown=True` keeps only registry columns. |
| `parse_month_column(series, ...)` | Parses one date column with one explicit whole-column format. Returns `(parsed, format_used)`. |
| `normalize_credit_schema(loans, ...)` | The entry point: canonicalize names, then parse every registered `MONTH_DATE` column. |
| `DEFAULT_DATE_FORMATS` | Eight unambiguous formats, in priority order. |
| `MIN_DATE_PARSE_RATE` | 0.98. |

## Inputs and outputs

In: a DataFrame with whatever the extract called its columns, and optionally an alias mapping from `configs/run.yaml`.

Out: a DataFrame with canonical names and real `datetime64` date columns, plus a `SchemaReport`.

No file I/O.
`data_loading.py` reads the file; this module only transforms what it is handed, which is what lets the same code path serve a single-row scoring request.

## Invariants and failure modes

### Audit B02: no duplicate column names

Resolution was "first source column in the frame wins", skipping any canonical name already claimed.
The standard extract has both `funded_amnt` and `loan_amnt`.
`funded_amnt` came first, was renamed to `loan_amnt`, and then the genuine `loan_amnt` was skipped and kept its own name - two columns with the same label.
`loans["loan_amnt"]` then returned a DataFrame, and the run died two modules later inside a numeric coercion with an error about ambiguous truth values.
The same collision existed for `status`/`loan_status` and `state`/`addr_state`.

The contest is now settled by the position of the matched key in `ColumnSpec.match_keys()`, canonical name first.
`loan_amnt` at priority 0 beats `funded_amnt` at priority 3 regardless of frame order.
Losers are dropped and recorded in `report.dropped_duplicate_sources`.

Config-supplied aliases sort after every declared one, so an explicit registry entry outranks a config patch.

### Audit B28: one date format per column, and it is never ambiguous

`pd.to_datetime(..., format="mixed")` infers a format per row.
On a column of `03/04/2016` values it can read some rows as 3 April and others as 4 March - a date column silently wrong for an arbitrary subset of rows, with no error and no warning.
Since `issue_d` decides which partition a loan lands in, that is a split that cannot be reproduced.

Each candidate format is now applied to the whole column and scored.
The first to clear `MIN_DATE_PARSE_RATE` wins.
Slash-separated formats are not candidates at all: they raise, naming the formats tried and five sample values, so the caller states their intent with `date_formats=`.

If no single format clears the floor, the formats are combined best-first.
This is safe precisely because no ambiguous format is a candidate - combining can never flip a day and a month.

### Other guarantees

- A blank or whitespace-only cell is missing data, not a parse failure, so it does not inflate the unparseable count.
- A handful of genuinely corrupt cells are quarantined as `NaT` and counted rather than aborting a 1.8M-row run. Too many (over 2%) do abort.
- Every registered `MONTH_DATE` column is parsed, not just the split column. Previously only `issue_d` was, which is the other half of audit B01.
- Parsed columns are built into a dict and assigned once. Assigning inside the loop would trigger a fragmentation warning and a copy per column under pandas 3 copy-on-write.
- Missing required columns are *reported*, not raised. An audit of an unfamiliar extract wants the list; a training run wants to abort on it, and does so in its own code.

## What must NOT live here

- **File reading.** `data_loading.py` owns that, including the projection.
- **Row filtering.** Statuses, the maturity embargo, and dropna all belong to `data_loading.py`. This module changes names and dtypes, never the row count.
- **Value parsing beyond dates.** Percentages, terms, and employment length are `feature_engineering.py`'s job.
- **A second date-format policy.** `apply_outcome_maturity_embargo` deliberately raises `TypeError` on a string date column rather than parsing it, because two format policies in one pipeline is how B28 got in.

## Related tests

`tests/test_schema.py`.

Named audit regressions: `test_b02_the_canonical_name_beats_an_alias_regardless_of_order`, `test_b02_two_aliases_resolve_by_declared_priority`, `test_b28_ambiguous_slash_dates_raise_instead_of_being_guessed`, `test_b28_one_format_applies_to_the_whole_column`, `test_b01_every_registered_date_column_is_parsed_not_just_the_split_column`.

## Known limits

- Trying eight formats means up to eight passes over a date column before one is accepted. On the full extract with two date columns this is a few seconds, and the first candidate is the one the standard extract uses, so the common case is one pass. Not worth optimizing until it shows up in a profile.
- The 2% corruption floor is a judgement, not a measurement. A column that is 3% corrupt aborts, which is arguably too strict; the error names the count so the caller can widen `date_formats` or fix the source.
- `%b-%y` maps `Aug-03` to 2003, not 1903 or 2103. That is Python's two-digit year pivot and it is correct for consumer lending data, but it is an assumption.
- The combining path reports its formats joined with `+`, which is enough to reproduce the parse but does not say which rows came from which format.
