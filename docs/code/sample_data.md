# `src/risk_score/sample_data.py`

## Purpose

Generate a synthetic Lending Club extract that is realistic in the ways that matter for testing this pipeline, so neither the tests nor the demo needs the 1.19 GB download.

"Realistic in the ways that matter" is a specific claim.
The generator produces raw *source-format* columns - `' 36 months'`, `'13.56%'`, `'10+ years'`, `'Aug-2003'` - because half the audit bugs were parsing bugs that a clean DataFrame fixture cannot reach.
And it reproduces outcome-maturity survivorship bias through the same mechanism as the real data rather than by hard-coding a biased label distribution, which is what makes the embargo tests prove something.

## Public API

| Name | What it does |
| --- | --- |
| `make_synthetic_loans(...)` | Return a raw-format DataFrame of 42 columns. |
| `write_sample_dataset(output_path, **kwargs)` | Write one to CSV. Backs `riskscore make-sample-data`. |
| `STATUS_*` constants | The real status vocabulary, including `'Late (31-120 days)'` and the two "does not meet the credit policy" variants. |

Keyword arguments: `n_rows`, `seed`, `start_month`, `end_month`, `snapshot`, `target_default_rate`, `percent_strings`, `include_fico`, `include_post_origination`, `include_junk_columns`.

## Inputs and outputs

No inputs beyond its arguments; no file reads.
`write_sample_dataset` is the only function that touches disk.

Output is a DataFrame whose columns use raw source names and raw source formats.
It is meant to be fed to `read_raw_loans` / `normalize_credit_schema`, exactly as the real file is.

## Invariants and failure modes

### Determinism

Same `seed`, same frame, byte for byte.
`tests/test_sample_data.py::test_generator_is_deterministic` asserts it, because a flaky fixture produces a flaky test suite and the cause is nearly impossible to find later.

### The default rate is solved for, not typed in

`target_default_rate` is achieved by bisecting the intercept of the linear score over 60 steps, rather than by hand-tuning a constant.
This matters because the coefficients are meant to be edited: adding a feature or changing a weight would otherwise silently move the base rate, and a test asserting "the embargo brings the rate near 15%" would start failing for an unrelated reason.

### Survivorship bias is emergent, not asserted

This is the design decision the whole file exists for.

Each loan gets a default month drawn from `Beta(2, 3)` scaled over its term, so defaults cluster early - as they do in reality.
Status is then assigned by comparing both the maturity date and the default date against the snapshot:

- Defaulted *and* the default month has passed: `Charged Off`.
- Did not default *and* the full term has elapsed: `Fully Paid`.
- Anything else: `Current`.

Filtering to closed statuses therefore inflates the late vintages for the same causal reason it does in the real extract: only early defaults have had time to close.
Measured on 12,000 rows with a true rate of 15%, closed-only vintage rates run 14.5 / 14.3 / 19.2 / 19.7 / 100.0 percent for 2012-2016.
The 2016 vintage reading 100% is the mechanism at its clearest.

`test_closed_only_filtering_reproduces_vintage_survivorship_bias` guards this property, so it cannot silently regress and quietly turn every embargo test into a tautology.

### Weights cannot fail to sum to one

Categorical distributions go through a `_normalized` helper rather than being written to sum to 1.0 by hand.
An early version had `_PURPOSE_WEIGHTS` summing to 1.001, which `rng.choice` rejects outright - the fix is structural so a future edit to any single weight cannot break the generator.

### The signal is learnable and directionally correct

`test_signal_is_learnable_and_directionally_correct` fits a model and asserts both that AUC clears a floor and that the coefficient signs match the generative model.
A shape assertion (`0 <= auc <= 1`) is exactly why the inverted-KS bug survived 25 passing tests, so it is not used here.

### Installments are internally consistent

`installment` is computed from `loan_amnt`, `int_rate`, and `term` with the standard amortization formula, so a model given all four sees the real collinearity - which is the point of the `LENDER_PRICED` tier.

### The default snapshot is `2019-06`

Chosen by measurement, not convenience.
`2018-12` produced a degenerate 100% default rate for the 2016 vintage in the *default* configuration, which is a fine thing for a test to construct deliberately but a bad default for a demo dataset.
`2019-06` yields closed-only rates of 14.7 / 17.0 / 17.7 / 31.8 percent for 2013-2016 against a true lifetime rate of 14.1%, which matches the real extract's shape.

## What must NOT live here

- **Anything the pipeline imports at run time.** This is a test and demo fixture. Nothing in the training or serving path may depend on it.
- **Canonical column names.** The whole value is emitting *raw* names and formats so the alias resolver and the parsers are genuinely exercised.
- **Pre-cleaned values.** A generator that emitted `36` instead of `' 36 months'` would have let every parsing bug through.
- **Test assertions.** They belong in `tests/`.

## Related tests

`tests/test_sample_data.py`, twelve tests.

The load-bearing ones: `test_generator_is_deterministic`, `test_raw_columns_arrive_in_source_format`, `test_signal_is_learnable_and_directionally_correct`, `test_closed_only_filtering_reproduces_vintage_survivorship_bias`, `test_installment_is_consistent_with_rate_amount_and_term`.

## Known limits

- The generative model is linear in the log-odds with independent features. Real credit data has correlated features and non-linear interactions, so a tree model's advantage over logistic regression is understated on synthetic data. The LR-versus-XGBoost comparison is therefore only meaningful on the real extract.
- 42 columns, not 145. The missing ones are post-origination fields and free text that the pipeline refuses anyway; the audit path is exercised, the memory pressure is not.
- Missingness is applied independently per column at fixed rates. Real missingness is correlated - a borrower missing `emp_length` is likelier to be missing `annual_inc` - which means missing-indicator features look less informative here than they are.
- `include_junk_columns=True` currently adds only `funded_amnt`, which is enough to exercise the B02 duplicate-column path but is not a broad unknown-column test. The genuinely-unrecognized-column tests use small hand-built frames instead.
