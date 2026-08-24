# `src/risk_score/drift.py`

## Purpose

Answer the question that decides whether a published AUC is still worth anything: **are the rows still the same shape as the rows the model was fitted on**.

`evaluation.py` measures how well the model ranked the loans in its test window.
That number is a fact about a population that closed years ago.
The Population Stability Index is the standard credit-risk answer to what happened next:

```
PSI = sum over bins (share_new - share_old) * ln(share_new / share_old)
```

the symmetrized Kullback-Leibler divergence between two binned distributions.
It needs **no labels**, which is the entire point.
A lender learns whether the score has drifted long before it learns whether the loans defaulted, and on a 36-month product that gap is three years.

This project also has a *known* drift to report, which is why the section exists rather than being decoration.
With the outcome-maturity embargo applied, 60-month loans only survive in the earliest vintages, so a split admitting both terms would train on 13.9% 60-month loans and validate on 0.0%.
The default configuration restricts to 36-month terms, and the PSI row for `term` is how that restriction stays a measured decision instead of a comment in a config file.

The second half of the module is the per-vintage breakdown, which is the table this project most needs to publish.
One AUC over a multi-year window hides whether the model works in every year of it, and the per-vintage default rate is where the embargo's correction becomes visible as a flat series rather than the 15.6% -> 24.3% climb the uncorrected data shows.

## Public API

| Name | What it is |
| --- | --- |
| `population_stability_index` | The scalar, for one variable, between two populations. |
| `psi_table` | The per-bin working behind that scalar. Published as `psi_score.csv`. |
| `feature_drift` | One PSI row per model feature, worst first. Published as `psi_features.csv`. |
| `metrics_by_vintage` | Discrimination, default rate, and approval rate per origination year. Published as `metrics_by_vintage.csv`. |
| `psi_band` | `stable` / `moderate` / `significant`. |
| `PSI_BUCKETS`, `PSI_MODERATE`, `PSI_SIGNIFICANT` | 10, 0.10, 0.25. |
| `MISSING_BUCKET`, `UNSEEN_BUCKET` | `__missing__`, `__unseen__`. |

## Inputs and outputs

In: two `Series` for a PSI, two **engineered** frames plus a `FeatureSpec` for `feature_drift`, and a labels/scores/dates triple for the vintage table.
Out: DataFrames. Nothing here reads or writes a file; `pipeline.py` decides where the CSVs go.

Engineered frames, not the design matrix and not the raw extract.
A PSI per one-hot column would report fifty numbers about `addr_state` and none about `addr_state`, and the whole value of the table is that a human reads a row and knows what to do.
`modeling.engineering_prefix` is the one place that slice is defined, shared with `explain.py`.

The run compares **train against test** - the population the model was fitted on against the population it is used on.
Validation sits between them in time, so measuring against it would report a smaller shift than the one that matters.

Measured on a 4,000-row synthetic extract:

```
             feature   kind      psi   band  buckets
          addr_state  categ.  0.0465 stable       19
loan_to_income_ratio numeric  0.0436 stable       11
           revol_bal numeric  0.0414 stable       11

 partition  vintage  rows  default_rate  auc_roc  approval_rate
     train     2013   711        0.1280   0.7464         0.6484
     train     2014   559        0.1306   0.7841         0.7120
validation     2015   205        0.1659   0.7207         0.6683
      test     2015   478        0.1234   0.6102         0.6778
```

The AUC falling from 0.78 in train to 0.61 in test is exactly the finding a single headline number hides, and it is the kind of thing this table exists to make unmissable.

## Invariants and failure modes

**PSI is never negative, and zero only for an identical binning.**
Each bin contributes `(p - q) * ln(p / q)`, which carries the sign of `p - q` twice, so no bin can offset another.
A negative contribution would mean the formula had been implemented as a plain KL divergence, which is asymmetric and *can* cancel across bins.

**A population against itself is exactly `0.0`.**
Not "close to zero" - the assertion in the test uses `==`, because every term is `(p - p) * ln(p / p)`.

**Missingness is a bin.**
A feature going from 2% missing to 40% missing has drifted, however stable its observed values are.
Dropping NaN is the obvious implementation and it would report 0.0 for that feature.
Non-finite numbers count as missing too: an infinity is not a value the reference quantiles can place, and putting it in the top bin would claim it was merely large.

**An unseen category is a bin.**
A level absent from the reference gets `__unseen__`, whose reference share is zero and whose contribution is therefore large.
That is the correct reading: the model's one-hot encoder maps that level to nothing at all.
Levels are compared as strings, so a column read as `object` in one partition and `category` in another does not register as total drift for a reason that is about dtypes rather than about borrowers.

**The outer bin edges are open.**
A comparison row beyond the reference's observed range is the single most interesting thing a drift check can find.
Closed edges would leave it in no bin, shrinking the comparison's total and *understating* the drift - the wrong direction for a monitoring metric to fail in.
The tests assert that every row of both populations is counted.

**Zero-count bins are floored at half an observation, `0.5 / n`, not at an arbitrary epsilon.**
`ln(0)` is what this prevents.
The continuity correction has the useful property that the same empty bin counts for less on a small sample - where an empty bin is not yet surprising - than on a large one.
The floored shares are deliberately *not* renormalized, because spreading the correction back across the bins would change the bins that had data in them.

**Duplicate quantiles collapse, and the table says how many bins it used.**
`pub_rec` is zero for most borrowers, so its 10th through 70th percentiles are all 0 and ten bins are not available.
A reader comparing one feature's 0.02 to another's needs to see that one had four bins and the other had ten, so `buckets` is a published column.

**A constant reference reports `0.0`, by construction.**
There is one bin, `(-inf, inf)`, and every comparison value lands in it.
That is the honest answer rather than a defect: with no spread in the reference there is no distribution to compare against.

**A non-finite PSI is refused rather than banded.**
`nan < 0.10` is `False`, so an unguarded comparison would silently report a NaN as `significant`.

**Vintage dates must share an index with the labels.**
Aligning by position would pair a 2013 loan's date with a 2015 loan's outcome, and every number in the table would still look plausible.
A row with no origination date is dropped and logged at `WARNING` rather than becoming a NaN group that reads like a real cohort - it should be impossible downstream of a time split, which cannot place a row without a date.

**A single-class or tiny vintage gets NaN in the rank-based columns, not an exception.**
A partition boundary lands mid-year often enough that refusing to produce the table would mean never producing it.
`rows` and `defaults` are always populated, so a NaN is explicable rather than mysterious.
`brier_score` is still computed on one class, because "you said 8% and nobody defaulted" is a real, usable finding.

## What must NOT live here

- **Thresholds, alerting, or anything that fires.** This module returns numbers and bands. Whether a band is an alarm is a policy question, and a monitoring module that decides it is one that has to be edited to change a policy.
- **Reading or writing artifacts.** `pipeline.py` names the CSVs; a drift function that wrote `psi_features.csv` could not be called on two arbitrary populations, which is the only reason it is useful.
- **Any fit or transform.** Callers pass frames that are already engineered. Doing the transform here would mean this module knew the pipeline's shape, and there would then be two places that did.
- **A second definition of any metric.** `auc_roc`, `ks_statistic`, and `brier_score` per vintage come from `evaluation.py`, so a per-vintage AUC and the headline AUC cannot be computed two different ways.
- **Distribution tests as an alternative to PSI.** A KS or chi-square p-value over 400k rows is significant for shifts nobody would act on. PSI is reported because it measures magnitude, which is what a decision needs.

## Related tests

`tests/test_drift.py`, 29 tests. Every PSI assertion is against a hand-computed constant rather than a range, because there is a whole family of plausible-but-wrong implementations - dropping NaN, closing the outer edges, renormalizing after the floor - and all of them return numbers in the same ballpark as the correct one. `assert psi > 0.25` would pass for most of them.

- `test_psi_matches_the_arithmetic_written_out_by_hand` is the anchor: 50/50 shifting to 25/75 is 0.27465307216702742, written out longhand in the test body.
- `test_a_population_against_itself_scores_exactly_zero` and `test_every_bucket_contributes_a_non_negative_amount` pin the two structural properties.
- `test_a_shift_in_missingness_alone_is_drift`, `test_an_infinity_is_counted_as_missing_rather_than_as_an_extreme`, and `test_a_level_the_reference_never_showed_lands_in_the_unseen_bucket` cover the three bucketing decisions that distinguish this implementation from the naive one.
- `test_a_value_beyond_the_reference_range_is_still_counted` asserts on the totals, which is the only way the open-edge property is visible.
- `test_an_empty_bin_is_floored_at_half_an_observation` pins the correction at its exact value.
- `test_the_bands_are_the_conventional_ones_at_their_exact_edges` states that 0.10 is already `moderate`, because a report flipping at 0.1000001 is indistinguishable by eye and differs on real data.
- `test_each_rows_psi_is_the_psi_of_that_column` is what stops `feature_drift` becoming a second implementation of the scalar.
- `test_the_run_publishes_both_psi_tables_and_the_headline_numbers` asserts the `metrics.json` scalar is the sum of the CSV beside it, so the dashboard number and its drilldown cannot disagree.
- `test_the_run_publishes_a_vintage_table_covering_all_three_partitions` reconciles each partition's vintage rows against its published row count, so no partition can silently lose a year.

## Known limits

- **Quantile bins are cut from the reference, so PSI is not symmetric.** `psi(a, b)` and `psi(b, a)` differ. This is the standard definition and the asymmetry is meaningful - the reference is the population the model was fitted on - but it means the number cannot be read as a distance.
- **Two empty bins on populations of different sizes contribute a little more than nothing.** The floor is `0.5 / n` per side, so an empty bin's two shares differ when the sample sizes do. On the synthetic run above that shows up as 0.00064 out of a total of 0.0295. It is well below the resolution of any band, and removing it would mean a shared floor that no longer means "half an observation of *this* sample".
- **Ten bins is a convention, not a derivation.** More bins detect a narrower shift but make each share noisier, and PSI sums over bins so that noise accumulates rather than averaging out. `buckets` is a keyword argument for callers who have a reason.
- **PSI says a feature moved, not that the model got worse.** A feature the model barely uses can drift hard and change nothing. Reading `psi_features.csv` beside `shap_summary.csv` is the actual diagnosis; nothing here combines them into a single importance-weighted score, because that number would hide which of its two inputs moved.
- **`feature_drift` scores the whole frames it is given.** On the real extract's train partition that is a few hundred thousand rows of quantile computation. Fine once per run, and the caller should sample before calling it in a loop.
- **The vintage table groups by calendar year.** A quarter-level breakdown would be more useful on a 2013-2015 window and is a one-line change; year is what the embargo's before/after report already uses, and two different vintage granularities in one project would be worse than a coarse one.
