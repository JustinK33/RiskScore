# 0004 - Require outcome maturity, not just a closed status

Status: accepted.
Supersedes nothing.
Affects `src/risk_score/data_loading.py`.

## Context

Supervised default modeling needs a resolved outcome, so the obvious first filter is "keep loans whose status is terminal".
That is what the pipeline did: keep `Fully Paid`, `Charged Off`, `Default`, and the two "does not meet the credit policy" variants, drop everything else.

The filter is necessary and it is not sufficient, because "terminal" is measured against the extract's snapshot date rather than against the loan's own term.

Consider a 36-month loan issued in June 2016, in an extract snapshotted December 2018.
Thirty months have elapsed.
If the borrower defaulted in month 8, the loan is `Charged Off` and the filter keeps it.
If the borrower is paying on schedule, the loan is `Current` and the filter drops it.
The only 2016 loans that survive are the ones that failed.

This is not a subtle effect.
Measured default rate by issue year in `data/raw/1/loan.csv`, after filtering to closed statuses:

| Vintage | Default rate |
| --- | --- |
| 2013 | 15.6% |
| 2014 | 18.5% |
| 2015 | 20.2% |
| 2016 | 24.3% |
| 2018 | 14.7% |

The shape looks like credit quality deteriorating through 2016 and then sharply recovering.
It is entirely an artifact of the snapshot: 2018 reads low because barely any 2018 loan has closed for *any* reason, so the survivors are the 36-month loans that charged off within months plus a handful of early payoffs.

The consequence for this project specifically is that a time-based split makes the bias worse, not better.
Train on 2013-2014, test on 2015-2016, and the test set has a structurally higher base rate than the training set for reasons no feature explains.
Every calibration metric, every threshold cost, and the headline AUC are then measured against a label distribution the model could not have learned.

## Decision

Apply an explicit outcome-maturity embargo before the split:

```
keep the loan when  issue_d + term_months <= snapshot
```

Rows whose `issue_d` or `term` did not parse are also removed, because they cannot be *shown* to be mature, and counted separately from the ones that are genuinely still running.

The embargo runs before the train/validation/test split, so all three partitions share one outcome definition.
Running it per partition would let the definition drift across the split boundary, which is the same class of mistake one level down.

`EmbargoResult` carries the per-vintage default rate before and after, the rows removed, and the snapshot used.
Those go into the run manifest.
A run that discards forty percent of its input must not look identical to one that discards none.

## Consequences

**The usable data shrinks, and the recent vintages go first.**
With a 2018-12 snapshot and 36-month terms, nothing issued after 2015-12 survives.
That is the correct amount of data, not a limitation to work around: the discarded rows have no known outcome.

**The headline metrics get worse and become honest.**
The inflated late-vintage base rate was making the model look more discriminating than it is.

**The 36-month default in the split configuration is a consequence of this decision, not an independent one.**
With the embargo applied, 60-month loans only exist through 2013Q4, so an unrestricted run would train on 13.9% 60-month loans and test on 0.0%.
That term-mix cliff is reported in the drift section rather than silently absorbed.

**The snapshot is now a required input.**
It cannot be inferred from the data - the latest `issue_d` is a lower bound on the snapshot, not the snapshot - so it is passed explicitly and recorded in the manifest.
Getting it wrong in the pessimistic direction (too early) discards usable loans; getting it wrong in the optimistic direction (too late) reintroduces exactly the bias this exists to remove, which is why there is no default.

## Verification

`tests/test_data_loading.py::test_embargo_removes_the_survivorship_bias_in_a_closed_loan_filter`.

The synthetic generator in `risk_score.sample_data` reproduces the bias for the same causal reason the real extract does: it draws a default month from a Beta distribution over the loan's term and then censors both status and outcome against a snapshot date.
It does not hard-code a biased label distribution, so the test is proving that the embargo corrects a mechanism rather than that it reverses a constant someone typed.

On 12,000 synthetic loans with a true lifetime default rate of 15%, closed-only filtering gives:

| Vintage | Closed-only | After embargo |
| --- | --- | --- |
| 2012 | 14.5% | 14.5% |
| 2013 | 14.3% | 14.3% |
| 2014 | 19.2% | 13.3% |
| 2015 | 19.7% | 14.4% |
| 2016 | 100.0% | (removed) |

The 2016 vintage reading 100% is the mechanism at its clearest: with a 2018-12 snapshot, the only 2016 loans that have closed are the ones that defaulted.

## Alternatives considered

**Treat `Current` as non-default.**
Cheap, keeps every row, and wrong in the other direction: a loan four months from a charge-off is labelled as a success.
It converts a bias that inflates the default rate into one that deflates it, and adds label noise concentrated in exactly the recent vintages a time-based test set is made of.

**Restrict to vintages old enough by inspection.**
This is what the embargo does, but hard-coded, so it silently becomes wrong the next time the extract is refreshed with a later snapshot.

**Survival analysis on the censored rows.**
Correct, and a genuinely better use of the data - a Cox model or a discrete-time hazard model can use a loan that is 18 months into a 36-month term.
Rejected for now because it changes the deliverable from a probability of default to a hazard function, which the threshold logic, the calibration report, and the scoring API would all need to be rebuilt around.
It is the right next step if this project grows, and the embargo is the correct baseline to compare it against.
