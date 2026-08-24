# 0004 - Require outcome maturity, not just a closed status

Status: accepted.
Supersedes nothing.
Affects `src/risk_score/data_loading.py`, `src/risk_score/config.py`, `src/risk_score/pipeline.py`.

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

There is a second half to the context, and it is the more embarrassing one.
The first version of this ADR was written alongside a working, tested `apply_outcome_maturity_embargo`, and **nothing called it**.
`pipeline.py` went from `load_lending_club_data` straight to the label, and `RunConfig` had no field a snapshot could have been passed in.
So the correction existed as a documented capability and every artifact this project has ever produced still contained the bias.
That is a worse failure than not having written the function, for the same reason the un-applied calibrator was: a reader of the docs and the tests would have concluded the problem was solved.

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

And the call is in `pipeline.py`, in a numbered section with every other row filter, before the split:

```python
# --- 1. which rows are admissible at all ---
loans = load_lending_club_data(raw_data_path, ...)
embargo = apply_outcome_maturity_embargo(loans, snapshot=config.data.snapshot, ...)
loans = filter_to_terms(embargo.loans, terms=config.data.term_months_in)
```

The snapshot arrives from a new `data` section of the run config, alongside `term_months_in`.
That section holds exactly the values that describe the *extract* rather than the experiment, which is why it is separate from `split`.

## Consequences

**The usable data shrinks, and the recent vintages go first.**
With a 2018-12 snapshot and 36-month terms, nothing issued after 2015-12 survives.
That is the correct amount of data, not a limitation to work around: the discarded rows have no known outcome.

**The headline metrics get worse and become honest.**
The inflated late-vintage base rate was making the model look more discriminating than it is.

**The split windows are downstream of the snapshot, so they are not independently configurable in practice.**
Under a 2018-12 snapshot nothing issued after 2015-12 survives, so the previously shipped test window of `2016-01`..`2016-12` would be empty and `split_by_time` would raise.
The defaults moved together: train `2013-01`..`2014-09`, validation `2014-10`..`2015-03`, test `2015-04`..`2015-12`.
`DEFAULT_SNAPSHOT` and `DEFAULT_SPLIT_WINDOWS` sit next to each other in `config.py` with a comment saying so, because the failure mode is a config file that changes one of them.

**The 36-month restriction is a consequence of this decision, not an independent one.**
With the embargo applied, 60-month loans only exist in the earliest vintages - 28% of the 2013 rows in the synthetic extract and 0% of the 2015 rows.
An unrestricted run would train on a term mix the validation and test partitions do not contain, which is a train/serve mismatch dressed up as more data.
`filter_to_terms` implements it, `data.term_months_in` configures it, and `[]` turns it off.
The cliff itself is reported in the drift section rather than silently absorbed.

**The snapshot is a required input with a shipped default, which is a compromise.**
It cannot be inferred - the latest `issue_d` is a lower bound on the snapshot, not the snapshot - and getting it wrong in the optimistic direction (too late) reintroduces exactly the bias this exists to remove.
The first version of this ADR concluded from that there should be no default at all.
That was reversed for one reason: `RunConfig()` with no arguments has to be a complete, correct configuration, or the demo path and every test grows a mandatory field whose value they all copy.
`2018-12-01` is correct for `data/raw/1/loan.csv` and conservative for the synthetic extract, whose observer runs to 2019-06.

The mitigation for a wrong snapshot is that it is *visible* rather than prevented.
`default_rate_by_vintage_before_embargo` and `..._after_embargo` both go into the metrics payload, so a snapshot set too late shows up as a residual climb in the "after" column - the same signature as the uncorrected data, in the same table, one row apart.

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

That test covers the function. Four more cover the *run*, because a working function nothing calls was the original failure:

- `tests/test_pipeline.py::test_the_maturity_embargo_runs_and_flattens_the_vintage_default_rate` reads the metrics payload a run wrote and asserts the before column climbs, the after column does not, and the censored vintage is gone.
- `..::test_immature_loans_never_reach_any_partition` checks the calendar in all three partitions, which is how a filter accidentally applied after the split would be caught.
- `..::test_the_term_filter_removes_the_sixty_month_cliff` and `..::test_admitting_every_term_is_a_config_change_not_a_code_change`.

End to end on 12,000 synthetic rows through the shipped config, the embargo removes 1,997 immature loans of 9,255 closed ones and moves the vintage rates from 0.146 / 0.158 / 0.202 / 0.311 to 0.146 / 0.131 / 0.143.
Test AUC falls from 0.751 to 0.638 and test Brier from 0.177 to 0.118.
The AUC was partly the model learning which vintages were censored, so that drop is the deliverable rather than a regression.

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
