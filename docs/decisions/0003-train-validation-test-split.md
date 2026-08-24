# 0003 - Train / validation / test, with every fitted decision on validation

Status: accepted.
Affects `src/risk_score/modeling.py`, `src/risk_score/pipeline.py`, `src/risk_score/evaluation.py`, `src/risk_score/calibration.py`.

## Context

The pipeline had two partitions, split on a single date.
The estimator was fitted on train, and everything else happened on test:

```python
threshold = select_threshold_by_cost(y_test, scores_test, cost_matrix=cost_matrix)
metrics = ClassificationMetrics(auc_roc=compute_auc_roc(y_test, scores_test), ...)
```

The threshold search evaluates 99 candidate cut-offs and returns the cheapest.
Doing that on the test set and then reporting that set's cost is not a slightly optimistic estimate of anything - it is reporting the minimum of 99 draws and calling it the expected value.
The reported `selected_threshold_total_cost` was, by construction, the best number obtainable on those rows (audit B04).

The same argument applies with more force to calibration, because a calibrator has parameters.
`calibrate_model` was written to fit a sigmoid on held-out scores and was never called, which is the only reason the Brier score in the committed artifact was not also fitted on the rows it described.
[ADR 0006](0006-calibration-on-validation.md) settles where that calibrator is fitted now, and what it costs.

Two further problems compounded it.

**The threshold was not stored with the model.**
It was written to a metrics JSON while the pipeline went to a pickle, so the model and the decision rule could be loaded independently and drift apart.
A bundle where `predict_proba` comes from one run and `0.14` comes from another is indistinguishable, at serving time, from a correct one.

**Rows between the two windows vanished without a trace.**
`train_end_date` and `test_start_date` were independent parameters, so a gap between them silently discarded every loan in it (audit B09).
A run that dropped a third of its data produced a metrics file identical in shape to one that dropped none.

Under all of it sits a structural point about this dataset: Lending Club vintages are not exchangeable.
Underwriting standards, the term mix, and the macro environment all move year to year, so a random split measures interpolation and a time split measures what a lender actually faces - a model fitted on the past, scoring the future.
Whatever the partitioning scheme, it has to be chronological.

## Decision

Three chronological, disjoint, inclusive windows, and one rule about who may fit what.

```
raw -> label -> leakage audit -> feature spec -> tri-split
  TRAIN       fit the preprocessor and the estimator. Nothing else.
  VALIDATION  early stopping, calibrator, decision threshold.
  TEST        scored once, reported, never fitted on.
```

Concretely:

- `split_by_time` takes three `(start, end)` windows, rejects overlap, and rejects non-chronological order. Both ends are inclusive, and a partial end date expands to the end of the period it names - `2014-09` means through 30 September, because that is how loan vintages are written.
- Every row that reaches no partition is **counted**: `rows_outside_windows` and `rows_unparseable_date` travel in the `TimeSplit` and into the run manifest. A gap is now a number in the log, not an absence.
- The threshold is selected on validation and *measured* on test. The approval rate reported is test's own, because the approval rate a lender sees is a property of the population being scored, not of the population the rule was tuned on.
- The threshold is persisted **inside the model bundle**, alongside the pipeline and the calibrator, so the three cannot be mismatched by loading them separately.
- XGBoost's early stopping is the one thing besides reporting that touches validation, and it touches it through `transform`, never `fit_transform`.
- Defaults: train `2013-01`..`2014-12`, validation `2015-01`..`2015-12`, test `2016-01`..`2016-12`, all subject to the outcome-maturity embargo applied *before* the split so all three partitions share one outcome definition (see [0004](0004-outcome-maturity-embargo.md)).

## Consequences

**Every headline metric drops, and that is the deliverable.**
The old artifact reported an AUC of 0.070 next to a KS of 0.930 - inverted labels, presented with total confidence.
A tri-split does not fix that particular bug, but it removes the mechanism by which a selection artifact reads as a result.
A number that got smaller for a stated reason is worth more than a large number of unknown provenance.

**Validation costs data twice over.**
A year of vintages is removed from training and is not available for reporting either.
On this dataset that is affordable. On a small one it would not be, and nested cross-validation would be the honest alternative - at the cost of a substantially more complex artifact, since there would be no single fitted model to serve.

**"Which partition is this?" becomes a type-level question.**
`fit_calibrator` and `select_threshold` accept a partition-tagged `ValidationScores` wrapper rather than a bare array, so handing them test scores is a mypy error *and* a runtime error.
A convention that lives only in a docstring is a convention that gets violated during a refactor, at which point nothing raises and the metrics simply get better.

**Three windows mean three chances to misconfigure.**
Mitigated by making the failure modes loud: overlap raises and names both boundaries, an empty partition raises and reports the observed date range, and unknown keys in the `split` config section raise rather than falling back to a default (audit B30).

**Reproducibility now depends on recorded windows.**
The resolved window labels go in the manifest, in `TimeSplit.summary()`, and in the model card, because a metric without its evaluation period is not comparable to anything.

## Alternatives considered

**Keep two partitions, select the threshold by cross-validation on train.**
Cheaper in data and genuinely defensible for the threshold alone.
Rejected because it does not generalize: early stopping and the calibrator also need held-out scores, and doing k-fold for each produces k models and no single artifact to serve.
It also would not have surfaced the gap-row bug, which was the second-largest correctness problem in the split.

**Keep two partitions and pick a fixed threshold, say 0.5.**
Genuinely simple and genuinely wrong for a 15% base rate with asymmetric costs.
0.5 approves nearly everyone, which makes the cost matrix decorative and throws away the most useful thing this project computes.

**Random split, or k-fold across all vintages.**
Higher AUC, and the higher number is the argument against it.
It lets 2016 rows inform a model scoring 2014 rows, which no lender can do, and it hides exactly the vintage drift the drift section is meant to measure.

**Nested time-series cross-validation (rolling origin).**
The most statistically complete option, and the right one if the question were "how much does performance vary by period".
Rejected as the *default* because it produces a distribution rather than a model, and this project has to end at a servable bundle.
The per-vintage metric breakdown recovers most of the diagnostic value at a fraction of the complexity.

**A gap - an embargo period - between the partitions.**
Standard practice where the label depends on a forward window, and unnecessary here: the outcome-maturity embargo already ensures every retained loan's term has fully elapsed, so adjacent windows do not share information.
An additional gap would discard rows for no stated reason, and unexplained discards are what audit B09 was about.
