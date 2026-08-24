# 0006 - The calibrator is fitted on validation, and applied

Status: accepted.
Affects `src/risk_score/calibration.py`, `src/risk_score/modeling.py`, `src/risk_score/pipeline.py`.

## Context

The project measured calibration and then discarded the correction.

`calibrate_model` existed, was never called from anywhere, and could not have worked if it had been.
It passed `cv="prefit"`, which scikit-learn removed in 1.9, so the first caller would have got an `InvalidParameterError` rather than a calibrated model.
Meanwhile `compute_calibration_curve` and `plot_calibration_curve` ran on every run and wrote a chart.

That combination is worse than having no calibration story at all, because the chart looked like a diagnosis.
The logistic baseline trained with `class_weight="balanced"`, which multiplies the minority-class weight by roughly one over the base rate and inflates every predicted probability by construction.
So the committed calibration figure was a picture of a deliberate reweighting, presented as a finding about the model (audit B05).
The Brier score in the metrics file described the same inflated probabilities.

This matters more than a cosmetic reporting issue, because the decision rule multiplies probabilities.
`select_threshold_by_cost` weighs false negatives against false positives at every candidate cut-off, and the counts it uses come from comparing scores to a threshold.
If the scores are systematically inflated, the cost table is a table about a different model than the one that would be served, and the threshold it selects is the right answer to the wrong question.

Two further questions had to be settled before the correction could be applied at all.

**Which partition fits it.**
A calibrator has parameters, so fitting it on test makes the test Brier score a description of the fit rather than a measurement.
ADR 0003 already established that every fitted decision belongs on validation; this is the second such decision, and the first one with real parameters.

**How the correction is persisted and applied.**
A calibration map is only meaningful next to the exact model whose scores it corrects.
Two files that can be loaded independently can be loaded from different runs, and a mismatched pair is indistinguishable at serving time from a correct one.

## Decision

**Fit the calibrator on the validation partition, wrap it around a frozen model, and score everything through it.**

```python
calibrator = fit_calibrator(model, split.validation)  # validation only
scores_validation = _predict_default_probability(calibrator, split.x_validation)
threshold = select_threshold_by_cost(ValidationScores(...), cost_matrix=cost_matrix)
scores_test = _predict_default_probability(calibrator, split.x_test)  # report only
```

Five parts to it.

**1. `CalibratedClassifierCV(FrozenEstimator(model), method=...)`.**
`FrozenEstimator.fit` is a no-op, so only the calibration map is fitted and the training partition is never revisited.
This is the supported replacement for the removed `cv="prefit"`, and it was verified empirically to produce exactly one entry in `calibrated_classifiers_` rather than a cross-validated ensemble.

**2. A `ValidationPartition` type, and `fit_calibrator` accepts nothing else.**
`ValidationPartition` is a frozen dataclass holding the validation `x` and `y`, produced by the new `TimeSplit.validation` property.
A function taking two frames cannot tell which partition it was handed; a function taking this can only be given the wrong one on purpose.
It is a second type rather than a reuse of `ValidationScores` because the calibrator needs the *features* - it re-scores them through the frozen model - while the threshold search needs only the scores.
The check is at runtime as well as in the annotation, because pandas ships no type stubs in this project, so `pd.DataFrame` is `Any` to mypy and static checking alone would let a test frame through.

**3. Isotonic when the partition carries at least 250 defaults, Platt scaling below that, decided automatically.**
Isotonic regression is non-parametric: it fits a step function with as many steps as the data supports.
With few positives it produces a handful of wide plateaus that reproduce the calibration sample rather than the population, and those plateaus are permanent, because every score inside one collapses to a single calibrated value.
The choice is automatic, not a caller's parameter, because the wrong choice does not look like an error.
Isotonic on 30 defaults yields a *better* calibration curve on the partition it memorized, so a reviewer looking at the output sees an improvement and concludes, wrongly, that nothing needs attention.

**4. The threshold is selected on calibrated validation scores.**
The served decision applies the calibrator and then compares to the threshold, so that is the score distribution the cost table has to describe.
Selecting on uncalibrated scores and applying the result to calibrated ones costs one policy and publishes another.

**5. Two artifacts that cannot drift apart.**
`models/<type>.joblib` stays a bare `sklearn.pipeline.Pipeline`, and `models/<type>_calibrator.joblib` holds the calibrator.
They cannot disagree, because the calibrator holds that same pipeline object by reference inside its `FrozenEstimator`, so the calibrator pickle contains the model.
The bare pipeline is still written because it is the only way to inspect the fitted preprocessing without unwrapping a calibrator, and the tests use it for exactly that.
Phase 4's `ScoringBundle` supersedes the two-file arrangement by putting the pipeline, the calibrator, and the threshold in one object.

**And report the pair of numbers that makes the step falsifiable.**
The metrics payload carries `brier_score` (calibrated), `brier_score_uncalibrated`, `expected_calibration_error` on test, `expected_calibration_error_validation_in_sample`, `calibration_method`, and `calibration_fitted_on`.
If the calibrated Brier score is not lower than the uncalibrated one, the correction is not earning its place in the artifact, and a reader can see that without rerunning anything.

On the synthetic extract the step moves test Brier from 0.1850 to 0.1768, reports a test ECE of 0.0946 against an in-sample validation ECE of 0.0360, and moves the selected threshold from 0.14 to 0.17.

## Consequences

**The reported Brier score now describes the model that would be served.**
It is also the metric that will get worse if the preprocessing or the estimator regresses, which the rank-based metrics will not notice.

**The calibrator is fitted on the same rows the threshold is selected on, so the validation cost table is mildly optimistic.**
This is a real cost and it is stated at the call site in `pipeline.py` rather than buried.
The mitigation is that the reported headline numbers come from test, which the calibrator never saw, and that the in-sample validation ECE is published beside the test ECE rather than in place of it.

**AUC, average precision, and KS are unchanged in the sigmoid case and can move slightly under isotonic.**
Platt scaling is strictly monotone, so it cannot reorder anything.
Isotonic is monotone but not strictly, so it collapses score bands into plateaus and creates ties, which the tie-collapsed KS then treats as the single achievable operating point they are.
That is the correct behaviour, and it means a KS drop after switching to isotonic is information rather than a bug.

**Serving must go through the calibrator, and there is now a wrong way to do it.**
Loading `models/<type>.joblib` and calling `predict_proba` produces uncalibrated scores that do not match the published threshold.
The test suite loads the calibrator for every score-based assertion for this reason, and the Phase 4 bundle removes the choice entirely by shipping one object.

**Calibration is fitted once, on one vintage window.**
A map fitted on 2015 loans is applied to 2016 applicants, and the level is the first thing to drift when underwriting standards move.
Nothing recalibrates automatically; the PSI work is what surfaces the need.

## Alternatives considered

**Keep measuring and not applying, and just remove `class_weight="balanced"`.**
Removing the weighting was necessary and is done, but it is not sufficient.
An unweighted logistic model on a 15% base rate is still miscalibrated, just less obviously, and a gradient-boosted model with a log-loss objective is miscalibrated in its own way.
Measuring a gap and declining to close it is the state this ADR exists to leave.

**Carve out a fourth partition to fit the calibrator on, so the threshold search sees out-of-sample calibrated scores.**
This is the statistically clean answer and the one to revisit if the dataset grows.
Rejected for now because it cuts validation roughly in half, and the validation partition is already the smallest of the three and the one that both early stopping and the threshold search depend on.
Trading a known, bounded, documented optimism in the validation cost table for a materially noisier threshold is the wrong trade at this size.

**Cross-validated calibration on train, using `CalibratedClassifierCV` with `cv=5` as designed.**
Uses the data efficiently and refits the base estimator five times, producing an ensemble of five models rather than the one this project has to serve.
It also puts calibration back on the training distribution, which is precisely the distribution whose optimism the tri-split exists to avoid.

**Isotonic always, since it is more flexible.**
Rejected on the failure mode: with few positives it memorizes the calibration partition and reports better calibration for having done so.
A default that fails invisibly is worse than a default that is occasionally too rigid, and Platt scaling being too rigid is visible in the curve.

**A fixed choice of method in the run config.**
Rejected because it is a decision about the *data*, not about the run, and a config value gets copied between projects while the number of defaults does not.
`method=` is still available for a caller who knows better, and an unknown value raises with the two supported ones.

**Keep returning the figure from `plot_calibration_curve` and let the caller close it.**
Rejected: the old signature returned a figure and made the output path optional, so the default behaviour was to build a figure, hand it to a caller that ignored it, and leave it open.
Under the retrain endpoint that is a memory leak an HTTP request can trigger (audit B26).
The function now writes and returns nothing, and closes the figure in a `finally` so a failed `savefig` cannot leak one either.
