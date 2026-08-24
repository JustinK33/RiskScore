# `src/risk_score/calibration.py`

## Purpose

Measure whether the model's stated probabilities are true, correct them when they are not, and draw the result.

A credit model makes two separate claims, and this file owns the second one.
AUC and KS describe the *ranking*, and both are invariant to any monotone rescaling of the score.
The Brier score and the calibration curve describe the *level*: whether a stated 14% chance of default actually defaults 14% of the time.
Only the second kind supports a decision, because the cost matrix multiplies probabilities.

Three quarters of this module did not work.

**The calibrator was dead code, and would have raised if called.**
`calibrate_model` constructed `CalibratedClassifierCV(model, method=method, cv="prefit")`.
Nothing in the project ever called it, and `cv="prefit"` was *removed* in scikit-learn 1.9, so the first caller would have got an `InvalidParameterError` rather than a calibrated model.
The supported spelling is `CalibratedClassifierCV(FrozenEstimator(model))`, and it is now called on every run.

**So the reported calibration described a model nobody would have shipped.**
The logistic baseline trained with `class_weight="balanced"`, which multiplies the minority-class weight by roughly one over the base rate and inflates every predicted probability.
The old calibration plot was not a diagnosis of a miscalibrated model.
It was a picture of a deliberate reweighting, presented as a finding (audit B05).
The weighting is gone from `modeling.py`, and whatever gap remains is now corrected rather than only drawn.

**The curve had no sample counts and no summary number.**
`sklearn.calibration.calibration_curve` silently drops empty bins and returns no counts, so a plot point built from four loans looked exactly like one built from forty thousand and the eye weighted them equally.

**And the plot mutated global state twice per call.**
It wrote `MPLCONFIGDIR` into `os.environ`, a process-wide side effect from a plotting helper that every later import then observes, and it never closed its figure, so each call leaked one for the life of the process.
Under the retrain endpoint that is a slow memory leak an HTTP request can trigger (audit B26).

## Public API

| Name | What it is |
| --- | --- |
| `CalibrationReport` | Frozen: the curve, its `expected_calibration_error`, the number of `bins` actually produced, and a one-line `summary()`. |
| `compute_calibration_curve` | Observed default rate against mean predicted probability, by quantile bin, with per-bin counts and score ranges. |
| `compute_expected_calibration_error` | The curve reduced to one row-weighted number. |
| `build_calibration_report` | Both of the above together, which is how they are always used. |
| `choose_calibration_method` | `"isotonic"` or `"sigmoid"`, decided from the number of defaults available. |
| `fit_calibrator` | Fit a `CalibratedClassifierCV` around a frozen model, on a `ValidationPartition`. |
| `plot_calibration_curve` | Write the reliability diagram to a path. Returns nothing. |
| `DEFAULT_CALIBRATION_BINS` | `10`. |
| `MIN_POSITIVES_FOR_ISOTONIC` | `250`. |
| `CALIBRATION_METHODS` | `("isotonic", "sigmoid")`. |

## Inputs and outputs

`compute_calibration_curve` takes labels and scores and returns a DataFrame with `bin`, `rows`, `mean_predicted_probability`, `observed_default_rate`, `lower_score`, `upper_score`.
The middle two column names are load-bearing: `dashboard/js/panels.js` reads `mean_predicted_probability` and `observed_default_rate` by name, so they are preserved and new columns are additive.

`fit_calibrator` takes a fitted estimator and a `risk_score.modeling.ValidationPartition`, and returns a fitted `CalibratedClassifierCV`.
That returned object is self-contained.
It holds the frozen pipeline inside it, accepts a *raw* frame because canonicalization is the pipeline's first step, and returns calibrated probabilities.
There is nothing to keep in sync with the model file, because it contains the model.

`plot_calibration_curve` writes one PNG and returns `None`.
It imports matplotlib inside the function body, because matplotlib is a heavy import and the serving path has no use for it, and it selects the `Agg` backend before importing `pyplot`, since the backend cannot be switched afterwards.

Nothing in this file reads a file, and only the plot writes one.

## Invariants and failure modes

**The calibrator is fitted on validation, and only validation.**
`fit_calibrator` accepts a `ValidationPartition` and raises `TypeError` on bare frames, which mirrors `select_threshold_by_cost`'s `ValidationScores` guard for the same reason.
A calibrator fitted on test turns the test Brier score into a description of the fit.
The runtime check is load-bearing rather than belt-and-braces: pandas ships no type stubs in this project, so `pd.DataFrame` is `Any` to mypy and the annotation alone would let the call through.
See [../decisions/0006-calibration-on-validation.md](../decisions/0006-calibration-on-validation.md).

**The base model is genuinely frozen.**
`FrozenEstimator.fit` is a no-op, so `CalibratedClassifierCV` fits only the calibration map and the training partition is never revisited.
`calibrated_classifiers_` has exactly one entry, which is how a test asserts no cross-validation ensemble was built behind the scenes.

**The method is chosen from the data, not by the caller.**
Isotonic regression is non-parametric: it fits a step function with as many steps as the data supports.
With few positives that means a handful of wide plateaus that reproduce the calibration sample rather than the population, and those plateaus are permanent, because every score inside one collapses to a single calibrated value.
Platt scaling fits two parameters and cannot overfit in that way.
The choice is automatic because the wrong one does not look like an error: isotonic on 30 defaults produces a *better* looking calibration curve on the partition it memorized.
250 is the conventional order of magnitude at which isotonic starts to be preferred, and `method=` remains available to override it.
An unknown method raises with the list of the two supported ones rather than failing deep inside a fit.

**Bins are quantile bins, and a tie-heavy score yields fewer of them rather than an exception.**
Predicted default probabilities pile up in the bottom decile, so equal-width binning puts almost every row in the first bin or two and leaves the rest empty.
`pd.qcut(..., duplicates="drop")` is what lets a boosted model with a few hundred distinct probabilities be binned at all, and `CalibrationReport.bins` reports how many bins actually resulted.
Nothing pretends there were ten.

**The ECE is weighted by rows and taken over absolute gaps.**
Weighted, because an unweighted mean over bins lets a sparse tail bin of twelve loans that all happened to default dominate a summary of forty thousand.
Absolute, because one bin 0.1 too high and one 0.1 too low average to zero error on a model that is not calibrated.
A curve missing a required column raises and names the column.

**Every reported number goes through the same input gate.**
The curve calls `risk_score.evaluation.as_metric_arrays`, so an empty partition, a NaN score, mismatched lengths, or a label outside `{0, 1}` fails here exactly as it does for AUC.

**The figure is always closed, including on failure.**
`plt.close(figure)` is in a `finally`, so a read-only directory or a full disk cannot leak one on the way out.
`MPLCONFIGDIR` is never set from here; where matplotlib needs a writable config directory, the Dockerfile provides one.

**The plot is legible about sample size.**
Bin size is encoded as marker area rather than a per-point text label.
The labels were tried and were unreadable in practice: quantile bins are equal-sized, so all ten read `n=35`, and the ones in the crowded bottom-left corner where a credit model puts most of its mass overlapped each other.
`_marker_sizes` therefore scales area *only* when the largest bin is at least twice the smallest, which is the tie-heavy tree-score case a count is actually for.
Otherwise the counts differ by a single row, every marker would be drawn near-maximal, and the subtitle would promise "area proportional to count" about a difference of one loan.
The range is stated outright either way.

**The axis window is square and sized to the book, not to `(0, 1)`.**
`_axis_extent` returns one number used for both limits.
One number, because the window has to stay square or the perfect-calibration reference line is no longer at 45 degrees, and the whole chart is read as distance from that line.
Sized to the data, because a portfolio at a 15% base rate has its entire curve below 0.2, and a `(0, 1)` window spends most of the figure on probabilities no loan has while compressing the informative part into a corner - measured at 27% of the figure's width on the shipped run before this changed.
The extent is the largest predicted or observed rate plus 15% headroom, rounded up to a tenth so the tick labels stay round, floored at 0.2 so a well-calibrated low-risk book does not read as a zoom artifact, and capped at 1.0.

## What must NOT live here

- **Choosing the threshold.** That is `evaluation.py`. This module produces the corrected probabilities the threshold is then chosen against.
- **Deciding which partition to fit on.** `pipeline.py` owns the order of operations; this file only refuses to be handed the wrong one.
- **Any other chart.** The reliability diagram lives here because it is a view of this module's own output. ROC curves, PSI tables, and SHAP plots belong with the code that computes them.
- **Persisting the calibrator.** `joblib.dump` is the pipeline's business, and from Phase 4 the bundle's.
- **A second calibration split.** If one is ever carved out, `modeling.py` cuts it, because that file owns who sees which rows.

## Related tests

`tests/test_calibration.py`, 24 tests.
The old module had none, which is how a function with a removed keyword argument sat in the repository looking fine.

- `test_the_curve_is_a_hand_computable_table_on_a_small_frame` pins the whole output against four rows worked out by hand.
- `test_the_curve_reports_the_row_count_in_every_bin` and `test_the_calibration_curve_carries_the_bin_sizes` (in `test_pipeline.py`) cover the column sklearn does not return.
- `test_a_heavily_tied_score_yields_fewer_bins_rather_than_raising` covers the tree-model case.
- `test_the_curve_uses_quantile_bins_not_equal_width_ones` contrasts the two binnings on a skewed score, where equal-width puts 90 of 100 rows in one bucket.
- `test_the_expected_error_is_weighted_by_rows_not_by_bin` and `test_the_expected_error_is_signed_absolutely_so_bins_cannot_cancel` pin both halves of the ECE definition against hand-computed constants.
- `test_the_calibrator_wraps_a_frozen_model_and_refits_nothing` asserts the base coefficients are untouched and that there is exactly one calibrator.
- `test_calibrating_a_deliberately_inflated_model_lowers_its_brier_score` builds a `class_weight="balanced"` model, the exact failure of audit B05, and asserts the correction improves the held-out Brier score. A calibration step that does not is not doing its job.
- `test_b05_the_calibration_correction_is_applied_and_not_merely_drawn` (in `test_pipeline.py`) asserts the reported Brier score is the calibrated one, by recomputing both from the two persisted artifacts.
- `test_bare_frames_are_refused_by_the_calibrator` is the partition guard.
- `test_the_axes_are_scaled_to_the_book_and_stay_square` and `test_marker_area_encodes_the_count_only_when_the_bins_are_uneven` test `_axis_extent` and `_marker_sizes` directly. They are private, and testing them through the PNG is the reason the first version of the marker test asserted nothing at all: a written file proves a figure was produced, not that anything on it is readable. Pulling both rules out as pure functions is what made them assertable.
- `test_a_curve_with_no_rows_does_not_crash_the_axis_scaling` covers the reduction over an empty array, which would otherwise raise before any message about the empty curve.
- `test_the_plot_writes_a_file_and_leaves_no_open_figure` and `test_the_plot_closes_the_figure_even_when_saving_fails` cover audit B26, counting `plt.get_fignums()` before and after.
- `test_the_plot_does_not_reach_into_the_environment` asserts `MPLCONFIGDIR` is absent afterwards.

## Known limits

- **The calibrator is fitted on the same rows the threshold is selected on,** so the validation cost table is mildly optimistic. The pipeline reports the in-sample validation ECE beside the test ECE rather than passing one off as the other, and the honest number is the test one. Carving out a fourth partition is the clean fix and is not worth it at this dataset size; the reasoning is in [ADR 0006](../decisions/0006-calibration-on-validation.md).
- **Calibration is fitted once, on one vintage window.** A calibration map fitted on 2015 loans is applied to 2016 applicants, and the level is the part of a credit model that drifts first. The PSI work in Phase 5 is what detects it; nothing here recalibrates automatically.
- **The ECE depends on the bin count.** Ten quantile bins is a convention, not a property of the data, and the same model reports a different ECE at 20 bins. It is comparable across runs of this project and not across papers.
- **The figure's legibility is only tested one rule at a time.** Nothing asserts anything about the rendered pixels; `_axis_extent` and `_marker_sizes` are checked as functions and the PNG is checked for existence. Both defects they fix were found by a human looking at the image, and the next one will be too.
- **No confidence band on the curve.** Each point is a raw observed rate, so with 35 loans per bin a swing of 0.1 is unremarkable noise and the chart does not say so. A binomial interval per point is the fix.
- **`choose_calibration_method` counts positives, not effective sample size.** A partition with 300 defaults concentrated in two score bands supports isotonic far less well than 300 spread evenly, and the constant cannot see the difference.
