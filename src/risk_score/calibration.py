"""Measuring calibration, correcting it, and plotting it without leaking.

A credit model's ranking and its *level* are two different claims. AUC and KS
describe the ranking and are invariant to any monotone rescaling of the score.
The Brier score and the calibration curve describe the level: whether a stated
14% chance of default actually defaults 14% of the time. Only the second kind
supports a decision, because the cost matrix multiplies probabilities.

This module was three-quarters non-functional.

**The calibrator was dead code, and would have raised if called.**
``calibrate_model`` constructed ``CalibratedClassifierCV(model, cv="prefit")``.
Nothing in the project ever called it - and ``cv="prefit"`` was *removed* in
scikit-learn 1.9, so the first caller would have got an
``InvalidParameterError``. The supported spelling is
``CalibratedClassifierCV(FrozenEstimator(model))``, which is what
:func:`fit_calibrator` uses, and it is now called on every run.

**So the reported calibration described a knowingly miscalibrated model.**
The logistic baseline trained with ``class_weight="balanced"``, which multiplies
the minority-class weight by roughly 1/base-rate and inflates every predicted
probability. The old calibration plot was not a diagnosis of a miscalibrated
model; it was a picture of a deliberate reweighting, presented as a finding
(audit B05). The weighting is gone from ``modeling.py``, and what remains of the
gap is now corrected rather than only drawn.

**The curve had no sample counts and no summary number.**
``sklearn.calibration.calibration_curve`` silently drops empty bins and returns
no counts, so a plot point built from four loans looked exactly like one built
from forty thousand. The curve here carries ``rows`` per bin, and
:func:`compute_expected_calibration_error` reduces it to one row-weighted
number, so calibration can be compared across runs without reading a chart.

**And the plot mutated global state twice per call.**
It set ``MPLCONFIGDIR`` in ``os.environ`` - a process-wide side effect from a
plotting helper, invisible to any caller - and it never closed the figure, so
every call leaked one for the life of the process. Under the retrain endpoint
that is a slow memory leak triggered by an HTTP request (audit B26).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from risk_score.evaluation import as_metric_arrays
from risk_score.modeling import ValidationPartition

#: Bins in the calibration curve. Ten quantile bins over a validation partition
#: of a few thousand rows keeps a few hundred loans per bin, which is enough for
#: the observed default rate in a bin to mean something.
DEFAULT_CALIBRATION_BINS = 10

#: Below this many defaults in the calibration partition, isotonic regression is
#: replaced by Platt scaling.
#:
#: Isotonic fitting is non-parametric: it fits a step function with as many steps
#: as the data supports, so with few positives it produces a handful of wide
#: plateaus that reproduce the calibration sample rather than the population - and
#: those plateaus are permanent, since every score inside one collapses to a
#: single calibrated value. Platt scaling fits two parameters and cannot overfit
#: in that way. 250 is the conventional order of magnitude at which isotonic
#: starts to be preferred, and it is a threshold worth being explicit about
#: because the failure it prevents looks like *better* calibration on the
#: partition it was fitted on.
MIN_POSITIVES_FOR_ISOTONIC = 250

#: The two methods scikit-learn offers, named here so an unknown one fails with
#: the list rather than deep inside a fit.
CALIBRATION_METHODS: tuple[str, ...] = ("isotonic", "sigmoid")


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """A calibration curve, its summary error, and how it was produced."""

    curve: pd.DataFrame
    expected_calibration_error: float
    #: Bins actually produced. Fewer than requested when the score distribution
    #: has ties spanning a quantile boundary, which a tree model routinely does.
    bins: int

    def summary(self) -> str:
        """One line for a log or a manifest."""
        return (
            f"calibration bins={self.bins} ece={self.expected_calibration_error:.4f} "
            f"rows={int(self.curve['rows'].sum())}"
        )


def compute_calibration_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> pd.DataFrame:
    """Observed default rate against predicted probability, by quantile bin.

    Columns: ``bin``, ``rows``, ``mean_predicted_probability``,
    ``observed_default_rate``, ``lower_score``, ``upper_score``.

    ``rows`` is the column ``sklearn.calibration.calibration_curve`` does not
    return, and its absence is why the committed artifact's plot was readable at
    all: a point built from four loans is drawn the same size as one built from
    forty thousand, and the eye weights them equally.

    Quantile bins rather than equal-width ones, because predicted default
    probabilities pile up in the bottom decile - equal-width binning puts almost
    every row in the first two bins and leaves the rest empty.

    Binning is done with :func:`pandas.qcut` and ``duplicates="drop"``, so a score
    with heavy ties yields *fewer* bins rather than raising. The returned frame
    says how many there are; nothing pretends there were ten.
    """
    if n_bins < 2:
        raise ValueError(f"A calibration curve needs at least 2 bins; got {n_bins}.")
    labels, scores = as_metric_arrays(y_true, y_score)

    frame = pd.DataFrame({"label": labels, "score": scores})
    frame["bin"] = pd.qcut(frame["score"], q=n_bins, labels=False, duplicates="drop")

    grouped = frame.groupby("bin", observed=True)
    curve = pd.DataFrame(
        {
            "bin": grouped.size().index.astype(int),
            "rows": grouped.size().to_numpy(),
            "mean_predicted_probability": grouped["score"].mean().to_numpy(),
            "observed_default_rate": grouped["label"].mean().to_numpy(),
            "lower_score": grouped["score"].min().to_numpy(),
            "upper_score": grouped["score"].max().to_numpy(),
        }
    )
    return curve.reset_index(drop=True)


def compute_expected_calibration_error(curve: pd.DataFrame) -> float:
    """Row-weighted mean absolute gap between predicted and observed rates.

    The number the curve is for. 0.02 means that, averaged over applicants, the
    model's stated default probability is two percentage points away from the
    rate actually observed at that score.

    Weighted by ``rows``, not by bin, because an unweighted mean lets a sparse
    tail bin - twelve loans, all of which happened to default - dominate a summary
    of forty thousand.
    """
    required = {"rows", "mean_predicted_probability", "observed_default_rate"}
    missing = sorted(required - set(curve.columns))
    if missing:
        raise ValueError(f"Calibration curve is missing column(s): {missing}.")

    rows = curve["rows"].to_numpy(dtype=np.float64)
    if rows.sum() == 0:
        raise ValueError("Calibration curve contains no rows.")
    gap = (curve["observed_default_rate"] - curve["mean_predicted_probability"]).abs().to_numpy()
    return float(np.average(gap, weights=rows))


def build_calibration_report(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> CalibrationReport:
    """The curve and its summary error together, since neither is useful alone."""
    curve = compute_calibration_curve(y_true, y_score, n_bins=n_bins)
    return CalibrationReport(
        curve=curve,
        expected_calibration_error=compute_expected_calibration_error(curve),
        bins=len(curve),
    )


def choose_calibration_method(y_true: pd.Series) -> str:
    """``"isotonic"`` when there are enough defaults to support it, else ``"sigmoid"``.

    Automatic because the wrong choice does not look like an error. Isotonic on a
    partition with 30 defaults produces a beautiful calibration curve *on that
    partition* and a step function that has memorized it.
    """
    positives = int(pd.Series(y_true).astype(int).sum())
    return "isotonic" if positives >= MIN_POSITIVES_FOR_ISOTONIC else "sigmoid"


def fit_calibrator(
    model: Any,
    partition: ValidationPartition,
    *,
    method: str = "auto",
) -> CalibratedClassifierCV:
    """Fit a probability calibrator on the validation partition, model frozen.

    Takes a :class:`~risk_score.modeling.ValidationPartition` rather than an
    ``x``/``y`` pair: this is a fitted object, the partition it is fitted on
    determines whether the reported Brier score means anything, and a function
    taking two frames cannot tell which it was given.

    :class:`~sklearn.frozen.FrozenEstimator` is what makes the base model
    genuinely frozen - its ``fit`` is a no-op, so ``CalibratedClassifierCV`` fits
    only the calibration map and the training partition is never revisited. This
    replaces ``cv="prefit"``, which scikit-learn removed in 1.9; the old code
    would have raised ``InvalidParameterError`` on its first call, and never had
    one.

    The returned object is self-contained: it holds the frozen pipeline inside
    it, takes a *raw* frame, and returns calibrated probabilities. There is
    nothing to keep in sync with the model file, because it contains the model.
    """
    if not isinstance(partition, ValidationPartition):
        raise TypeError(
            "fit_calibrator takes a ValidationPartition, not bare frames. A "
            "calibrator fitted on test data makes the test Brier score a "
            "description of the fit; wrap the partition so that cannot happen "
            "by accident."
        )
    resolved = choose_calibration_method(partition.y) if method == "auto" else method
    if resolved not in CALIBRATION_METHODS:
        raise ValueError(
            f"Calibration method must be one of {list(CALIBRATION_METHODS)} or 'auto'; "
            f"got {method!r}."
        )

    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=resolved)
    calibrator.fit(partition.x, partition.y)
    return calibrator


def plot_calibration_curve(
    calibration_data: pd.DataFrame,
    *,
    output_path: str | Path,
) -> None:
    """Draw the calibration curve to ``output_path`` and close the figure.

    Returns nothing on purpose. The old signature returned the figure and made
    ``output_path`` optional, so the default behaviour was to build a figure,
    hand it to a caller that ignored it, and leave it open - one leaked figure
    per call, for the life of the process, which under a retrain endpoint is a
    memory leak an HTTP request can trigger (audit B26).

    ``MPLCONFIGDIR`` is not set here. The old version wrote it into
    ``os.environ`` - a plotting helper reaching into process-wide state that
    every later import then observes. Where matplotlib needs a writable config
    directory, the environment provides one; the Dockerfile sets it.
    """
    # Imported inside the function: matplotlib is a heavy import and the API's
    # scoring path has no use for it. `Agg` before pyplot, because the backend
    # cannot be switched after pyplot has chosen one.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    predicted = calibration_data["mean_predicted_probability"].to_numpy(dtype=np.float64)
    observed = calibration_data["observed_default_rate"].to_numpy(dtype=np.float64)
    rows = (
        calibration_data["rows"].to_numpy(dtype=np.float64)
        if "rows" in calibration_data.columns
        else np.full(predicted.size, np.nan)
    )

    figure, axes = plt.subplots(figsize=(6, 6))
    try:
        axes.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1,
            color="#999999",
            label="Perfect calibration",
        )
        axes.plot(predicted, observed, color="#1f77b4", linewidth=1.5, zorder=2, label="Model")
        if np.isfinite(rows).all():
            # Bin size is encoded as marker area rather than as a text label per
            # point. The labels were unreadable in practice: quantile bins are
            # equal-sized, so all ten read "n=35" and the ones in the crowded
            # bottom-left corner - where a credit model puts most of its mass -
            # overlapped each other. Area still distinguishes the collapsed bins
            # that a tie-heavy score produces, which is the case the count is for.
            low, high = int(rows.min()), int(rows.max())
            if low == high:
                # Equal-sized bins are the normal case, and scaling identical
                # counts would only make every marker large for no information.
                areas = np.full(predicted.size, 45.0)
                size_note = f"{low} loans per bin"
            else:
                areas = 30.0 + 170.0 * rows / rows.max()
                size_note = f"{low}-{high} loans per bin, marker area proportional to count"
            axes.scatter(predicted, observed, s=areas, color="#1f77b4", zorder=3)
            axes.set_title(f"Calibration\n{len(predicted)} quantile bins, {size_note}", fontsize=10)
        else:
            axes.scatter(predicted, observed, s=40, color="#1f77b4", zorder=3)
            axes.set_title("Calibration", fontsize=10)

        axes.set_xlabel("Mean predicted default probability")
        axes.set_ylabel("Observed default rate")
        # Fixed limits and an equal aspect, so the reference line is drawn at a
        # true 45 degrees. On autoscaled axes it is not, and the eye reads
        # distance from that line as the size of the miscalibration.
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.0)
        axes.set_aspect("equal")
        axes.grid(True, linewidth=0.4, color="#dddddd")
        axes.set_axisbelow(True)
        axes.legend(loc="upper left", frameon=False, fontsize=9)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
    finally:
        # `finally`, so a failed savefig - a read-only directory, a full disk -
        # does not leak the figure on its way out.
        plt.close(figure)
