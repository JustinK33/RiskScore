"""Tests for the calibration curve, the ECE, and the fitted correction.

The old module's calibrator was never called, so nothing here had a test at all -
and ``cv="prefit"`` had been removed from scikit-learn underneath it, meaning the
first caller would have got an ``InvalidParameterError`` rather than a
calibrated model. Every assertion below is against a hand-computed constant or a
property that a broken implementation could not satisfy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from risk_score.calibration import (
    MIN_POSITIVES_FOR_ISOTONIC,
    _axis_extent,
    _marker_sizes,
    build_calibration_report,
    choose_calibration_method,
    compute_calibration_curve,
    compute_expected_calibration_error,
    fit_calibrator,
    plot_calibration_curve,
)
from risk_score.modeling import ValidationPartition


def fitted_model(rng: np.random.Generator, n: int = 600) -> tuple[LogisticRegression, pd.DataFrame]:
    """A trained logistic model and a frame of held-out rows shaped like it."""
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.binomial(1, 1.0 / (1.0 + np.exp(-(2.0 * x["a"] - x["b"])))))
    model = LogisticRegression().fit(x, y)
    return model, x


def partition(rng: np.random.Generator, n: int = 400) -> ValidationPartition:
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.binomial(1, 1.0 / (1.0 + np.exp(-(2.0 * x["a"] - x["b"])))))
    return ValidationPartition(x=x, y=y)


# --- the curve -----------------------------------------------------------------


def test_the_curve_reports_the_row_count_in_every_bin() -> None:
    """The column ``sklearn.calibration.calibration_curve`` does not return, and
    whose absence made a four-loan plot point look like a four-thousand-loan one."""
    scores = pd.Series(np.linspace(0.0, 1.0, 100))
    labels = pd.Series((scores > 0.5).astype(int))

    curve = compute_calibration_curve(labels, scores, n_bins=5)

    assert list(curve["rows"]) == [20, 20, 20, 20, 20]
    assert curve["rows"].sum() == 100


def test_the_curve_is_a_hand_computable_table_on_a_small_frame() -> None:
    """Four rows, two bins. Lower bin: scores 0.1/0.2, labels 0/0. Upper bin:
    scores 0.8/0.9, labels 0/1. So predicted 0.15 against observed 0.0, and
    predicted 0.85 against observed 0.5."""
    labels = pd.Series([0, 0, 0, 1])
    scores = pd.Series([0.1, 0.2, 0.8, 0.9])

    curve = compute_calibration_curve(labels, scores, n_bins=2)

    assert list(curve["mean_predicted_probability"]) == pytest.approx([0.15, 0.85])
    assert list(curve["observed_default_rate"]) == pytest.approx([0.0, 0.5])
    assert list(curve["lower_score"]) == pytest.approx([0.1, 0.8])
    assert list(curve["upper_score"]) == pytest.approx([0.2, 0.9])


def test_a_heavily_tied_score_yields_fewer_bins_rather_than_raising() -> None:
    """A tree model emits a few hundred distinct probabilities over a million
    rows, so quantile edges collide constantly. ``duplicates="drop"`` is why the
    run survives it, and the returned length is how the caller finds out."""
    labels = pd.Series([0] * 50 + [1] * 50)
    scores = pd.Series([0.2] * 50 + [0.8] * 50)

    curve = compute_calibration_curve(labels, scores, n_bins=10)

    assert len(curve) == 2
    assert list(curve["rows"]) == [50, 50]


def test_the_curve_uses_quantile_bins_not_equal_width_ones() -> None:
    """Predicted default probabilities pile up at the bottom of the range, which
    is what makes equal-width binning useless here: 90 of these 100 scores are
    below 0.05, so one equal-width bin in five holds 90% of the portfolio while
    quantile bins hold 20 rows each."""
    scores = pd.Series(np.concatenate([np.linspace(0.001, 0.05, 90), np.linspace(0.5, 0.9, 10)]))
    labels = pd.Series(np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)]))

    curve = compute_calibration_curve(labels, scores, n_bins=5)

    assert list(curve["rows"]) == [20, 20, 20, 20, 20]
    # Equal-width binning, for contrast: 90 rows in the first of five buckets.
    equal_width = np.histogram(scores, bins=5, range=(0.0, 1.0))[0]
    assert equal_width[0] == 90


def test_a_single_bin_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2 bins"):
        compute_calibration_curve(pd.Series([0, 1]), pd.Series([0.2, 0.8]), n_bins=1)


def test_the_curve_inherits_the_shared_input_guards() -> None:
    """Same door as every metric: :func:`risk_score.evaluation.as_metric_arrays`."""
    with pytest.raises(ValueError, match="finite"):
        compute_calibration_curve(pd.Series([0, 1, 1]), pd.Series([0.1, np.nan, 0.9]))
    with pytest.raises(ValueError, match="empty partition"):
        compute_calibration_curve(pd.Series([], dtype=int), pd.Series([], dtype=float))


# --- the ECE -------------------------------------------------------------------


def test_a_perfectly_calibrated_curve_has_zero_expected_error() -> None:
    curve = pd.DataFrame(
        {
            "rows": [100, 100],
            "mean_predicted_probability": [0.2, 0.8],
            "observed_default_rate": [0.2, 0.8],
        }
    )

    assert compute_expected_calibration_error(curve) == 0.0


def test_the_expected_error_is_weighted_by_rows_not_by_bin() -> None:
    """900 rows off by 0.01 and 100 rows off by 0.5. Row-weighted that is 0.059;
    an unweighted mean over bins would report 0.255 and let the sparse tail bin
    speak for the whole portfolio."""
    curve = pd.DataFrame(
        {
            "rows": [900, 100],
            "mean_predicted_probability": [0.20, 0.50],
            "observed_default_rate": [0.21, 1.00],
        }
    )

    assert compute_expected_calibration_error(curve) == pytest.approx(0.059)


def test_the_expected_error_is_signed_absolutely_so_bins_cannot_cancel() -> None:
    """One bin 0.1 too high, one 0.1 too low. The mean *error* is zero and the
    model is not calibrated; the mean *absolute* error is 0.1, which is the claim."""
    curve = pd.DataFrame(
        {
            "rows": [100, 100],
            "mean_predicted_probability": [0.3, 0.7],
            "observed_default_rate": [0.2, 0.8],
        }
    )

    assert compute_expected_calibration_error(curve) == pytest.approx(0.1)


def test_a_curve_missing_a_column_names_the_column() -> None:
    with pytest.raises(ValueError, match=r"missing column\(s\): \['rows'\]"):
        compute_expected_calibration_error(
            pd.DataFrame({"mean_predicted_probability": [0.2], "observed_default_rate": [0.2]})
        )


def test_the_report_bundles_the_curve_with_its_summary() -> None:
    rng = np.random.default_rng(5)
    scores = pd.Series(rng.beta(2.0, 8.0, size=500))
    labels = pd.Series(rng.binomial(1, scores))

    report = build_calibration_report(labels, scores, n_bins=5)

    assert report.bins == len(report.curve) == 5
    assert report.expected_calibration_error == compute_expected_calibration_error(report.curve)
    assert "ece=" in report.summary()
    assert "rows=500" in report.summary()


# --- the fitted correction -----------------------------------------------------


def test_the_method_falls_back_to_platt_scaling_when_positives_are_scarce() -> None:
    """Isotonic on 30 defaults fits a step function that has memorized them, and
    the resulting calibration curve looks *better*, which is why the choice is
    made from the data rather than left to a caller."""
    scarce = pd.Series([1] * (MIN_POSITIVES_FOR_ISOTONIC - 1) + [0] * 1000)
    plentiful = pd.Series([1] * MIN_POSITIVES_FOR_ISOTONIC + [0] * 1000)

    assert choose_calibration_method(scarce) == "sigmoid"
    assert choose_calibration_method(plentiful) == "isotonic"


def test_the_calibrator_wraps_a_frozen_model_and_refits_nothing() -> None:
    """``FrozenEstimator`` is what replaced the removed ``cv="prefit"``. If the
    base model were refitted, its coefficients would move."""
    rng = np.random.default_rng(7)
    model, _ = fitted_model(rng)
    before = model.coef_.copy()

    calibrator = fit_calibrator(model, partition(rng))

    assert isinstance(calibrator, CalibratedClassifierCV)
    # One calibrator, not a CV ensemble: nothing was refitted or cross-split.
    assert len(calibrator.calibrated_classifiers_) == 1
    assert np.array_equal(model.coef_, before)


def test_the_calibrator_actually_changes_the_probabilities() -> None:
    """The whole point of applying rather than only measuring the correction."""
    rng = np.random.default_rng(11)
    model, held_out = fitted_model(rng)

    calibrator = fit_calibrator(model, partition(rng))

    raw = model.predict_proba(held_out)[:, 1]
    calibrated = calibrator.predict_proba(held_out)[:, 1]
    assert not np.allclose(raw, calibrated)


def test_calibrating_a_deliberately_inflated_model_lowers_its_brier_score() -> None:
    """The class-weighted logistic baseline inflated every probability, and the
    old code drew that as a finding instead of correcting it (audit B05). A
    correction that does not improve the Brier score is not doing its job."""
    rng = np.random.default_rng(13)
    n = 3000
    x = pd.DataFrame({"a": rng.normal(size=n)})
    y = pd.Series(rng.binomial(1, 1.0 / (1.0 + np.exp(-(1.5 * x["a"] - 2.0)))))
    inflated = LogisticRegression(class_weight="balanced").fit(x, y)

    validation = ValidationPartition(x=x.iloc[:1500], y=y.iloc[:1500])
    test_x, test_y = x.iloc[1500:], y.iloc[1500:]
    calibrator = fit_calibrator(inflated, validation)

    raw_brier = brier_score_loss(test_y, inflated.predict_proba(test_x)[:, 1])
    calibrated_brier = brier_score_loss(test_y, calibrator.predict_proba(test_x)[:, 1])
    assert calibrated_brier < raw_brier


def test_bare_frames_are_refused_by_the_calibrator() -> None:
    """Same guard as the threshold search, for the same reason: a calibrator
    fitted on test makes the test Brier score a description of the fit.

    The runtime check is load-bearing rather than belt-and-braces: pandas ships
    no type stubs in this project, so ``pd.DataFrame`` is ``Any`` to mypy and this
    call passes static checking untouched - there is not even an error to ignore.
    """
    rng = np.random.default_rng(17)
    model, held_out = fitted_model(rng)

    with pytest.raises(TypeError, match="ValidationPartition"):
        fit_calibrator(model, held_out)


def test_an_unknown_calibration_method_is_refused_with_the_alternatives() -> None:
    rng = np.random.default_rng(19)
    model, _ = fitted_model(rng)

    with pytest.raises(ValueError, match=r"\['isotonic', 'sigmoid'\]"):
        fit_calibrator(model, partition(rng), method="beta")


def test_an_explicit_method_overrides_the_automatic_choice() -> None:
    rng = np.random.default_rng(23)
    model, _ = fitted_model(rng)

    calibrator = fit_calibrator(model, partition(rng), method="isotonic")

    assert calibrator.method == "isotonic"


# --- the plot ------------------------------------------------------------------


def test_the_axes_are_scaled_to_the_book_and_stay_square() -> None:
    """A (0, 1) window puts a whole credit portfolio in one corner of the figure.

    One number is returned for both axes on purpose: the window has to stay
    square or the reference line is no longer at 45 degrees, and the whole chart
    is read as distance from that line.
    """
    tenth = np.array([0.02, 0.09], dtype=np.float64)
    # 0.09 * 1.15 = 0.1035, up to the next tenth.
    assert _axis_extent(tenth, tenth) == pytest.approx(0.2)  # the floor, not 0.2 by rounding
    assert _axis_extent(np.array([0.26]), np.array([0.27])) == pytest.approx(0.4)
    # The observed rate can exceed the predicted one; the window covers both.
    assert _axis_extent(np.array([0.1]), np.array([0.62])) == pytest.approx(0.8)
    # And a model that really does predict near 1.0 gets the full square back.
    assert _axis_extent(np.array([0.05, 0.97]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_a_curve_with_no_rows_does_not_crash_the_axis_scaling() -> None:
    """`plot_calibration_curve` is public and the reduction over an empty array
    would otherwise raise before any message about the empty curve."""
    empty = np.array([], dtype=np.float64)

    assert _axis_extent(empty, empty) == pytest.approx(0.2)


def test_marker_area_encodes_the_count_only_when_the_bins_are_uneven() -> None:
    """Quantile bins differ by a single row, so scaling on that would make every
    marker large and every marker a lie.

    The note is asserted alongside the sizes because it is the part a reader
    sees: it may promise "area proportional to count" only when area carries
    information.
    """
    areas, note = _marker_sizes(np.array([157.0, 156.0]))
    assert len(set(areas)) == 1
    assert note == "156-157 loans per bin"

    # A tie-heavy tree score collapses bins, and then the counts are the point.
    areas, note = _marker_sizes(np.array([400.0, 12.0]))
    assert areas[0] > areas[1]
    assert note == "12-400 loans per bin, marker area proportional to count"

    _, note = _marker_sizes(np.array([35.0, 35.0]))
    assert note == "35 loans per bin"


def test_the_plot_writes_a_file_and_leaves_no_open_figure(tmp_path: Path) -> None:
    """One leaked figure per call used to accumulate for the life of the process,
    which under a retrain endpoint is a memory leak an HTTP request can trigger
    (audit B26)."""
    import matplotlib.pyplot as plt

    curve = build_calibration_report(
        pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.2, 0.8, 0.9]), n_bins=2
    ).curve
    destination = tmp_path / "calibration.png"

    before = len(plt.get_fignums())
    plot_calibration_curve(curve, output_path=destination)

    assert destination.exists()
    assert destination.stat().st_size > 0
    assert len(plt.get_fignums()) == before


def test_the_plot_does_not_reach_into_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It used to ``os.environ.setdefault("MPLCONFIGDIR", ...)`` - a plotting
    helper silently changing process-wide state every later import observes."""
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    curve = build_calibration_report(
        pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.2, 0.8, 0.9]), n_bins=2
    ).curve

    plot_calibration_curve(curve, output_path=tmp_path / "calibration.png")

    assert "MPLCONFIGDIR" not in os.environ


def test_the_plot_closes_the_figure_even_when_saving_fails(tmp_path: Path) -> None:
    """`finally`, not a trailing close: a read-only directory or a full disk must
    not leak the figure on its way out."""
    import matplotlib.pyplot as plt

    curve = build_calibration_report(
        pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.2, 0.8, 0.9]), n_bins=2
    ).curve
    before = len(plt.get_fignums())

    with pytest.raises(Exception):  # noqa: B017 - matplotlib's error type is not the point
        plot_calibration_curve(curve, output_path=tmp_path / "no_such_dir" / "c.png")

    assert len(plt.get_fignums()) == before
