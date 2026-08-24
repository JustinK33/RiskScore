"""Tests for PSI and the per-vintage breakdown.

Every PSI assertion here is against a hand-computed constant rather than a range.
PSI is a sum of signed logarithms and there is a whole family of plausible
implementations of it - dropping NaN, closing the outer bin edges, renormalizing
after the zero-count correction - all of which return numbers in the same ballpark
as the correct one. ``assert psi > 0.25`` would pass for most of them. The
arithmetic is written out in the tests that pin it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_score.drift import (
    MISSING_BUCKET,
    PSI_BUCKETS,
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    UNSEEN_BUCKET,
    feature_drift,
    metrics_by_vintage,
    population_stability_index,
    psi_band,
    psi_table,
)
from risk_score.evaluation import compute_auc_roc, compute_brier_score
from risk_score.modeling import engineering_prefix
from risk_score.pipeline import RunResult

# --- the arithmetic ------------------------------------------------------------


def test_a_population_against_itself_scores_exactly_zero() -> None:
    """Not "close to zero". Each bin contributes ``(p - p) * ln(p / p)``, so an
    unchanged population has to give a hard 0.0 - including its empty bins, whose
    floored shares are equal on both sides."""
    population = pd.Series(np.arange(500.0))

    assert population_stability_index(population, population) == 0.0


def test_psi_matches_the_arithmetic_written_out_by_hand() -> None:
    """50/50 shifting to 25/75, computed longhand.

    The empty ``__unseen__`` and ``__missing__`` buckets are floored to 0.5/100 on
    both sides, so they contribute exactly nothing and the whole value comes from
    the two real levels.
    """
    reference = pd.Series(["a"] * 50 + ["b"] * 50)
    comparison = pd.Series(["a"] * 25 + ["b"] * 75)

    expected = (0.25 - 0.50) * np.log(0.25 / 0.50) + (0.75 - 0.50) * np.log(0.75 / 0.50)

    assert expected == pytest.approx(0.27465307216702742)
    assert population_stability_index(reference, comparison) == pytest.approx(expected)
    assert psi_band(population_stability_index(reference, comparison)) == "significant"


def test_the_scalar_is_the_sum_of_the_tables_contributions() -> None:
    """The table is the working; the scalar must be its total, or the published
    ``psi_score.csv`` would not explain the published PSI."""
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(size=400))
    comparison = pd.Series(rng.normal(loc=0.4, size=400))

    table = psi_table(reference, comparison)

    assert population_stability_index(reference, comparison) == pytest.approx(
        float(table["psi_contribution"].sum())
    )


def test_every_bucket_contributes_a_non_negative_amount() -> None:
    """``(p - q) * ln(p / q)`` carries the sign of ``p - q`` twice, so a negative
    contribution would mean the formula was implemented as a plain KL divergence -
    which is asymmetric and can cancel across bins."""
    rng = np.random.default_rng(1)
    table = psi_table(pd.Series(rng.normal(size=300)), pd.Series(rng.gamma(2.0, size=300)))

    assert (table["psi_contribution"] >= 0).all()


def test_an_empty_bin_is_floored_at_half_an_observation() -> None:
    """The zero-count correction, at its exact value. ``ln(0)`` is what this
    prevents, and an arbitrary 1e-6 would make the result depend on a constant
    nobody could justify."""
    reference = pd.Series([1.0] * 100)
    comparison = pd.Series([1.0] * 100)

    table = psi_table(reference, comparison)
    missing = table[table["bucket"] == MISSING_BUCKET].iloc[0]

    assert missing["reference_count"] == 0
    assert missing["reference_share"] == pytest.approx(0.5 / 100)


# --- what counts as a bucket ---------------------------------------------------


def test_a_level_the_reference_never_showed_lands_in_the_unseen_bucket() -> None:
    """The one-hot encoder maps an unseen level to nothing at all, so this is the
    most actionable row a drift table can have."""
    reference = pd.Series(["a"] * 90 + ["b"] * 10)
    comparison = pd.Series(["a"] * 50 + ["z"] * 50)

    table = psi_table(reference, comparison).set_index("bucket")

    assert table.loc[UNSEEN_BUCKET, "comparison_count"] == 50
    assert table.loc[UNSEEN_BUCKET, "reference_count"] == 0
    assert psi_band(float(table["psi_contribution"].sum())) == "significant"


def test_a_shift_in_missingness_alone_is_drift() -> None:
    """The observed values are identical in both populations here. Dropping NaN -
    the obvious implementation - would report a PSI of 0.0 for a feature that went
    from fully populated to 40% absent."""
    reference = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    comparison = pd.Series([1.0, 2.0, 3.0] * 20 + [np.nan] * 40)

    table = psi_table(reference, comparison).set_index("bucket")

    assert table.loc[MISSING_BUCKET, "comparison_count"] == 40
    assert population_stability_index(reference, comparison) > PSI_SIGNIFICANT


def test_an_infinity_is_counted_as_missing_rather_than_as_an_extreme() -> None:
    """An infinity is not a value the reference quantiles can place, and putting it
    in the top bin would claim it was merely large."""
    reference = pd.Series(np.arange(100.0))
    comparison = pd.Series([np.inf] * 10 + list(np.arange(90.0)))

    table = psi_table(reference, comparison).set_index("bucket")

    assert table.loc[MISSING_BUCKET, "comparison_count"] == 10


def test_a_value_beyond_the_reference_range_is_still_counted() -> None:
    """The outer bin edges are open. Closed edges would drop these rows, shrinking
    the comparison's total and *understating* the drift they represent - the exact
    wrong direction for a monitoring metric to fail in."""
    reference = pd.Series(np.arange(100.0))
    comparison = pd.Series([-500.0] * 10 + [900.0] * 10 + list(np.arange(80.0)))

    table = psi_table(reference, comparison)

    assert int(table["comparison_count"].sum()) == len(comparison)
    assert int(table["reference_count"].sum()) == len(reference)


# --- binning behaviour ---------------------------------------------------------


def test_duplicate_quantiles_collapse_and_the_table_says_how_many_bins_it_used() -> None:
    """`pub_rec` is zero for most borrowers, so its 10th through 70th percentiles
    are all 0. Ten distinct bins are not available and the table must not imply
    they were."""
    reference = pd.Series([0.0] * 90 + [1.0] * 5 + [2.0] * 5)

    table = psi_table(reference, reference, buckets=10)

    # Real bins, plus the missingness bucket.
    assert 2 <= len(table) - 1 < 10


def test_a_constant_reference_cannot_detect_a_shift_and_says_zero() -> None:
    """Every comparison value lands in the single bin ``(-inf, inf)``. Zero is the
    honest answer, not a defect: with no spread in the reference there is no
    distribution to compare against."""
    reference = pd.Series([7.0] * 100)
    comparison = pd.Series(np.arange(100.0))

    assert population_stability_index(reference, comparison) == 0.0


def test_the_default_bucket_count_is_the_credit_risk_convention() -> None:
    reference = pd.Series(np.arange(1000.0))

    assert PSI_BUCKETS == 10
    assert len(psi_table(reference, reference)) == PSI_BUCKETS + 1


def test_fewer_than_two_buckets_is_refused() -> None:
    reference = pd.Series(np.arange(100.0))

    with pytest.raises(ValueError, match="at least 2"):
        psi_table(reference, reference, buckets=1)


def test_an_empty_population_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        psi_table(pd.Series(np.arange(10.0)), pd.Series(dtype=float))


# --- the bands -----------------------------------------------------------------


def test_the_bands_are_the_conventional_ones_at_their_exact_edges() -> None:
    """The edges are inclusive-below: 0.10 is already moderate. Stated by test
    because a report that flips at 0.1000001 instead is indistinguishable by eye
    and differs on real data."""
    assert (PSI_MODERATE, PSI_SIGNIFICANT) == (0.10, 0.25)
    assert psi_band(0.0) == "stable"
    assert psi_band(0.0999) == "stable"
    assert psi_band(PSI_MODERATE) == "moderate"
    assert psi_band(0.2499) == "moderate"
    assert psi_band(PSI_SIGNIFICANT) == "significant"
    assert psi_band(3.0) == "significant"


def test_a_non_finite_psi_is_refused_rather_than_banded() -> None:
    """A NaN PSI means the caller has a bug upstream, and `nan < 0.10` is False, so
    an unguarded comparison would silently report it as `significant`."""
    with pytest.raises(ValueError, match="finite"):
        psi_band(float("nan"))


# --- per-feature ---------------------------------------------------------------


def _halves(trained_run: RunResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two engineered halves of the run's own extract.

    Engineered rather than raw, because that is the frame `feature_drift` is
    specified against, and rather than the design matrix, because a PSI per
    one-hot column is not a readable table.
    """
    raw = pd.read_csv(trained_run.metadata.dataset_path)
    engineer = engineering_prefix(trained_run.bundle.pipeline)
    middle = len(raw) // 2
    return engineer.transform(raw.head(middle)), engineer.transform(raw.tail(middle))


def test_there_is_exactly_one_row_per_model_feature(trained_run: RunResult) -> None:
    """Driven by the spec, so a feature the model uses cannot be left out of the
    monitoring table and a column it ignores cannot raise an alarm."""
    reference, comparison = _halves(trained_run)
    spec = trained_run.bundle.feature_spec

    drift = feature_drift(reference, comparison, spec=spec)

    assert sorted(drift["feature"]) == sorted(spec.model_features)
    assert list(drift.columns) == [
        "feature",
        "label",
        "kind",
        "psi",
        "band",
        "buckets",
        "missing_rate_reference",
        "missing_rate_comparison",
        "rank",
    ]


def test_the_table_is_ranked_worst_first(trained_run: RunResult) -> None:
    reference, comparison = _halves(trained_run)

    drift = feature_drift(reference, comparison, spec=trained_run.bundle.feature_spec)

    assert drift["psi"].is_monotonic_decreasing
    assert list(drift["rank"]) == list(range(1, len(drift) + 1))
    assert list(drift["band"]) == [psi_band(value) for value in drift["psi"]]


def test_each_rows_psi_is_the_psi_of_that_column(trained_run: RunResult) -> None:
    """The per-feature loop and the scalar function must not be two implementations."""
    reference, comparison = _halves(trained_run)

    drift = feature_drift(reference, comparison, spec=trained_run.bundle.feature_spec)

    for feature in ("loan_amnt", "purpose"):
        row = drift.set_index("feature").loc[feature]
        assert row["psi"] == pytest.approx(
            population_stability_index(reference[feature], comparison[feature])
        )
    assert drift.set_index("feature").loc["purpose", "kind"] == "categorical"
    assert drift.set_index("feature").loc["loan_amnt", "kind"] == "numeric"


def test_the_missing_rates_are_the_frames_own(trained_run: RunResult) -> None:
    """Reported per side, because a feature can be stable in its observed values
    and still be the reason a batch of predictions went strange."""
    reference, comparison = _halves(trained_run)

    drift = feature_drift(reference, comparison, spec=trained_run.bundle.feature_spec).set_index(
        "feature"
    )

    assert drift.loc["annual_inc", "missing_rate_reference"] == pytest.approx(
        float(reference["annual_inc"].isna().mean())
    )
    assert drift.loc["annual_inc", "missing_rate_comparison"] == pytest.approx(
        float(comparison["annual_inc"].isna().mean())
    )


def test_a_frame_missing_a_declared_feature_is_refused_by_name(trained_run: RunResult) -> None:
    """Naming the column and the side, because the alternative is a KeyError from
    inside pandas that says only the column name."""
    reference, comparison = _halves(trained_run)

    with pytest.raises(KeyError, match=r"loan_amnt.*comparison"):
        feature_drift(
            reference,
            comparison.drop(columns=["loan_amnt"]),
            spec=trained_run.bundle.feature_spec,
        )


# --- per vintage ---------------------------------------------------------------


def _vintage_frame() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Two vintages: 2013 separable and mixed, 2014 all defaults."""
    labels = pd.Series([0, 0, 1, 1, 1, 1])
    scores = pd.Series([0.1, 0.2, 0.7, 0.8, 0.6, 0.9])
    dates = pd.Series(pd.to_datetime(["2013-01-01"] * 4 + ["2014-06-01"] * 2))
    return labels, scores, dates


def test_a_vintage_row_carries_the_counts_and_the_metrics_of_that_year() -> None:
    labels, scores, dates = _vintage_frame()

    table = metrics_by_vintage(labels, scores, dates=dates, threshold=0.5).set_index("vintage")

    assert list(table.index) == [2013, 2014]
    assert table.loc[2013, "rows"] == 4
    assert table.loc[2013, "defaults"] == 2
    assert table.loc[2013, "default_rate"] == pytest.approx(0.5)
    # Two 0s below two 1s: perfectly ranked.
    assert table.loc[2013, "auc_roc"] == pytest.approx(1.0)
    # Approved is `score < threshold`, matching the serving rule.
    assert table.loc[2013, "approval_rate"] == pytest.approx(0.5)
    assert table.loc[2013, "brier_score"] == pytest.approx(
        compute_brier_score(labels.head(4), scores.head(4))
    )


def test_a_single_class_vintage_gets_nan_for_the_ranking_metrics_not_an_error() -> None:
    """A partition boundary lands mid-year often enough that raising here would
    mean never producing the table. The counts stay, so the NaN is explicable."""
    labels, scores, dates = _vintage_frame()

    row = (
        metrics_by_vintage(labels, scores, dates=dates, threshold=0.5)
        .set_index("vintage")
        .loc[2014]
    )

    assert row["rows"] == 2
    assert row["defaults"] == 2
    assert np.isnan(row["auc_roc"])
    assert np.isnan(row["ks_statistic"])
    # Calibration is still measurable on one class: "you said 0.6 and it defaulted".
    assert row["brier_score"] == pytest.approx(compute_brier_score(labels.tail(2), scores.tail(2)))


def test_the_vintage_metrics_agree_with_the_headline_helpers() -> None:
    """Same functions as the headline metrics, so a per-vintage AUC and the overall
    AUC cannot be computed two different ways."""
    labels, scores, dates = _vintage_frame()

    table = metrics_by_vintage(labels, scores, dates=dates, threshold=0.5)

    assert table.set_index("vintage").loc[2013, "auc_roc"] == pytest.approx(
        compute_auc_roc(labels.head(4), scores.head(4))
    )


def test_the_partition_label_is_a_column_when_asked_for_and_absent_when_not() -> None:
    """One table covers train, validation, and test, and a reader has to be able to
    tell which rows are which."""
    labels, scores, dates = _vintage_frame()

    labelled = metrics_by_vintage(labels, scores, dates=dates, threshold=0.5, partition="test")
    plain = metrics_by_vintage(labels, scores, dates=dates, threshold=0.5)

    assert next(iter(labelled.columns)) == "partition"
    assert set(labelled["partition"]) == {"test"}
    assert "partition" not in plain.columns


def test_dates_that_do_not_line_up_with_the_labels_are_refused() -> None:
    """Aligning by position would pair a 2013 loan's date with a 2015 loan's
    outcome, and every number in the table would still look plausible."""
    labels, scores, dates = _vintage_frame()

    with pytest.raises(ValueError, match="share an index"):
        metrics_by_vintage(labels, scores, dates=dates.set_axis(np.arange(100, 106)), threshold=0.5)


def test_a_row_with_no_origination_date_is_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Impossible downstream of a time split, which cannot place a row without a
    date. Counted rather than left to become a NaN group that reads like a cohort."""
    labels, scores, dates = _vintage_frame()
    dates = dates.copy()
    dates.iloc[-1] = pd.NaT

    with caplog.at_level("WARNING"):
        table = metrics_by_vintage(labels, scores, dates=dates, threshold=0.5)

    assert int(table["rows"].sum()) == len(labels) - 1
    assert "1 row(s) with no origination date" in caplog.text


# --- what a run publishes ------------------------------------------------------


def test_the_run_publishes_both_psi_tables_and_the_headline_numbers(
    trained_run: RunResult,
) -> None:
    """The artifact contract for monitoring: the scalar in `metrics.json` has to be
    the sum of the table beside it, or the dashboard number and the drilldown
    disagree."""
    score = pd.read_csv(trained_run.run_dir / "psi_score.csv")
    features = pd.read_csv(trained_run.run_dir / "psi_features.csv")
    payload = trained_run.payload

    assert payload["psi_score"] == pytest.approx(float(score["psi_contribution"].sum()))
    assert payload["psi_score_band"] == psi_band(payload["psi_score"])
    # Reference and comparison are named, because a PSI without them is a number
    # about nothing in particular.
    assert (payload["drift_reference"], payload["drift_comparison"]) == ("train", "test")
    assert payload["psi_feature_worst"] == features["feature"].iloc[0]
    assert payload["psi_feature_worst_value"] == pytest.approx(features["psi"].iloc[0])
    assert payload["psi_features_unstable"] == list(
        features.loc[features["band"] != "stable", "feature"]
    )


def test_the_run_publishes_a_vintage_table_covering_all_three_partitions(
    trained_run: RunResult,
) -> None:
    """One headline AUC over a multi-year window hides whether the model works in
    every year of it. Train is in the table too: without it the default-rate series
    starts mid-history and the embargo's flattening is invisible."""
    vintages = pd.read_csv(trained_run.run_dir / "metrics_by_vintage.csv")

    assert set(vintages["partition"]) == {"train", "validation", "test"}
    assert list(vintages.columns)[:5] == [
        "partition",
        "vintage",
        "rows",
        "defaults",
        "default_rate",
    ]
    # Every labelled row is accounted for exactly once, so no partition is silently
    # missing a vintage.
    for partition, rows in (
        ("train", trained_run.payload["rows_train"]),
        ("validation", trained_run.payload["rows_validation"]),
        ("test", trained_run.payload["rows_test"]),
    ):
        assert int(vintages.loc[vintages["partition"] == partition, "rows"].sum()) == rows
