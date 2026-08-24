"""Tests for the model card, the variant comparison, and the dashboard shapes.

The card is asserted by *claim*, not by layout: that a number which is present in
the payload reaches the document, that a number which is absent renders as ``-``
rather than as zero, and that no unsubstituted placeholder can survive. Asserting
the exact markdown would make every prose edit a test failure, which trains people
to update the fixture without reading it.

The comparison is asserted by sign and baseline, because those are the two things
a reader acts on and the two things a refactor can silently invert.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from risk_score.artifacts import RunMetadata
from risk_score.reporting import (
    COMPARISON_METRICS,
    SANITY_MIN_TEST_POSITIVES,
    SANITY_MIN_TEST_ROWS,
    columnar,
    comparison_payload,
    comparison_table,
    format_metric,
    render_model_card,
    sanity_warnings,
    variant_label,
)


def make_metadata(**overrides: Any) -> RunMetadata:
    defaults: dict[str, Any] = {
        "run_id": "20260101T000000Z-logistic_regression-origination_only-abc1234",
        "created_at": "2026-01-01T00:00:00Z",
        "model_type": "logistic_regression",
        "feature_tier": "origination_only",
        "git_commit": "abc1234",
        "dataset_path": "data/raw/1/loan.csv",
        "dataset_sha256": "0123456789abcdef",
        "dataset_bytes": 1_190_000_000,
        "target_definition": "Charged Off or Default -> 1, Fully Paid -> 0",
        "rows": {"read": 2000, "closed": 1600, "matured": 1400},
        "split_windows": {"train": "2013-01-01 .. 2014-09-30"},
        "embargo": {
            "summary": "removed 200 immature loans",
            "default_rate_by_vintage_before": {"2016": 0.243},
            "default_rate_by_vintage_after": {"2016": 0.149},
        },
        "cost_matrix": {"false_negative": 5.0, "false_positive": 1.0},
        "features": {
            "summary": "18 numeric, 4 categorical",
            "numeric": ["loan_amnt", "dti"],
            "categorical": ["addr_state"],
            "leakage": "0 post-origination columns admitted",
        },
        "library_versions": {"scikit-learn": "1.9.0"},
    }
    return RunMetadata(**{**defaults, **overrides})


def make_payload(**overrides: Any) -> dict[str, Any]:
    """A ``metrics.json`` payload that passes every sanity check.

    Healthy by default so each test can break exactly one thing, which is what
    makes a failure message point at the check it broke.
    """
    defaults: dict[str, Any] = {
        "run_id": "20260101T000000Z-logistic_regression-origination_only-abc1234",
        "model_type": "logistic_regression",
        "include_lender_priced": False,
        "rows_train": 8000,
        "rows_validation": 3000,
        "rows_test": 4000,
        "positives_train": 1200,
        "positives_validation": 450,
        "positives_test": 600,
        "auc_roc": 0.68,
        "average_precision": 0.31,
        "ks_statistic": 0.26,
        "brier_score": 0.108,
        "brier_score_uncalibrated": 0.121,
        "expected_calibration_error": 0.014,
        "selected_threshold": 0.183,
        "selected_threshold_total_cost": 1420.0,
        "threshold_selected_on": "validation",
        "approval_rate": 0.81,
        "false_negative_cost": 5.0,
        "false_positive_cost": 1.0,
        "psi_score": 0.04,
        "psi_score_band": "stable",
        "top_features": ["dti", "loan_amnt", "addr_state"],
        "explainer": "linear",
    }
    return {**defaults, **overrides}


# --- format_metric -------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.6812345, "0.6812"),
        (0, "0.0000"),
        (None, "-"),
        (float("nan"), "-"),
        (float("inf"), "-"),
        (np.float64("nan"), "-"),
        ("0.68", "-"),
        # `True` is an `int`, and a truthy flag printed as `1.0000` in a metric
        # column reads as a perfect score.
        (True, "-"),
    ],
)
def test_a_missing_metric_reads_as_a_dash_and_never_as_zero(value: Any, expected: str) -> None:
    assert format_metric(value) == expected


def test_zero_is_a_measurement_and_stays_a_number() -> None:
    """The distinction the whole function exists for: 0.0 was measured, `-` was
    not measured, and rendering the second as the first is how a broken run reads
    as a catastrophic model."""
    assert format_metric(0.0) != format_metric(None)


def test_the_digit_count_is_honoured() -> None:
    assert format_metric(0.123456, digits=2) == "0.12"


# --- sanity warnings -----------------------------------------------------------


def test_a_healthy_run_produces_no_warnings() -> None:
    assert sanity_warnings(make_payload()) == []


def test_a_small_test_window_is_reported_with_its_own_numbers() -> None:
    """The committed artifact this project replaced: 116 rows, one positive, a
    0.07 AUC rendered as a result."""
    warnings = sanity_warnings(make_payload(rows_test=116, positives_test=1))

    assert len(warnings) == 2
    assert "116" in warnings[0] and str(SANITY_MIN_TEST_ROWS) in warnings[0]
    assert "1 default" in warnings[1] and str(SANITY_MIN_TEST_POSITIVES) in warnings[1]


def test_significant_score_drift_is_a_warning() -> None:
    warnings = sanity_warnings(make_payload(psi_score=0.31, psi_score_band="significant"))

    assert len(warnings) == 1
    assert "0.3100" in warnings[0]


def test_a_moderate_psi_band_is_not_a_warning() -> None:
    """Only `significant` earns a banner. Warning on `moderate` would mean the
    banner is present on most honest runs, and a banner nobody can clear is a
    banner nobody reads."""
    assert sanity_warnings(make_payload(psi_score=0.15, psi_score_band="moderate")) == []


def test_calibration_that_made_the_brier_score_worse_is_reported() -> None:
    warnings = sanity_warnings(make_payload(brier_score=0.131, brier_score_uncalibrated=0.121))

    assert len(warnings) == 1
    assert "0.1210" in warnings[0] and "0.1310" in warnings[0]


def test_the_lender_priced_tier_is_always_a_warning() -> None:
    """Not a defect, but a run that cannot score an unpriced applicant must not be
    read as a screening model (ADR 0005)."""
    warnings = sanity_warnings(make_payload(include_lender_priced=True))

    assert len(warnings) == 1
    assert "int_rate" in warnings[0]


# --- the model card ------------------------------------------------------------


def test_the_card_states_the_run_identity_and_the_measured_numbers() -> None:
    card = render_model_card(make_payload(), make_metadata())

    assert card.startswith("# Model card: `20260101T000000Z-logistic_regression")
    assert card.endswith("\n")
    for expected in (
        "0.6800",  # AUC
        "0.1080",  # Brier
        "0.1830",  # threshold
        "abc1234",  # git commit
        "5.0:1",  # cost ratio
        "0.243",  # embargo, before
        "0.149",  # embargo, after
        "scikit-learn",
    ):
        assert expected in card, expected


def test_every_documented_section_is_present() -> None:
    """The card is a contract with a reader, so a section that stops being written
    is a regression rather than a formatting change."""
    card = render_model_card(make_payload(), make_metadata())

    for heading in (
        "## Intended use",
        "## Out of scope",
        "## Training data",
        "## Features",
        "## Metrics",
        "## Population stability",
        "## What the model uses",
        "## Limitations",
        "## Ethical considerations",
        "## Reproducing this run",
    ):
        assert heading in card, heading


def test_the_shap_section_is_omitted_when_the_run_recorded_no_ranking() -> None:
    """An older run without a SHAP summary gets a card without that section rather
    than a heading over an empty table."""
    card = render_model_card(make_payload(top_features=[]), make_metadata())

    assert "## What the model uses" not in card
    assert "## Limitations" in card


def test_no_unsubstituted_placeholder_can_reach_the_card() -> None:
    """The plan's acceptance check, kept as a test even though generating the
    markdown in code is what makes it pass: it also catches an f-string that was
    quoted by accident."""
    card = render_model_card(make_payload(), make_metadata())

    assert "$" not in card
    assert "{" not in card and "}" not in card


def test_a_card_renders_from_an_almost_empty_payload() -> None:
    """The failure mode that matters most: a metric this build added and an older
    run never wrote must render as `-`, not raise and not print `0.0000`.
    """
    card = render_model_card({"run_id": "r"}, make_metadata(embargo={}, features={}, rows={}))

    assert "0.0000" not in card
    assert card.count("-") > 0
    assert "## Metrics" in card


def test_the_warning_block_sits_under_the_headline_numbers() -> None:
    """Placement is the point. A caveat below the metrics is a caveat somebody
    reads after quoting them."""
    card = render_model_card(make_payload(rows_test=116, positives_test=1), make_metadata())

    assert "## Read this first" in card
    assert card.index("## Read this first") < card.index("## Intended use")
    assert "116" in card


def test_a_healthy_run_gets_no_warning_block_at_all() -> None:
    assert "## Read this first" not in render_model_card(make_payload(), make_metadata())


def test_the_vintage_table_is_rendered_when_supplied_and_omitted_otherwise() -> None:
    vintages = pd.DataFrame(
        {
            "vintage": [2015, 2016],
            "rows": [1200, 900],
            "positives": [180, 140],
            "auc_roc": [0.67, float("nan")],
        }
    )

    with_vintages = render_model_card(make_payload(), make_metadata(), vintages=vintages)
    without = render_model_card(make_payload(), make_metadata())

    assert "2016" in with_vintages
    # The single-class vintage: no AUC is legitimate and must not print as zero.
    assert "0.0000" not in with_vintages
    assert "## Performance by origination year" not in without


def test_the_tier_note_changes_with_the_tier() -> None:
    excluded = render_model_card(make_payload(), make_metadata())
    included = render_model_card(
        make_payload(include_lender_priced=True),
        make_metadata(feature_tier="with_lender_priced"),
    )

    assert "**excludes** the lender-priced tier" in excluded
    assert "**admits** the lender-priced tier" in included


# --- the comparison ------------------------------------------------------------


def variants() -> list[dict[str, Any]]:
    """Two models under both tiers, model-major, as `compare_runs` orders them."""
    return [
        make_payload(run_id="lr-excl", auc_roc=0.68),
        make_payload(run_id="lr-incl", include_lender_priced=True, auc_roc=0.74),
        make_payload(run_id="xgb-excl", model_type="xgboost", auc_roc=0.71, brier_score=0.101),
        make_payload(
            run_id="xgb-incl", model_type="xgboost", include_lender_priced=True, auc_roc=0.79
        ),
    ]


def test_a_variant_label_names_the_model_and_the_tier() -> None:
    assert variant_label(make_payload()) == "logistic_regression / origination_only"
    assert variant_label(make_payload(include_lender_priced=True)).endswith("with_lender_priced")


def test_the_baseline_is_the_first_variant_and_not_the_best_one() -> None:
    """A comparison answers "what does moving away from the default buy". If the
    winner defined the baseline, every delta would flip sign the moment a
    different variant won, and a reader comparing two comparisons would see
    improvements where nothing changed.
    """
    table = comparison_table(variants())

    assert table["variant"].iloc[0] == "logistic_regression / origination_only"
    assert table["auc_roc_delta"].iloc[0] == 0.0
    assert table["auc_roc_delta"].iloc[2] == pytest.approx(0.71 - 0.68)
    assert comparison_payload(variants())["baseline"] == table["variant"].iloc[0]
    assert comparison_payload(variants())["best_by_auc_roc"]["run_id"] == "xgb-incl"


def test_a_delta_is_signed_in_the_metrics_own_units() -> None:
    """No orientation applied. A lower Brier is better, so its delta is negative
    for the better variant, and `metrics` in the payload is what says so."""
    table = comparison_table(variants())
    payload = comparison_payload(variants())

    assert table["brier_score_delta"].iloc[2] == pytest.approx(0.101 - 0.108)
    assert payload["metrics"]["brier_score"] == "lower"
    assert payload["metrics"]["auc_roc"] == "higher"


def test_a_neutral_metric_gets_no_delta_column() -> None:
    """`approval_rate` is a policy consequence, not a score. A delta column would
    invite reading a difference as an improvement."""
    table = comparison_table(variants())

    assert "approval_rate" in table.columns
    assert "approval_rate_delta" not in table.columns
    for name, direction in COMPARISON_METRICS:
        assert (f"{name}_delta" in table.columns) == (direction != "neutral")


def test_the_leakage_cost_is_quantified_per_model_type() -> None:
    """The whole point of `--tiers both`: ADR 0005's policy defended with a
    measurement rather than an assertion."""
    deltas = {
        entry["model_type"]: entry
        for entry in comparison_payload(variants())["lender_priced_delta"]
    }

    assert set(deltas) == {"logistic_regression", "xgboost"}
    assert deltas["logistic_regression"]["auc_roc_gain"] == pytest.approx(0.74 - 0.68)
    assert deltas["xgboost"]["auc_roc_gain"] == pytest.approx(0.79 - 0.71)
    assert deltas["xgboost"]["run_id_origination_only"] == "xgb-excl"


def test_the_leakage_cost_is_absent_rather_than_zero_when_one_tier_was_fitted() -> None:
    """Reporting a gain of 0.0 would be a claim that admitting the lender's price
    changes nothing, which is the opposite of what one tier measures."""
    payload = comparison_payload([make_payload(), make_payload(model_type="xgboost")])

    assert payload["lender_priced_delta"] == []


def test_a_comparison_of_one_variant_is_a_table_with_zero_deltas() -> None:
    table = comparison_table([make_payload()])

    assert len(table) == 1
    assert table["auc_roc_delta"].iloc[0] == 0.0


def test_an_empty_comparison_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one variant"):
        comparison_table([])


def test_a_variant_with_no_auc_does_not_become_the_winner() -> None:
    payload = comparison_payload(
        [make_payload(run_id="broken", auc_roc=None), make_payload(run_id="fine", auc_roc=0.6)]
    )

    assert payload["best_by_auc_roc"]["run_id"] == "fine"


def test_a_comparison_with_no_auc_at_all_reports_no_winner() -> None:
    assert comparison_payload([make_payload(auc_roc=None)])["best_by_auc_roc"] is None


def test_the_comparison_payload_is_valid_json_with_nulls_for_missing_numbers() -> None:
    """`json.dumps` writes a bare `NaN` token by default, which `JSON.parse`
    rejects - so one missing metric would take out a dashboard panel."""
    payload = comparison_payload([make_payload(auc_roc=float("nan"), psi_score=None)])

    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text
    variant = json.loads(text)["variants"][0]
    assert variant["auc_roc"] is None
    assert variant["psi_score"] is None


# --- dashboard shapes ----------------------------------------------------------


def test_columnar_returns_one_list_per_column_and_survives_json() -> None:
    frame = pd.DataFrame(
        {
            "threshold": [0.1, 0.2],
            "total_cost": [12.0, float("nan")],
            "approved": [True, False],
            "vintage": pd.to_datetime(["2015-01-01", "2015-02-01"]),
        }
    )

    payload = columnar(frame)

    assert payload["threshold"] == [0.1, 0.2]
    assert payload["total_cost"] == [12.0, None]
    assert payload["approved"] == [True, False]
    assert payload["vintage"][0].startswith("2015-01-01")
    assert "NaN" not in json.dumps(payload, allow_nan=False)


def test_columnar_leaves_no_numpy_scalar_behind() -> None:
    """A numpy scalar is not JSON-serializable, and pandas hands them back from
    `tolist()` on some dtypes."""
    payload = columnar(pd.DataFrame({"n": np.array([1, 2], dtype=np.int64)}))

    assert all(type(value) is int for value in payload["n"])
    assert not any(isinstance(value, float) and math.isnan(value) for value in payload["n"])
