"""Tests for parsers and derived features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_score.feature_engineering import (
    MAX_CREDIT_UTILIZATION,
    MIN_ROWS_FOR_SANITY_CHECK,
    build_credit_history_months,
    build_credit_utilization,
    build_dti_clean,
    build_feature_matrix,
    build_fico_band,
    build_loan_to_income_ratio,
    coerce_numeric,
    parse_column,
    parse_declared_columns,
    parse_employment_years,
    parse_percent,
    parse_term_months,
)
from risk_score.features import ParseKind


def test_b24_coerce_numeric_returns_plain_float64_not_a_nullable_dtype() -> None:
    """Int64 and Float64 propagate through arithmetic and then reach sklearn,
    which converts them to object and fails inside a transformer with a message
    that does not name the column."""
    for series in (
        pd.Series([1, 2, pd.NA], dtype="Int64"),
        pd.Series([1.5, 2.5, None], dtype="Float64"),
        pd.Series(["1.5", "2.5", None], dtype="string"),
        pd.Series([1, 2, 3]),
    ):
        assert coerce_numeric(series).dtype == np.dtype("float64")


def test_coerce_numeric_strips_formatting_from_re_exported_extracts() -> None:
    result = coerce_numeric(pd.Series(["$1,200.50", " 13.56% ", "n/a"]))

    assert result.tolist()[:2] == [1200.5, 13.56]
    assert bool(result.isna().iloc[2])


def test_coerce_numeric_removes_infinities_and_sentinels() -> None:
    """inf parses cleanly through to_numeric and then breaks StandardScaler with
    an error about NaN, which sends you looking in the wrong place."""
    result = coerce_numeric(pd.Series(["inf", "-inf", "9999", "-1", "42"]))

    assert result.isna().tolist() == [True, True, True, True, False]
    assert result.iloc[4] == 42.0


def test_parse_percent_converts_to_a_fraction() -> None:
    result = parse_percent(pd.Series(["13.56%", "54.3", None]))

    assert result.iloc[0] == pytest.approx(0.1356)
    assert result.iloc[1] == pytest.approx(0.543)
    assert bool(result.isna().iloc[2])


def test_parse_percent_raises_when_the_source_was_already_a_fraction() -> None:
    """Dividing twice fits fine and is quietly wrong by 100x, so it has to be
    loud rather than inferred away."""
    already_fractional = pd.Series([0.15] * 200)

    with pytest.raises(ValueError, match="already a fraction"):
        parse_percent(already_fractional, column_name="revol_util")


def test_parse_percent_does_not_second_guess_a_single_applicant() -> None:
    """The scale check needs a sample. A one-row /predict request has none, and
    firing on one unusual input would reject a valid request."""
    result = parse_percent(pd.Series([1.0]), column_name="revol_util")

    assert result.iloc[0] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" 36 months", 36.0), ("60 months", 60.0), ("36", 36.0), ("unknown", None)],
)
def test_parse_term_months(raw: str, expected: float | None) -> None:
    result = parse_term_months(pd.Series([raw]))

    if expected is None:
        assert bool(result.isna().iloc[0])
    else:
        assert result.iloc[0] == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("10+ years", 10.0), ("< 1 year", 0.0), ("3 years", 3.0), ("1 year", 1.0), ("n/a", None)],
)
def test_parse_employment_years(raw: str, expected: float | None) -> None:
    """'< 1 year' contains the digit 1, so digit extraction alone would read it
    as a full year of employment."""
    result = parse_employment_years(pd.Series([raw]))

    if expected is None:
        assert bool(result.isna().iloc[0])
    else:
        assert result.iloc[0] == expected


def test_parse_column_dispatches_on_the_declared_kind() -> None:
    assert parse_column(pd.Series(["13.56%"]), ParseKind.PERCENT).iloc[0] == pytest.approx(0.1356)
    assert parse_column(pd.Series([" 36 months"]), ParseKind.TERM_MONTHS).iloc[0] == 36.0
    assert parse_column(pd.Series(["10+ years"]), ParseKind.EMP_LENGTH_YEARS).iloc[0] == 10.0


def test_parse_column_refuses_kinds_that_belong_elsewhere() -> None:
    with pytest.raises(ValueError, match="does not handle"):
        parse_column(pd.Series(["Aug-2003"]), ParseKind.MONTH_DATE, column_name="issue_d")


def test_build_dti_clean_drops_sentinels_and_impossible_values() -> None:
    loans = pd.DataFrame({"dti": ["18.5", "-1", "99999", "0"]})

    result = build_dti_clean(loans)

    assert result.iloc[0] == pytest.approx(18.5)
    assert result.isna().tolist() == [False, True, True, False]


def test_b20_credit_utilization_is_a_fraction_from_either_source_column() -> None:
    """The old code divided revol_util by 100 and the fallback path by nothing,
    so the feature's units depended on which extract was loaded.

    Both paths take fractions now: `parse_declared_columns` has already applied
    the declared PERCENT conversion by the time any builder runs.
    """
    from_revol_util = build_credit_utilization(
        parse_declared_columns(pd.DataFrame({"revol_util": ["54.3%"]}))
    )
    from_totals = build_credit_utilization(
        pd.DataFrame({"total_credit_utilized": [5430.0], "total_credit_limit": [10_000.0]})
    )

    assert from_revol_util.iloc[0] == pytest.approx(0.543)
    assert from_totals.iloc[0] == pytest.approx(0.543)


def test_b20_credit_utilization_is_bounded() -> None:
    loans = parse_declared_columns(pd.DataFrame({"revol_util": ["892%", "-5%", "45%"]}))

    result = build_credit_utilization(loans)

    assert result.iloc[0] == MAX_CREDIT_UTILIZATION  # winsorized, not dropped
    assert bool(result.isna().iloc[1])  # negative utilization is impossible
    assert result.iloc[2] == pytest.approx(0.45)


def test_b20_utilization_in_percentage_units_raises_instead_of_clipping() -> None:
    """A frame that skipped `parse_declared_columns` would otherwise be recorded
    as every borrower sitting exactly on the winsorization ceiling."""
    unparsed = pd.DataFrame({"revol_util": [54.3] * MIN_ROWS_FOR_SANITY_CHECK})

    with pytest.raises(ValueError, match="percentage units"):
        build_credit_utilization(unparsed)


def test_parse_declared_columns_applies_the_percent_conversion_exactly_once() -> None:
    """PERCENT is the one non-idempotent kind: a second pass would give 0.00543."""
    raw = pd.DataFrame({"revol_util": ["54.3%"], "int_rate": ["13.56%"]})

    once = parse_declared_columns(raw)
    twice = parse_declared_columns(once)

    assert once["revol_util"].iloc[0] == pytest.approx(0.543)
    assert once["int_rate"].iloc[0] == pytest.approx(0.1356)
    # Documents the hazard rather than pretending it away: this is why exactly one
    # caller owns the conversion, and why build_credit_utilization now guards it.
    assert twice["revol_util"].iloc[0] == pytest.approx(0.00543)


def test_parse_declared_columns_leaves_dates_and_unregistered_columns_alone() -> None:
    raw = pd.DataFrame(
        {
            "issue_d": ["Mar-2015"],
            "term": [" 36 months"],
            "lender_internal_score": ["7.5"],
        }
    )

    result = parse_declared_columns(raw)

    assert result["issue_d"].iloc[0] == "Mar-2015"  # schema.py owns date parsing
    assert result["term"].iloc[0] == 36.0
    assert result["lender_internal_score"].iloc[0] == "7.5"


def test_credit_utilization_treats_a_zero_limit_as_no_account() -> None:
    """Not as infinite utilization, which is what dividing by zero would say."""
    loans = pd.DataFrame({"total_credit_utilized": [500.0], "total_credit_limit": [0.0]})

    assert bool(build_credit_utilization(loans).isna().iloc[0])


def test_credit_utilization_names_what_it_needs() -> None:
    with pytest.raises(KeyError, match="revol_util"):
        build_credit_utilization(pd.DataFrame({"loan_amnt": [1000]}))


def test_b21_loan_to_income_ratio_rejects_zero_and_negative_income() -> None:
    """The old guard covered only zero, so a negative income - which the extract
    contains - produced a negative ratio that the model read as very low risk."""
    loans = pd.DataFrame({"loan_amnt": [10_000] * 3, "annual_inc": [50_000, 0, -1000]})

    result = build_loan_to_income_ratio(loans)

    assert result.iloc[0] == pytest.approx(0.2)
    assert result.isna().tolist() == [False, True, True]


def test_build_credit_history_months_counts_whole_months() -> None:
    loans = pd.DataFrame(
        {
            "earliest_cr_line": pd.to_datetime(["2003-08-01", "2014-01-01"]),
            "issue_d": pd.to_datetime(["2015-02-01", "2014-01-01"]),
        }
    )

    result = build_credit_history_months(loans)

    assert result.tolist() == [138.0, 0.0]  # Aug 2003 -> Feb 2015 is 11y6m


def test_build_credit_history_months_rejects_a_line_opened_after_origination() -> None:
    loans = pd.DataFrame(
        {
            "earliest_cr_line": pd.to_datetime(["2016-01-01"]),
            "issue_d": pd.to_datetime(["2015-01-01"]),
        }
    )

    assert bool(build_credit_history_months(loans).isna().iloc[0])


def test_build_credit_history_months_refuses_unparsed_dates() -> None:
    loans = pd.DataFrame({"earliest_cr_line": ["Aug-2003"], "issue_d": ["Feb-2015"]})

    with pytest.raises(TypeError, match="must already be datetime"):
        build_credit_history_months(loans)


def test_build_fico_band_puts_a_boundary_score_in_the_upper_band() -> None:
    """670 is published as the floor of 'good', which is not how pd.cut defaults."""
    result = build_fico_band(pd.DataFrame({"fico_midpoint": [579.0, 580.0, 670.0, 800.0]}))

    assert result.tolist() == ["poor", "fair", "good", "exceptional"]


def test_b01_build_feature_matrix_drops_the_raw_columns_it_replaced() -> None:
    """dti and dti_clean both used to reach the preprocessor, and because
    columns were routed by runtime dtype, the raw percent string was one-hot
    encoded."""
    loans = pd.DataFrame(
        {
            "dti": ["18.5"],
            "revol_util": ["54.3%"],
            "loan_amnt": [10_000],
            "annual_inc": [50_000],
            "earliest_cr_line": pd.to_datetime(["2003-08-01"]),
            "issue_d": pd.to_datetime(["2015-02-01"]),
        }
    )

    result = build_feature_matrix(loans)

    assert {"dti_clean", "credit_utilization", "credit_history_months"}.issubset(result.columns)
    for consumed in ("dti", "revol_util", "earliest_cr_line"):
        assert consumed not in result.columns
    # loan_amnt and annual_inc survive: each carries signal beyond their ratio.
    assert {"loan_amnt", "annual_inc"}.issubset(result.columns)


def test_build_feature_matrix_skips_features_whose_inputs_are_absent() -> None:
    """fico_range_* is absent from both real extracts, so requiring it would
    make the pipeline unrunnable on its own dataset."""
    loans = pd.DataFrame({"loan_amnt": [10_000], "annual_inc": [50_000]})

    result = build_feature_matrix(loans)

    assert "loan_to_income_ratio" in result.columns
    assert "fico_midpoint" not in result.columns


def test_build_feature_matrix_can_demand_every_feature() -> None:
    loans = pd.DataFrame({"loan_amnt": [10_000], "annual_inc": [50_000]})

    with pytest.raises(KeyError, match="dti_clean"):
        build_feature_matrix(loans, require_all=True)


def test_p02_build_feature_matrix_leaves_the_input_frame_untouched() -> None:
    """Seven full copies of a 1.8M-row frame produced four columns. One assign
    and one drop now, and the caller's frame is still theirs."""
    loans = pd.DataFrame({"loan_amnt": [10_000], "annual_inc": [50_000]})
    before = list(loans.columns)

    build_feature_matrix(loans)

    assert list(loans.columns) == before


def test_build_feature_matrix_builds_features_that_depend_on_other_features() -> None:
    """fico_band reads fico_midpoint, which is built in the same pass."""
    loans = pd.DataFrame({"fico_range_low": [660], "fico_range_high": [664]})

    result = build_feature_matrix(loans)

    assert result["fico_midpoint"].iloc[0] == pytest.approx(662.0)
    assert result["fico_band"].iloc[0] == "fair"
    # Both raw range columns are consumed by the midpoint.
    assert "fico_range_low" not in result.columns
