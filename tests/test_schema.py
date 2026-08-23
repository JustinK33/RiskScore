"""Tests for schema normalization: alias resolution and date parsing."""

from __future__ import annotations

import pandas as pd
import pytest

from risk_score.schema import (
    normalize_column_names,
    normalize_credit_schema,
    parse_month_column,
    resolve_column_names,
)


def test_b02_the_canonical_name_beats_an_alias_regardless_of_order() -> None:
    """Reproduces the original failure directly: the frame that produced two
    columns both named `loan_amnt`."""
    for order in (["funded_amnt", "loan_amnt"], ["loan_amnt", "funded_amnt"]):
        frame = pd.DataFrame({name: [1] for name in order})

        result, report = normalize_column_names(frame)

        assert list(result.columns) == ["loan_amnt"]
        assert report.dropped_duplicate_sources == {"funded_amnt": "loan_amnt"}


def test_b02_two_aliases_resolve_by_declared_priority() -> None:
    """Neither source is the canonical name, so the registry's declaration order
    decides - deterministically, not by frame order."""
    rename_map, losers, _unknown = resolve_column_names(["fundedamnt", "loan_amount"])

    assert rename_map == {"loan_amount": "loan_amnt"}
    assert losers == {"fundedamnt": "loan_amnt"}


def test_alias_matching_ignores_case_and_surrounding_whitespace() -> None:
    frame = pd.DataFrame({" Loan_Amnt ": [1], "ANNUAL_INC": [2]})

    result, _report = normalize_column_names(frame)

    assert set(result.columns) == {"loan_amnt", "annual_inc"}


def test_unknown_columns_are_reported_and_kept_by_default() -> None:
    frame = pd.DataFrame({"loan_amnt": [1], "some_new_column": [2]})

    result, report = normalize_column_names(frame)

    assert report.unknown_columns == ("some_new_column",)
    assert "some_new_column" in result.columns


def test_unknown_columns_can_be_dropped() -> None:
    frame = pd.DataFrame({"loan_amnt": [1], "some_new_column": [2]})

    result, report = normalize_column_names(frame, drop_unknown=True)

    assert list(result.columns) == ["loan_amnt"]
    assert report.unknown_columns == ("some_new_column",)


def test_missing_required_columns_are_reported_not_raised() -> None:
    """Reported so the caller decides: an audit of an unfamiliar extract wants
    the list, a training run wants to abort on it."""
    _result, report = normalize_column_names(pd.DataFrame({"loan_amnt": [1]}))

    assert set(report.missing_required) == {"loan_status", "issue_d", "term", "annual_inc"}


def test_config_aliases_are_honored() -> None:
    frame = pd.DataFrame({"borrower_pay": [50_000]})

    result, _report = normalize_column_names(frame, column_aliases={"annual_inc": ["borrower_pay"]})

    assert list(result.columns) == ["annual_inc"]


def test_report_summary_counts_what_happened() -> None:
    frame = pd.DataFrame({"funded_amnt": [1], "loan_amnt": [2], "mystery": [3]})

    _result, report = normalize_column_names(frame)

    assert "dropped_duplicates=1" in report.summary()
    assert "unknown=1" in report.summary()


@pytest.mark.parametrize(
    ("value", "expected", "expected_format"),
    [
        ("Aug-2003", "2003-08-01", "%b-%Y"),
        ("2015-04-01", "2015-04-01", "%Y-%m-%d"),
        ("2015-04", "2015-04-01", "%Y-%m"),
        ("01-Aug-2003", "2003-08-01", "%d-%b-%Y"),
        ("Aug-03", "2003-08-01", "%b-%y"),
        ("August 2003", "2003-08-01", "%B %Y"),
    ],
)
def test_each_supported_date_format_parses_to_the_right_day(
    value: str, expected: str, expected_format: str
) -> None:
    parsed, used = parse_month_column(pd.Series([value] * 4), column_name="issue_d")

    assert parsed.dt.strftime("%Y-%m-%d").unique().tolist() == [expected]
    assert used == expected_format


def test_b28_ambiguous_slash_dates_raise_instead_of_being_guessed() -> None:
    """`format="mixed"` inferred a format per row, so a column of `03/04/2016`
    values could be read as 3 April for some rows and 4 March for others - a
    date column silently wrong for an arbitrary subset, deciding which partition
    each loan lands in."""
    ambiguous = pd.Series(["03/04/2016", "05/06/2016", "07/08/2016"])

    with pytest.raises(ValueError, match="ambiguous and are never guessed"):
        parse_month_column(ambiguous, column_name="issue_d")


def test_b28_a_caller_can_state_the_slash_order_explicitly() -> None:
    parsed, used = parse_month_column(
        pd.Series(["03/04/2016"]), formats=("%d/%m/%Y",), column_name="issue_d"
    )

    assert parsed.dt.strftime("%Y-%m-%d").tolist() == ["2016-04-03"]
    assert used == "%d/%m/%Y"


def test_b28_one_format_applies_to_the_whole_column() -> None:
    """A day-first value cannot slip through a column parsed month-first."""
    parsed, used = parse_month_column(
        pd.Series(["2016-04-03"] * 50), formats=("%Y-%m-%d", "%b-%Y"), column_name="issue_d"
    )

    assert used == "%Y-%m-%d"
    assert parsed.notna().all()


def test_a_few_corrupt_cells_are_quarantined_rather_than_aborting_the_run() -> None:
    values = ["Aug-2003"] * 199 + ["garbage"]

    parsed, used = parse_month_column(pd.Series(values), column_name="issue_d")

    assert used == "%b-%Y"
    assert int(parsed.isna().sum()) == 1


def test_too_many_corrupt_cells_do_abort() -> None:
    values = ["Aug-2003"] * 5 + ["garbage"] * 5

    with pytest.raises(ValueError, match="Could not parse date column"):
        parse_month_column(pd.Series(values), column_name="issue_d")


def test_a_genuinely_mixed_column_parses_by_combining_formats() -> None:
    """Unambiguous formats can be combined safely; ambiguous ones are not
    candidates, so combining can never flip a day and a month."""
    values = ["Aug-2003"] * 5 + ["2011-01-01"] * 5

    parsed, used = parse_month_column(pd.Series(values), column_name="issue_d")

    assert parsed.notna().all()
    assert "+" in used


def test_blank_cells_are_missing_data_not_parse_failures() -> None:
    parsed, _used = parse_month_column(
        pd.Series(["Aug-2003", "", "   ", None]), column_name="issue_d"
    )

    assert parsed.notna().tolist() == [True, False, False, False]


def test_an_entirely_empty_date_column_yields_nat_without_raising() -> None:
    parsed, used = parse_month_column(pd.Series([None, None], dtype="object"), column_name="x")

    assert parsed.isna().all()
    assert used == "none"


def test_b01_every_registered_date_column_is_parsed_not_just_the_split_column() -> None:
    """Only `issue_d` used to be parsed, so `earliest_cr_line` reached the
    preprocessor as a string with 655 distinct values and was one-hot encoded."""
    frame = pd.DataFrame(
        {
            "issue_d": ["Jan-2015"],
            "earliest_cr_line": ["Aug-2003"],
            "loan_status": ["Fully Paid"],
        }
    )

    result, report = normalize_credit_schema(frame)

    assert pd.api.types.is_datetime64_any_dtype(result["earliest_cr_line"])
    assert set(report.date_formats_used) == {"issue_d", "earliest_cr_line"}


def test_normalize_credit_schema_counts_unparseable_dates_per_column() -> None:
    frame = pd.DataFrame({"issue_d": ["Jan-2015"] * 199 + ["nope"]})

    _result, report = normalize_credit_schema(frame)

    assert report.unparseable_dates == {"issue_d": 1}


def test_normalize_credit_schema_is_a_no_op_when_there_are_no_date_columns() -> None:
    frame = pd.DataFrame({"loan_amnt": [1000], "annual_inc": [50_000]})

    result, report = normalize_credit_schema(frame)

    assert list(result.columns) == ["loan_amnt", "annual_inc"]
    assert report.date_formats_used == {}
