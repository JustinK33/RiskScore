"""Tests for raw ingestion: projected reads, status filtering, and the embargo."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from risk_score.data_loading import (
    CLOSED_LOAN_STATUSES,
    apply_outcome_maturity_embargo,
    create_default_target,
    filter_to_closed_loans,
    filter_to_terms,
    load_lending_club_data,
    read_raw_loans,
    term_months,
)
from risk_score.schema import normalize_credit_schema


def test_load_lending_club_data_filters_to_closed_loans(tmp_path: Path) -> None:
    raw_path = tmp_path / "loans.csv"
    pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Current", "Charged Off", "In Grace Period"],
            "loan_amnt": [1000, 2000, 3000, 4000],
        }
    ).to_csv(raw_path, index=False)

    result = load_lending_club_data(raw_path)

    assert result["loan_status"].tolist() == ["Fully Paid", "Charged Off"]


def test_load_lending_club_data_normalizes_column_aliases(tmp_path: Path) -> None:
    raw_path = tmp_path / "loans.csv"
    pd.DataFrame(
        {
            "status": ["Fully Paid", "Current", "Charged Off"],
            "loan_amount": [1000, 2000, 3000],
            "annual_income": [50_000, 60_000, 40_000],
            "debt_to_income": [10.0, 12.0, 20.0],
            "issue_month": ["Jan-2018", "Feb-2018", "Mar-2018"],
        }
    ).to_csv(raw_path, index=False)

    result = load_lending_club_data(raw_path)

    assert result["loan_status"].tolist() == ["Fully Paid", "Charged Off"]
    assert {"loan_amnt", "annual_inc", "dti", "issue_d"}.issubset(result.columns)
    assert result["issue_d"].dt.strftime("%Y-%m-%d").tolist() == ["2018-01-01", "2018-03-01"]


def test_b02_a_frame_with_both_loan_amnt_and_funded_amnt_yields_one_column(
    tmp_path: Path,
) -> None:
    """The standard extract has both. First-source-wins renamed `funded_amnt` to
    `loan_amnt` and then skipped the real one, leaving two columns with the same
    name - so `loans["loan_amnt"]` returned a DataFrame and the run died two
    modules later inside a numeric coercion."""
    raw_path = tmp_path / "loans.csv"
    pd.DataFrame(
        {
            "funded_amnt": [111, 222],
            "loan_amnt": [1000, 2000],
            "status": ["Fully Paid", "Charged Off"],
            "state": ["CA", "NY"],
        }
    ).to_csv(raw_path, index=False)

    loans, report = read_raw_loans(raw_path)

    assert list(loans.columns).count("loan_amnt") == 1
    # The canonical spelling wins, not the one that happened to come first.
    assert loans["loan_amnt"].tolist() == [1000, 2000]
    assert report.dropped_duplicate_sources == {"funded_amnt": "loan_amnt"}
    assert report.renamed == {"status": "loan_status", "state": "addr_state"}


def test_p01_the_read_is_projected_to_registered_columns(
    raw_csv_factory: Callable[..., Path],
) -> None:
    """145 columns read as 27. The old unprojected read is what makes the
    1.8M-row file exhaust memory before a single feature exists."""
    raw_path = raw_csv_factory()
    all_columns = pd.read_csv(raw_path, nrows=0).columns

    loans, report = read_raw_loans(raw_path)

    assert len(loans.columns) < len(all_columns)
    assert "url" not in loans.columns
    # Projection happens at read time, so an unwanted column is never in memory
    # at all - hence it is not reported as merely "unknown".
    assert "url" not in report.unknown_columns


def test_reading_without_projection_reports_unknown_columns(tmp_path: Path) -> None:
    """An unfamiliar extract's extra columns should be listed, not silently
    absorbed, so onboarding it starts from a fact rather than a guess."""
    raw_path = tmp_path / "loans.csv"
    pd.DataFrame(
        {"loan_amnt": [1000], "loan_status": ["Fully Paid"], "lender_internal_score": [7]}
    ).to_csv(raw_path, index=False)

    _loans, report = read_raw_loans(raw_path, project=False)

    assert report.unknown_columns == ("lender_internal_score",)


def test_read_rejects_a_file_with_no_recognizable_columns(tmp_path: Path) -> None:
    raw_path = tmp_path / "wrong.csv"
    pd.DataFrame({"alpha": [1], "beta": [2]}).to_csv(raw_path, index=False)

    with pytest.raises(ValueError, match="matched a known column"):
        read_raw_loans(raw_path)


def test_read_rejects_an_unsupported_extension(tmp_path: Path) -> None:
    raw_path = tmp_path / "loans.xlsx"
    raw_path.write_bytes(b"not really a spreadsheet")

    with pytest.raises(ValueError, match="Unsupported raw data format"):
        read_raw_loans(raw_path)


def test_read_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_raw_loans(tmp_path / "absent.csv")


def test_closed_statuses_default_is_immutable() -> None:
    """A mutable default would let one caller change the filter for every other
    caller in the process."""
    assert isinstance(CLOSED_LOAN_STATUSES, frozenset)


def test_filter_to_closed_loans_names_the_missing_column() -> None:
    with pytest.raises(KeyError, match="loan_status"):
        filter_to_closed_loans(pd.DataFrame({"loan_amnt": [1]}))


def test_create_default_target_maps_terminal_statuses() -> None:
    loans = pd.DataFrame(
        {"loan_status": ["Fully Paid", "Charged Off", "Default", "Current", "Late (31-120 days)"]}
    )

    result = create_default_target(loans)

    # Current and Late are NA, not 0: calling a still-performing loan "did not
    # default" is the survivorship mistake the embargo exists to prevent.
    assert result.tolist()[:3] == [0, 1, 1]
    assert result.isna().tolist() == [False, False, False, True, True]


def test_p02_create_default_target_returns_a_series_not_a_frame_copy() -> None:
    loans = pd.DataFrame({"loan_status": ["Fully Paid"], "loan_amnt": [1000]})

    result = create_default_target(loans)

    assert isinstance(result, pd.Series)
    assert result.name == "default_flag"
    # The input is untouched, so the caller decides when to pay for a copy.
    assert "default_flag" not in loans.columns


def _embargo_frame() -> pd.DataFrame:
    """Two mature 36-month loans and two immature ones, against a 2018-12 snapshot."""
    return pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2014-01-01", "2014-06-01", "2016-06-01", "2017-01-01"]),
            "term": [" 36 months", " 36 months", " 36 months", " 36 months"],
            "loan_status": ["Fully Paid", "Charged Off", "Charged Off", "Charged Off"],
        }
    )


def test_embargo_keeps_only_loans_whose_term_has_elapsed() -> None:
    result = apply_outcome_maturity_embargo(_embargo_frame(), snapshot="2018-12-01")

    assert result.rows_in == 4
    assert result.rows_out == 2
    assert result.rows_immature == 2
    assert result.loans["issue_d"].dt.year.tolist() == [2014, 2014]


def test_embargo_removes_the_survivorship_bias_in_a_closed_loan_filter(
    raw_loans_factory: Callable[..., pd.DataFrame],
) -> None:
    """The load-bearing test for the whole exercise. The synthetic generator
    censors default timing against a snapshot for the same reason the real
    extract does, so filtering to closed loans inflates the late vintages. The
    embargo must flatten that, and must land near the generator's true rate."""
    true_rate = 0.15
    raw = raw_loans_factory(
        n_rows=12_000,
        start_month="2012-01",
        end_month="2016-12",
        snapshot="2018-12",
        target_default_rate=true_rate,
    )
    # The real order: canonicalize (which parses dates) before filtering rows.
    canonical, _report = normalize_credit_schema(raw)
    closed = filter_to_closed_loans(canonical)

    before = apply_outcome_maturity_embargo(closed, snapshot="2018-12-01")

    biased = before.default_rate_before
    corrected = before.default_rate_after
    # Before: the last vintage reads far worse than the first, purely because
    # only its early defaults have had time to close.
    assert biased.loc[2016] > biased.loc[2012] + 0.05
    # After: no vintage strays far from the truth, and the trend is gone.
    assert corrected.max() - corrected.min() < biased.max() - biased.min()
    assert abs(corrected.mean() - true_rate) < 0.04


def test_embargo_counts_rows_it_cannot_prove_mature() -> None:
    """A row whose term or date did not parse cannot be shown to be mature, so it
    is removed - but counted separately, because "corrupt" and "still running"
    are different problems with different fixes."""
    frame = _embargo_frame()
    frame.loc[1, "term"] = "unknown"
    frame.loc[2, "issue_d"] = pd.NaT

    result = apply_outcome_maturity_embargo(frame, snapshot="2018-12-01")

    assert result.rows_unknown_maturity == 2
    assert result.rows_immature == 1  # the 2017 loan is genuinely still running
    assert result.rows_out == 1
    assert result.rows_immature + result.rows_unknown_maturity + result.rows_out == 4


def test_embargo_refuses_an_unparsed_date_column() -> None:
    """Re-parsing here would mean two different date-format policies in one
    pipeline, which is how B28 got in."""
    frame = _embargo_frame().assign(issue_d=["Jan-2014", "Jun-2014", "Jun-2016", "Jan-2017"])

    with pytest.raises(TypeError, match="must already be datetime"):
        apply_outcome_maturity_embargo(frame, snapshot="2018-12-01")


def test_embargo_error_reports_the_observed_date_span() -> None:
    """The old split code crashed inside its own error handler on all-NaT dates.
    An error about dates has to name the dates."""
    with pytest.raises(ValueError, match="issue dates span 2014-01-01 to 2017-01-01"):
        apply_outcome_maturity_embargo(_embargo_frame(), snapshot="2005-01-01")


def test_embargo_rejects_a_timezone_aware_snapshot() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        apply_outcome_maturity_embargo(
            _embargo_frame(), snapshot=pd.Timestamp("2018-12-01", tz="UTC")
        )


def test_embargo_names_a_missing_column() -> None:
    with pytest.raises(KeyError, match="term"):
        apply_outcome_maturity_embargo(
            _embargo_frame().drop(columns=["term"]), snapshot="2018-12-01"
        )


def _term_frame() -> pd.DataFrame:
    """The two term spellings the real extracts use, plus one that does not parse."""
    return pd.DataFrame({"term": [" 36 months", "60 months", "36 MONTHS", "n/a"]})


def test_term_months_reads_both_extract_spellings() -> None:
    result = term_months(_term_frame())

    assert result.tolist()[:3] == [36, 60, 36]
    # Not zero, and not an exception: the callers count these rows separately.
    assert pd.isna(result.iloc[3])


def test_term_months_names_a_missing_column() -> None:
    with pytest.raises(KeyError, match="term"):
        term_months(pd.DataFrame({"loan_amnt": [1000]}))


def test_filter_to_terms_keeps_only_the_requested_terms() -> None:
    kept = filter_to_terms(_term_frame(), terms=(36,))

    assert kept.index.tolist() == [0, 2]


def test_filter_to_terms_drops_rows_whose_term_did_not_parse() -> None:
    """Same reasoning as the embargo: a loan that cannot be shown to be in scope
    is not in scope."""
    assert 3 not in filter_to_terms(_term_frame(), terms=(36, 60)).index


def test_filter_to_terms_with_no_terms_keeps_everything() -> None:
    """`term_months_in: []` in the config means "every term", and must not mean
    "no rows" - an empty allow-list read as a filter would empty the run."""
    frame = _term_frame()

    assert filter_to_terms(frame, terms=()) is frame
