"""Tests for the synthetic data generator.

The generator is test infrastructure, so it needs its own tests: if it silently
stops producing a learnable signal or stops reproducing the vintage bias, every
downstream assertion turns into a tautology.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from risk_score.sample_data import make_synthetic_loans

CLOSED_STATUSES = (
    "fully paid",
    "charged off",
    "does not meet the credit policy. status:fully paid",
    "does not meet the credit policy. status:charged off",
)


def _closed_with_target(loans: pd.DataFrame) -> pd.DataFrame:
    """Reduce to closed loans with a 0/1 default flag and an issue year."""
    status = loans["loan_status"].astype("string").str.strip().str.lower()
    closed = loans.loc[status.isin(CLOSED_STATUSES)].copy()
    closed["default_flag"] = (
        status.loc[closed.index].str.contains("charged off", regex=False).astype(int)
    )
    closed["vintage"] = pd.to_datetime(closed["issue_d"], format="%b-%Y").dt.year
    return closed


def test_generator_is_deterministic() -> None:
    """The same seed must produce byte-identical output."""
    first = make_synthetic_loans(n_rows=200, seed=7)
    second = make_synthetic_loans(n_rows=200, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_generator_varies_with_seed() -> None:
    """Different seeds must produce different data, or fixtures share one sample."""
    first = make_synthetic_loans(n_rows=200, seed=7)
    second = make_synthetic_loans(n_rows=200, seed=8)

    assert not first["loan_amnt"].equals(second["loan_amnt"])


def test_raw_columns_arrive_in_source_format() -> None:
    """Percent strings, spaced term text, and month-name dates, exactly as in the CSV."""
    loans = make_synthetic_loans(n_rows=50, percent_strings=True)

    assert loans["term"].iloc[0] in {" 36 months", " 60 months"}
    assert loans["int_rate"].iloc[0].endswith("%")
    assert loans["revol_util"].dropna().iloc[0].endswith("%")
    # e.g. 'Aug-2003' - not ISO, and not parseable without an explicit format.
    assert pd.to_datetime(loans["issue_d"].iloc[0], format="%b-%Y") is not pd.NaT
    assert (
        loans["emp_length"]
        .dropna()
        .isin(
            {
                "< 1 year",
                "1 year",
                "2 years",
                "3 years",
                "4 years",
                "5 years",
                "6 years",
                "7 years",
                "8 years",
                "9 years",
                "10+ years",
            }
        )
        .all()
    )


def test_percent_strings_can_be_disabled() -> None:
    """The `loans_full_schema` extract has float rates; both variants must be available."""
    loans = make_synthetic_loans(n_rows=50, percent_strings=False)

    assert pd.api.types.is_numeric_dtype(loans["int_rate"])
    assert pd.api.types.is_numeric_dtype(loans["revol_util"])


def test_fico_and_post_origination_columns_are_optional() -> None:
    """The real `data/raw/1/loan.csv` has no FICO columns, so absence must be supported."""
    loans = make_synthetic_loans(
        n_rows=50,
        include_fico=False,
        include_post_origination=False,
        include_junk_columns=False,
    )

    assert "fico_range_low" not in loans.columns
    assert "total_pymnt" not in loans.columns
    assert "url" not in loans.columns


def test_lifetime_default_rate_matches_the_requested_target() -> None:
    """The solved intercept must hit the target rate, not merely land nearby."""
    loans = make_synthetic_loans(n_rows=20_000, target_default_rate=0.22, seed=11)
    status = loans["loan_status"].astype("string").str.lower()
    # Every loan that will ever default reads as charged off, late, or in grace
    # at the snapshot; everything else either paid or is current and healthy.
    ever_defaults = status.str.contains("charged off|late|grace", regex=True)

    assert ever_defaults.mean() == pytest.approx(0.22, abs=0.02)


def test_signal_is_learnable_and_directionally_correct() -> None:
    """Utilization and FICO must move default rates the way credit risk actually does.

    A generator with no signal would make every model comparison downstream
    meaningless while still passing shape assertions.
    """
    closed = _closed_with_target(make_synthetic_loans(n_rows=8000, seed=3))
    utilization = closed["revol_util"].astype("string").str.rstrip("%").astype(float)
    high_util = closed.loc[utilization > 90.0, "default_flag"].mean()
    low_util = closed.loc[utilization < 30.0, "default_flag"].mean()
    high_fico = closed.loc[closed["fico_range_low"] > 720.0, "default_flag"].mean()
    low_fico = closed.loc[closed["fico_range_low"] < 680.0, "default_flag"].mean()

    assert high_util > low_util + 0.05
    assert low_fico > high_fico + 0.05


def test_closed_only_filtering_reproduces_vintage_survivorship_bias() -> None:
    """Later vintages must look riskier on closed loans than they truly are.

    This is the whole reason the outcome-maturity embargo exists. If this
    assertion ever fails, the embargo tests are proving nothing.
    """
    loans = make_synthetic_loans(n_rows=12_000, seed=5)
    closed = _closed_with_target(loans)
    by_vintage = closed.groupby("vintage")["default_flag"].mean()

    assert by_vintage.loc[2016] > by_vintage.loc[2013] + 0.10
    # And the bias is upward: the closed-loan rate overstates the truth.
    status = loans["loan_status"].astype("string").str.lower()
    true_rate = status.str.contains("charged off|late|grace", regex=True).mean()
    assert closed["default_flag"].mean() > true_rate


def test_installment_is_consistent_with_rate_amount_and_term() -> None:
    """`installment` must be the amortizing payment, not an independent draw."""
    loans = make_synthetic_loans(n_rows=100, percent_strings=False, seed=2)
    monthly_rate = loans["int_rate"] / 100.0 / 12.0
    term = loans["term"].str.extract(r"(\d+)")[0].astype(int)
    growth = (1.0 + monthly_rate) ** term
    expected = loans["loan_amnt"] * monthly_rate * growth / (growth - 1.0)

    pd.testing.assert_series_equal(
        loans["installment"], expected.round(2), check_names=False, atol=0.01
    )


def test_missingness_is_present_in_the_columns_that_have_it_in_reality() -> None:
    """Imputers and missing indicators need something to act on."""
    loans = make_synthetic_loans(n_rows=5000, seed=4)

    assert loans["emp_length"].isna().sum() > 0
    assert loans["revol_util"].isna().sum() > 0
    assert loans["loan_amnt"].isna().sum() == 0


def test_rejects_invalid_arguments() -> None:
    """Bad configuration should fail immediately, not produce a degenerate frame."""
    with pytest.raises(ValueError, match="at least 1"):
        make_synthetic_loans(n_rows=0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        make_synthetic_loans(n_rows=10, target_default_rate=1.0)
    with pytest.raises(ValueError, match="must not be after"):
        make_synthetic_loans(n_rows=10, start_month="2016-01", end_month="2015-01")


def test_raw_csv_fixture_round_trips(raw_csv_factory: Callable[..., Path]) -> None:
    """The CSV fixture must survive a write/read cycle with its raw formats intact."""
    path = raw_csv_factory(n_rows=40)
    reloaded = pd.read_csv(path)

    assert reloaded["term"].iloc[0].strip().endswith("months")
    assert len(reloaded) == 40
