"""Tests for leakage controls."""

import pandas as pd

from risk_score.leakage_check import (
    exclude_leaky_columns,
    get_post_origination_columns,
    select_origination_time_columns,
)


def test_get_post_origination_columns_flags_known_leaky_fields() -> None:
    """Known servicing and outcome fields should be flagged as post-origination."""
    columns = ["loan_amnt", "loan_status", "total_pymnt", "annual_inc"]

    result = get_post_origination_columns(columns)

    assert result == ["loan_status", "total_pymnt"]


def test_exclude_leaky_columns_preserves_default_target() -> None:
    """Leakage exclusion should drop post-origination fields but keep target."""
    loans = pd.DataFrame(
        {
            "loan_status": ["Fully Paid"],
            "total_pymnt": [1200.0],
            "loan_amnt": [1000.0],
            "default_flag": [0],
        }
    )

    result = exclude_leaky_columns(loans)

    assert "loan_status" not in result.columns
    assert "total_pymnt" not in result.columns
    assert "default_flag" in result.columns
    assert "loan_amnt" in result.columns


def test_select_origination_time_columns_drops_unknown_raw_fields() -> None:
    """Only documented underwriting-time fields should reach the model."""
    loans = pd.DataFrame(
        {
            "default_flag": [0],
            "loan_amnt": [1000.0],
            "annual_inc": [50_000.0],
            "url": ["https://example.test/loan/1"],
            "desc": ["Borrower-written text."],
        }
    )

    result = select_origination_time_columns(loans)

    assert result.columns.tolist() == ["default_flag", "loan_amnt", "annual_inc"]
