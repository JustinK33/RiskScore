"""Tests for feature engineering behavior."""

import pandas as pd
import pytest

from risk_score import feature_engineering as fe


def test_add_loan_to_income_ratio_creates_expected_column() -> None:
    """Loan-to-income ratio should divide loan amount by annual income."""
    loans = pd.DataFrame({"loan_amnt": [10_000.0], "annual_inc": [50_000.0]})

    result = fe.add_loan_to_income_ratio(loans)

    assert "loan_to_income_ratio" in result.columns
    assert result.loc[0, "loan_to_income_ratio"] == pytest.approx(0.2)


def test_add_fico_band_feature_creates_interpretable_band() -> None:
    """FICO bands should be derived from the low and high score range."""
    loans = pd.DataFrame({"fico_range_low": [700], "fico_range_high": [704]})

    result = fe.add_fico_band_feature(loans)

    assert "fico_band" in result.columns
    assert result.loc[0, "fico_band"] == "good"


def test_add_credit_utilization_parses_percentage_strings() -> None:
    """Revolving utilization should be converted from percent to fraction."""
    loans = pd.DataFrame({"revol_util": ["47.5%"]})

    result = fe.add_credit_utilization_feature(loans)

    assert result.loc[0, "credit_utilization"] == pytest.approx(0.475)


def test_add_credit_utilization_uses_total_credit_ratio_when_revol_util_is_missing() -> None:
    """Utilization should work for bureau-style total balance and limit fields."""
    loans = pd.DataFrame({"total_credit_utilized": [2_500.0], "total_credit_limit": [10_000.0]})

    result = fe.add_credit_utilization_feature(loans)

    assert result.loc[0, "credit_utilization"] == pytest.approx(0.25)


def test_build_feature_matrix_adds_all_core_features() -> None:
    """Feature matrix builder should add all MVP engineered features."""
    loans = pd.DataFrame(
        {
            "dti": [18.2],
            "revol_util": ["42.0%"],
            "fico_range_low": [690],
            "fico_range_high": [694],
            "loan_amnt": [12_000.0],
            "annual_inc": [60_000.0],
        }
    )

    result = fe.build_feature_matrix(loans)

    assert {"dti_clean", "credit_utilization", "fico_band", "loan_to_income_ratio"}.issubset(
        result.columns
    )


def test_build_feature_matrix_allows_missing_fico_columns() -> None:
    """Datasets without origination FICO ranges should still be usable."""
    loans = pd.DataFrame(
        {
            "dti": [18.2],
            "revol_util": ["42.0%"],
            "loan_amnt": [12_000.0],
            "annual_inc": [60_000.0],
        }
    )

    result = fe.build_feature_matrix(loans)

    assert {"dti_clean", "credit_utilization", "loan_to_income_ratio"}.issubset(
        result.columns
    )
    assert "fico_band" not in result.columns
