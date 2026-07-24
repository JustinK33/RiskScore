"""Starter tests for feature engineering behavior.

These tests are marked as skipped until the feature functions are implemented.
Remove the skip marks as you fill in the TODOs.
"""

import pandas as pd
import pytest

from risk_score import feature_engineering as fe


@pytest.mark.skip(reason="Feature engineering scaffold is intentionally not implemented yet.")
def test_add_loan_to_income_ratio_creates_expected_column() -> None:
    """Loan-to-income ratio should divide loan amount by annual income."""
    loans = pd.DataFrame({"loan_amnt": [10_000.0], "annual_inc": [50_000.0]})

    result = fe.add_loan_to_income_ratio(loans)

    assert "loan_to_income_ratio" in result.columns
    assert result.loc[0, "loan_to_income_ratio"] == pytest.approx(0.2)


@pytest.mark.skip(reason="Feature engineering scaffold is intentionally not implemented yet.")
def test_add_fico_band_feature_creates_interpretable_band() -> None:
    """FICO bands should be derived from the low and high score range."""
    loans = pd.DataFrame({"fico_range_low": [700], "fico_range_high": [704]})

    result = fe.add_fico_band_feature(loans)

    assert "fico_band" in result.columns
    assert result.loc[0, "fico_band"] is not None
