"""Tests for Lending Club data loading helpers."""

from pathlib import Path

import pandas as pd

from risk_score.data_loading import create_default_target, load_lending_club_data


def test_load_lending_club_data_filters_to_closed_loans(tmp_path: Path) -> None:
    """Only terminal loan outcomes should survive the raw loading filter."""
    raw_path = tmp_path / "loans.csv"
    pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Current", "Charged Off"],
            "loan_amnt": [1000, 2000, 3000],
        }
    ).to_csv(raw_path, index=False)

    result = load_lending_club_data(raw_path)

    assert result["loan_status"].tolist() == ["Fully Paid", "Charged Off"]


def test_load_lending_club_data_normalizes_common_column_aliases(tmp_path: Path) -> None:
    """Kaggle-style Lending Club schemas should be normalized on load."""
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
    assert result["issue_d"].dt.strftime("%Y-%m-%d").tolist() == [
        "2018-01-01",
        "2018-03-01",
    ]


def test_create_default_target_maps_terminal_statuses() -> None:
    """Default target should map fully paid to 0 and charged off to 1."""
    loans = pd.DataFrame({"loan_status": ["Fully Paid", "Charged Off", "Default"]})

    result = create_default_target(loans)

    assert result["default_flag"].tolist() == [0, 1, 1]
