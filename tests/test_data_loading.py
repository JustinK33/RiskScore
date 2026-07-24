"""Tests for Lending Club data loading helpers."""

import pandas as pd

from risk_score.data_loading import create_default_target, load_lending_club_data


def test_load_lending_club_data_filters_to_closed_loans(tmp_path) -> None:
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


def test_create_default_target_maps_terminal_statuses() -> None:
    """Default target should map fully paid to 0 and charged off to 1."""
    loans = pd.DataFrame({"loan_status": ["Fully Paid", "Charged Off", "Default"]})

    result = create_default_target(loans)

    assert result["default_flag"].tolist() == [0, 1, 1]
