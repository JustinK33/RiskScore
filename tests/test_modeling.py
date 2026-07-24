"""Tests for train/test splitting and baseline modeling."""

import pandas as pd

from risk_score.modeling import time_based_train_test_split, train_logistic_regression


def test_time_based_split_uses_chronological_holdout() -> None:
    """Rows before the cutoff should train and rows after it should test."""
    features = pd.DataFrame(
        {
            "issue_d": ["2016-01-01", "2016-06-01", "2017-01-01", "2017-06-01"],
            "loan_amnt": [1000, 2000, 3000, 4000],
        }
    )
    target = pd.Series([0, 1, 0, 1])

    split = time_based_train_test_split(
        features,
        target,
        date_column="issue_d",
        train_end_date="2016-12-31",
        test_start_date="2017-01-01",
    )

    assert split.x_train["loan_amnt"].tolist() == [1000, 2000]
    assert split.x_test["loan_amnt"].tolist() == [3000, 4000]
    assert "issue_d" not in split.x_train.columns


def test_train_logistic_regression_returns_probability_model() -> None:
    """The baseline pipeline should fit and produce default probabilities."""
    x_train = pd.DataFrame(
        {
            "loan_amnt": [1000, 2000, 3000, 4000, 5000, 6000],
            "annual_inc": [80_000, 70_000, 60_000, 50_000, 40_000, 30_000],
            "grade": ["A", "A", "B", "C", "D", "E"],
        }
    )
    y_train = pd.Series([0, 0, 0, 1, 1, 1])

    model = train_logistic_regression(x_train, y_train)
    probabilities = model.predict_proba(x_train)[:, 1]

    assert probabilities.shape == (6,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
