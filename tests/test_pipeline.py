"""Smoke tests for the runnable MVP pipeline."""

from pathlib import Path

import pandas as pd

from risk_score.pipeline import run_baseline_pipeline


def test_run_baseline_pipeline_writes_metrics_and_calibration_outputs(tmp_path: Path) -> None:
    """The MVP pipeline should run end to end on Lending Club-shaped data."""
    raw_path = tmp_path / "loans.csv"
    output_dir = tmp_path / "reports"
    pd.DataFrame(
        {
            "loan_status": [
                "Fully Paid",
                "Charged Off",
                "Fully Paid",
                "Charged Off",
                "Fully Paid",
                "Charged Off",
                "Fully Paid",
                "Charged Off",
            ],
            "issue_d": [
                "2016-01-01",
                "2016-02-01",
                "2016-03-01",
                "2016-04-01",
                "2017-01-01",
                "2017-02-01",
                "2017-03-01",
                "2017-04-01",
            ],
            "loan_amnt": [1000, 4000, 1500, 5000, 1200, 4500, 1700, 5500],
            "annual_inc": [80_000, 35_000, 75_000, 30_000, 82_000, 38_000, 78_000, 32_000],
            "dti": [10, 30, 12, 35, 11, 32, 13, 37],
            "revol_util": ["20%", "80%", "25%", "85%", "22%", "82%", "27%", "87%"],
            "fico_range_low": [730, 660, 720, 650, 735, 665, 725, 655],
            "fico_range_high": [734, 664, 724, 654, 739, 669, 729, 659],
            "grade": ["A", "D", "A", "E", "A", "D", "B", "E"],
            "total_pymnt": [1000, 250, 1500, 300, 1200, 275, 1700, 325],
        }
    ).to_csv(raw_path, index=False)

    metrics = run_baseline_pipeline(
        raw_path,
        train_end_date="2016-12-31",
        test_start_date="2017-01-01",
        output_dir=output_dir,
    )

    assert 0 <= metrics.auc_roc <= 1
    assert (output_dir / "metrics" / "logistic_regression_metrics.json").exists()
    assert (output_dir / "metrics" / "logistic_regression_calibration.csv").exists()
    assert (output_dir / "figures" / "logistic_regression_calibration.png").exists()
    assert (output_dir / "models" / "logistic_regression.joblib").exists()


def test_run_baseline_pipeline_accepts_alternate_lending_club_schema(
    tmp_path: Path,
) -> None:
    """Pipeline should run on Lending Club-like CSVs with common alias names."""
    raw_path = tmp_path / "loans.csv"
    output_dir = tmp_path / "reports"
    pd.DataFrame(
        {
            "status": [
                "Fully Paid",
                "Charged Off",
                "Fully Paid",
                "Charged Off",
                "Fully Paid",
                "Charged Off",
                "Fully Paid",
                "Charged Off",
            ],
            "issue_month": [
                "Jan-2016",
                "Feb-2016",
                "Mar-2016",
                "Apr-2016",
                "Jan-2017",
                "Feb-2017",
                "Mar-2017",
                "Apr-2017",
            ],
            "loan_amount": [1000, 4000, 1500, 5000, 1200, 4500, 1700, 5500],
            "annual_income": [80_000, 35_000, 75_000, 30_000, 82_000, 38_000, 78_000, 32_000],
            "debt_to_income": [10, 30, 12, 35, 11, 32, 13, 37],
            "revolUtil": ["20%", "80%", "25%", "85%", "22%", "82%", "27%", "87%"],
            "grade": ["A", "D", "A", "E", "A", "D", "B", "E"],
            "url": ["https://example.test"] * 8,
        }
    ).to_csv(raw_path, index=False)

    metrics = run_baseline_pipeline(
        raw_path,
        train_end_date="2016-12-31",
        test_start_date="2017-01-01",
        output_dir=output_dir,
    )

    assert 0 <= metrics.auc_roc <= 1
    assert (output_dir / "metrics" / "logistic_regression_metrics.json").exists()
