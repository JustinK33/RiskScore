"""Runnable MVP pipeline for the credit default risk project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from risk_score.calibration import compute_calibration_curve, plot_calibration_curve
from risk_score.data_loading import create_default_target, load_lending_club_data
from risk_score.evaluation import (
    ClassificationMetrics,
    CostMatrix,
    compute_auc_roc,
    compute_ks_statistic,
    compute_precision_recall,
    select_threshold_by_cost,
)
from risk_score.feature_engineering import build_feature_matrix
from risk_score.leakage_check import exclude_leaky_columns
from risk_score.modeling import time_based_train_test_split, train_logistic_regression


def _predict_default_probability(model: Any, features: pd.DataFrame) -> pd.Series:
    """Return positive-class default probabilities from a fitted classifier."""
    probabilities = model.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=features.index, name="default_probability")


def run_baseline_pipeline(
    raw_data_path: str | Path,
    *,
    train_end_date: str,
    test_start_date: str,
    date_column: str = "issue_d",
    output_dir: str | Path = "reports",
    cost_matrix: CostMatrix | None = None,
) -> ClassificationMetrics:
    """Run the minimal end-to-end logistic regression baseline pipeline.

    This MVP intentionally favors clarity over configurability.
    The next step is to move column selection, date cutoffs, and estimator
    settings fully into YAML configs.
    """
    output_path = Path(output_dir)
    metrics_path = output_path / "metrics"
    figures_path = output_path / "figures"
    metrics_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    if cost_matrix is None:
        cost_matrix = CostMatrix(false_negative_cost=5.0, false_positive_cost=1.0)

    loans = load_lending_club_data(raw_data_path)
    loans = create_default_target(loans)
    loans = loans.dropna(subset=["default_flag"])
    loans = exclude_leaky_columns(loans)
    loans = build_feature_matrix(loans)

    target = loans["default_flag"].astype(int)
    features = loans.drop(columns=["default_flag"])

    split = time_based_train_test_split(
        features,
        target,
        date_column=date_column,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
    )
    model = train_logistic_regression(split.x_train, split.y_train)
    scores = _predict_default_probability(model, split.x_test)

    metrics = ClassificationMetrics(
        auc_roc=compute_auc_roc(split.y_test, scores),
        average_precision=float(
            compute_precision_recall(split.y_test, scores)["average_precision"].iloc[0]
        ),
        ks_statistic=compute_ks_statistic(split.y_test, scores),
    )
    threshold = select_threshold_by_cost(split.y_test, scores, cost_matrix=cost_matrix)

    metrics_payload = {
        "auc_roc": metrics.auc_roc,
        "average_precision": metrics.average_precision,
        "ks_statistic": metrics.ks_statistic,
        "selected_threshold": threshold,
        "false_negative_cost": cost_matrix.false_negative_cost,
        "false_positive_cost": cost_matrix.false_positive_cost,
    }
    (metrics_path / "baseline_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    calibration_data = compute_calibration_curve(split.y_test, scores)
    calibration_data.to_csv(metrics_path / "baseline_calibration.csv", index=False)
    plot_calibration_curve(
        calibration_data,
        output_path=str(figures_path / "baseline_calibration.png"),
    )

    return metrics
