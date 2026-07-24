"""Runnable MVP pipeline for the credit default risk project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from risk_score.calibration import compute_calibration_curve, plot_calibration_curve
from risk_score.data_loading import create_default_target, load_lending_club_data
from risk_score.evaluation import (
    ClassificationMetrics,
    CostMatrix,
    compute_auc_roc,
    compute_brier_score,
    compute_ks_statistic,
    compute_precision_recall,
    compute_threshold_cost_table,
    select_threshold_by_cost,
)
from risk_score.feature_engineering import build_feature_matrix
from risk_score.leakage_check import exclude_leaky_columns, select_origination_time_columns
from risk_score.modeling import (
    time_based_train_test_split,
    train_logistic_regression,
    train_xgboost_model,
)


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
    model_type: str = "logistic_regression",
    model_config: dict[str, Any] | None = None,
    schema_config: dict[str, Any] | None = None,
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
    models_path = output_path / "models"
    metrics_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    if cost_matrix is None:
        cost_matrix = CostMatrix(false_negative_cost=5.0, false_positive_cost=1.0)

    schema_config = schema_config or {}
    loans = load_lending_club_data(
        raw_data_path,
        column_aliases=schema_config.get("column_aliases"),
        date_column=date_column,
    )
    loans = create_default_target(loans)
    loans = loans.dropna(subset=["default_flag"])
    loans = exclude_leaky_columns(loans)
    loans = select_origination_time_columns(loans)
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
    if model_type == "logistic_regression":
        model = train_logistic_regression(split.x_train, split.y_train, config=model_config)
    elif model_type == "xgboost":
        model = train_xgboost_model(split.x_train, split.y_train, config=model_config)
    else:
        raise ValueError("Supported model types are `logistic_regression` and `xgboost`.")

    scores = _predict_default_probability(model, split.x_test)
    joblib.dump(model, models_path / f"{model_type}.joblib")

    threshold = select_threshold_by_cost(split.y_test, scores, cost_matrix=cost_matrix)
    threshold_cost_table = compute_threshold_cost_table(
        split.y_test,
        scores,
        cost_matrix=cost_matrix,
    )
    selected_threshold_row = threshold_cost_table.loc[
        threshold_cost_table["threshold"].eq(threshold)
    ].iloc[0]
    metrics = ClassificationMetrics(
        auc_roc=compute_auc_roc(split.y_test, scores),
        average_precision=float(
            compute_precision_recall(split.y_test, scores)["average_precision"].iloc[0]
        ),
        ks_statistic=compute_ks_statistic(split.y_test, scores),
        brier_score=compute_brier_score(split.y_test, scores),
        default_rate=float(split.y_test.mean()),
        approval_rate=float(selected_threshold_row["approval_rate"]),
    )

    metrics_payload = {
        "auc_roc": metrics.auc_roc,
        "average_precision": metrics.average_precision,
        "ks_statistic": metrics.ks_statistic,
        "brier_score": metrics.brier_score,
        "default_rate": metrics.default_rate,
        "approval_rate": metrics.approval_rate,
        "selected_threshold": threshold,
        "selected_threshold_total_cost": float(selected_threshold_row["total_cost"]),
        "false_negative_cost": cost_matrix.false_negative_cost,
        "false_positive_cost": cost_matrix.false_positive_cost,
        "model_type": model_type,
    }
    (metrics_path / f"{model_type}_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    calibration_data = compute_calibration_curve(split.y_test, scores)
    calibration_data.to_csv(metrics_path / f"{model_type}_calibration.csv", index=False)
    threshold_cost_table.to_csv(metrics_path / f"{model_type}_threshold_costs.csv", index=False)
    plot_calibration_curve(
        calibration_data,
        output_path=str(figures_path / f"{model_type}_calibration.png"),
    )

    return metrics
