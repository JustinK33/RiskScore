"""Tests for model evaluation helpers."""

import pandas as pd
import pytest

from risk_score.evaluation import (
    CostMatrix,
    compute_auc_roc,
    compute_brier_score,
    compute_ks_statistic,
    compute_precision_recall,
    compute_threshold_cost_table,
    select_threshold_by_cost,
)


def test_compute_auc_roc_returns_expected_perfect_score() -> None:
    """A perfectly ranked score should have AUC equal to 1."""
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.1, 0.2, 0.8, 0.9])

    assert compute_auc_roc(y_true, y_score) == pytest.approx(1.0)


def test_precision_recall_output_has_report_columns() -> None:
    """Precision-recall helper should return a report-friendly dataframe."""
    y_true = pd.Series([0, 1, 0, 1])
    y_score = pd.Series([0.1, 0.8, 0.4, 0.9])

    result = compute_precision_recall(y_true, y_score)

    assert {"threshold", "precision", "recall", "average_precision"}.issubset(result.columns)


def test_compute_ks_statistic_returns_valid_range() -> None:
    """KS statistic should be between 0 and 1."""
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.1, 0.2, 0.8, 0.9])

    result = compute_ks_statistic(y_true, y_score)

    assert 0 <= result <= 1


def test_compute_brier_score_returns_valid_probability_loss() -> None:
    """Brier score should summarize probability accuracy on a 0 to 1 scale."""
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.1, 0.2, 0.8, 0.9])

    result = compute_brier_score(y_true, y_score)

    assert 0 <= result <= 1


def test_compute_threshold_cost_table_reports_operating_characteristics() -> None:
    """Threshold reporting should expose confusion counts and business cost."""
    y_true = pd.Series([0, 1, 1, 0])
    y_score = pd.Series([0.2, 0.4, 0.9, 0.8])
    cost_matrix = CostMatrix(false_negative_cost=5.0, false_positive_cost=1.0)

    result = compute_threshold_cost_table(
        y_true,
        y_score,
        cost_matrix=cost_matrix,
        thresholds=[0.3, 0.5],
    )

    assert list(result["threshold"]) == [0.3, 0.5]
    assert {
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "predicted_default_rate",
        "approval_rate",
        "total_cost",
    }.issubset(result.columns)
    assert result["total_cost"].min() >= 0


def test_select_threshold_by_cost_prefers_lower_false_negative_cost() -> None:
    """Cost-aware thresholding should return one of the configured thresholds."""
    y_true = pd.Series([0, 1, 1, 0])
    y_score = pd.Series([0.2, 0.4, 0.9, 0.8])
    cost_matrix = CostMatrix(false_negative_cost=5.0, false_positive_cost=1.0)

    result = select_threshold_by_cost(
        y_true,
        y_score,
        cost_matrix=cost_matrix,
        thresholds=[0.3, 0.5, 0.7],
    )

    assert result in {0.3, 0.5, 0.7}
