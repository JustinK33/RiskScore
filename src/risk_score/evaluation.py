"""Evaluation stubs for credit default risk models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Core binary classification metrics for credit default risk models."""

    auc_roc: float
    average_precision: float
    ks_statistic: float
    brier_score: float
    default_rate: float
    approval_rate: float


@dataclass(frozen=True)
class CostMatrix:
    """Business costs used for threshold selection."""

    false_negative_cost: float
    false_positive_cost: float


def compute_auc_roc(y_true: pd.Series, y_score: pd.Series) -> float:
    """Compute AUC-ROC from true labels and predicted default probabilities."""
    return float(roc_auc_score(y_true, y_score))


def compute_brier_score(y_true: pd.Series, y_score: pd.Series) -> float:
    """Compute Brier score for probability calibration quality."""
    return float(brier_score_loss(y_true, y_score))


def compute_precision_recall(y_true: pd.Series, y_score: pd.Series) -> pd.DataFrame:
    """Compute precision-recall curve values.

    TODO:
        Add plotting helpers and persistence to `reports/metrics/`.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    threshold_values = np.append(thresholds, np.nan)
    return pd.DataFrame(
        {
            "threshold": threshold_values,
            "precision": precision,
            "recall": recall,
            "average_precision": average_precision_score(y_true, y_score),
        }
    )


def compute_ks_statistic(y_true: pd.Series, y_score: pd.Series) -> float:
    """Compute the Kolmogorov-Smirnov statistic for default score separation.

    TODO:
        Add confidence intervals through bootstrapping for reporting.
    """
    frame = pd.DataFrame({"y_true": y_true, "y_score": y_score}).sort_values(
        "y_score", ascending=False
    )
    positives = frame["y_true"].sum()
    negatives = len(frame) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("KS statistic requires both positive and negative classes.")

    cumulative_positive_rate = frame["y_true"].cumsum() / positives
    cumulative_negative_rate = (1 - frame["y_true"]).cumsum() / negatives
    return float((cumulative_positive_rate - cumulative_negative_rate).abs().max())


def compute_threshold_cost_table(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    cost_matrix: CostMatrix,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Return operating characteristics and business cost for candidate thresholds."""
    if thresholds is None:
        thresholds = [round(value, 2) for value in np.arange(0.01, 1.0, 0.01)]
    if not thresholds:
        raise ValueError("At least one threshold is required.")

    y_true_array = pd.Series(y_true).astype(int).to_numpy()
    y_score_array = pd.Series(y_score).astype(float).to_numpy()
    rows = []
    for threshold in thresholds:
        predictions = (y_score_array >= threshold).astype(int)
        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
            y_true_array,
            predictions,
            labels=[0, 1],
        ).ravel()
        predicted_defaults = true_positives + false_positives
        predicted_approvals = true_negatives + false_negatives
        total_cost = (
            false_negatives * cost_matrix.false_negative_cost
            + false_positives * cost_matrix.false_positive_cost
        )
        rows.append(
            {
                "threshold": float(threshold),
                "true_positives": int(true_positives),
                "false_positives": int(false_positives),
                "true_negatives": int(true_negatives),
                "false_negatives": int(false_negatives),
                "predicted_default_rate": float(predicted_defaults / len(y_true_array)),
                "approval_rate": float(predicted_approvals / len(y_true_array)),
                "total_cost": float(total_cost),
            }
        )

    return pd.DataFrame(rows)


def select_threshold_by_cost(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    cost_matrix: CostMatrix,
    thresholds: list[float] | None = None,
) -> float:
    """Select a decision threshold using false negative and false positive costs."""
    cost_table = compute_threshold_cost_table(
        y_true,
        y_score,
        cost_matrix=cost_matrix,
        thresholds=thresholds,
    )
    best_row = cost_table.sort_values(["total_cost", "threshold"], ascending=[True, True]).iloc[0]
    return float(best_row["threshold"])
