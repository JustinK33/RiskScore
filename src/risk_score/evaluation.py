"""Evaluation stubs for credit default risk models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Core binary classification metrics for credit default risk models."""

    auc_roc: float
    average_precision: float
    ks_statistic: float


@dataclass(frozen=True)
class CostMatrix:
    """Business costs used for threshold selection."""

    false_negative_cost: float
    false_positive_cost: float


def compute_auc_roc(y_true: pd.Series, y_score: pd.Series) -> float:
    """Compute AUC-ROC from true labels and predicted default probabilities."""
    return float(roc_auc_score(y_true, y_score))


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


def select_threshold_by_cost(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    cost_matrix: CostMatrix,
    thresholds: list[float] | None = None,
) -> float:
    """Select a decision threshold using false negative and false positive costs.

    TODO:
        Return the full threshold cost table for reporting.
    """
    if thresholds is None:
        thresholds = [round(value, 2) for value in np.arange(0.01, 1.0, 0.01)]

    y_true_array = pd.Series(y_true).astype(int).to_numpy()
    y_score_array = pd.Series(y_score).astype(float).to_numpy()

    best_threshold = thresholds[0]
    best_cost = float("inf")
    for threshold in thresholds:
        predictions = (y_score_array >= threshold).astype(int)
        false_negatives = int(((y_true_array == 1) & (predictions == 0)).sum())
        false_positives = int(((y_true_array == 0) & (predictions == 1)).sum())
        total_cost = (
            false_negatives * cost_matrix.false_negative_cost
            + false_positives * cost_matrix.false_positive_cost
        )
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold

    return float(best_threshold)
