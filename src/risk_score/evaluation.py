"""Evaluation stubs for credit default risk models."""

from dataclasses import dataclass

import pandas as pd


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
    raise NotImplementedError("Compute AUC-ROC.")


def compute_precision_recall(y_true: pd.Series, y_score: pd.Series) -> pd.DataFrame:
    """Compute precision-recall curve values.

    TODO:
        Return thresholds, precision, recall, and average precision in a
        report-friendly format.
    """
    raise NotImplementedError("Compute precision-recall curve.")


def compute_ks_statistic(y_true: pd.Series, y_score: pd.Series) -> float:
    """Compute the Kolmogorov-Smirnov statistic for default score separation.

    TODO:
        Compare cumulative score distributions for defaulted and non-defaulted
        loans.
    """
    raise NotImplementedError("Compute KS statistic.")


def select_threshold_by_cost(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    cost_matrix: CostMatrix,
    thresholds: list[float] | None = None,
) -> float:
    """Select a decision threshold using false negative and false positive costs.

    TODO:
        Evaluate total expected cost across thresholds and return the threshold
        with the lowest cost.
    """
    raise NotImplementedError("Select cost-minimizing decision threshold.")
