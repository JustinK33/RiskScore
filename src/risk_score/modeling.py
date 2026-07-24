"""Model training stubs for credit default risk modeling."""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    """Container for time-based train and test partitions."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def time_based_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    date_column: str,
    train_end_date: str,
    test_start_date: str,
) -> TimeSplit:
    """Split observations by date so later vintages are held out for testing.

    TODO:
        Validate date parsing, prevent overlap between train and test windows,
        and preserve chronological ordering within each partition.
    """
    raise NotImplementedError("Create time-based train/test split.")


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: dict[str, Any] | None = None,
) -> Any:
    """Train the regularized logistic regression baseline.

    TODO:
        Build a scikit-learn pipeline with preprocessing, class weighting, and
        a calibrated probability output if needed.
    """
    raise NotImplementedError("Train logistic regression baseline.")


def train_xgboost_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: dict[str, Any] | None = None,
) -> Any:
    """Train the main XGBoost credit default model.

    TODO:
        Add parameter loading, validation monitoring, early stopping, and model
        artifact persistence.
    """
    raise NotImplementedError("Train XGBoost model.")
