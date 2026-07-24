"""Model training stubs for credit default risk modeling."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
        Add optional validation windows for calibration and hyperparameter
        tuning without contaminating the final test set.
    """
    if date_column not in features.columns:
        raise KeyError(f"Date column `{date_column}` is missing from features.")
    if len(features) != len(target):
        raise ValueError("Features and target must have the same number of rows.")

    working = features.copy()
    working["_split_date"] = pd.to_datetime(working[date_column], errors="coerce")
    if working["_split_date"].isna().any():
        raise ValueError(f"Date column `{date_column}` contains invalid or missing dates.")

    train_end = pd.Timestamp(train_end_date)
    test_start = pd.Timestamp(test_start_date)
    if train_end >= test_start:
        raise ValueError("`train_end_date` must be earlier than `test_start_date`.")

    train_mask = working["_split_date"] <= train_end
    test_mask = working["_split_date"] >= test_start

    x_train = working.loc[train_mask].sort_values("_split_date").drop(
        columns=["_split_date", date_column]
    )
    x_test = working.loc[test_mask].sort_values("_split_date").drop(
        columns=["_split_date", date_column]
    )
    y_train = target.loc[x_train.index]
    y_test = target.loc[x_test.index]

    if x_train.empty or x_test.empty:
        observed_min = working["_split_date"].min().date().isoformat()
        observed_max = working["_split_date"].max().date().isoformat()
        raise ValueError(
            "Time-based split produced an empty train or test set. "
            f"Available date range is {observed_min} to {observed_max}. "
            f"Train rows: {len(x_train)}. Test rows: {len(x_test)}."
        )

    return TimeSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing transformer for mixed numeric and categorical data."""
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in set(numeric_columns)
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns were provided.")

    return ColumnTransformer(transformers=transformers)


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: dict[str, Any] | None = None,
) -> Any:
    """Train the regularized logistic regression baseline.

    TODO:
        Add experiment tracking and model artifact persistence.
    """
    params = {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    }
    if config:
        params.update(config)

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("classifier", LogisticRegression(**params)),
        ]
    )
    return model.fit(x_train, y_train)


def train_xgboost_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: dict[str, Any] | None = None,
) -> Any:
    """Train the main XGBoost credit default model.

    TODO:
        Add validation monitoring, early stopping, and artifact persistence.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install xgboost to train the main model.") from exc

    params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": 42,
    }
    if config:
        params.update(config)

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("classifier", XGBClassifier(**params)),
        ]
    )
    return model.fit(x_train, y_train)
