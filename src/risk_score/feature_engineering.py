"""Feature engineering stubs for credit default risk modeling."""

import pandas as pd


def add_dti_feature(loans: pd.DataFrame) -> pd.DataFrame:
    """Add a cleaned debt-to-income feature.

    TODO:
        Normalize Lending Club DTI values, handle missing and extreme values,
        and decide whether to cap or winsorize outliers.
    """
    raise NotImplementedError("Engineer cleaned DTI feature.")


def add_credit_utilization_feature(loans: pd.DataFrame) -> pd.DataFrame:
    """Add a credit utilization feature from revolving utilization inputs.

    TODO:
        Parse percentage strings if needed, handle missing utilization, and
        document whether the resulting feature is a fraction or percentage.
    """
    raise NotImplementedError("Engineer credit utilization feature.")


def add_fico_band_feature(loans: pd.DataFrame) -> pd.DataFrame:
    """Add a FICO band feature from FICO range columns.

    TODO:
        Compute midpoint scores from `fico_range_low` and `fico_range_high`,
        then bin into interpretable credit score bands.
    """
    raise NotImplementedError("Engineer FICO band feature.")


def add_loan_to_income_ratio(loans: pd.DataFrame) -> pd.DataFrame:
    """Add loan amount divided by annual income.

    TODO:
        Handle zero or missing income safely and decide how to encode undefined
        ratios.
    """
    raise NotImplementedError("Engineer loan-to-income ratio feature.")


def build_feature_matrix(loans: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return model-ready features.

    TODO:
        Chain feature functions, select configured columns, and separate the
        target from predictors.
    """
    raise NotImplementedError("Build model-ready feature matrix.")
