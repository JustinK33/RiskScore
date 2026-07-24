"""Feature engineering utilities for credit default risk modeling."""

import numpy as np
import pandas as pd


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce numeric Lending Club fields, including percentage strings."""
    cleaned = series.astype("string").str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def add_dti_feature(loans: pd.DataFrame) -> pd.DataFrame:
    """Add a cleaned debt-to-income feature.

    TODO:
        Decide whether to cap or winsorize extreme values after EDA.
    """
    if "dti" not in loans.columns:
        raise KeyError("Expected column `dti` is missing.")

    result = loans.copy()
    result["dti_clean"] = _coerce_numeric(result["dti"])
    return result


def add_credit_utilization_feature(loans: pd.DataFrame) -> pd.DataFrame:
    """Add a credit utilization feature from revolving utilization inputs.

    TODO:
        Decide whether missing utilization should be imputed, flagged, or both.
    """
    if "revol_util" not in loans.columns:
        raise KeyError("Expected column `revol_util` is missing.")

    result = loans.copy()
    utilization = _coerce_numeric(result["revol_util"])
    result["credit_utilization"] = utilization / 100.0
    return result


def add_fico_band_feature(loans: pd.DataFrame) -> pd.DataFrame:
    """Add a FICO band feature from FICO range columns.

    TODO:
        Tune bin definitions for the exact project narrative and reporting.
    """
    required_columns = {"fico_range_low", "fico_range_high"}
    missing = required_columns.difference(loans.columns)
    if missing:
        raise KeyError(f"Missing required FICO columns: {sorted(missing)}")

    result = loans.copy()
    fico_midpoint = (
        _coerce_numeric(result["fico_range_low"]) + _coerce_numeric(result["fico_range_high"])
    ) / 2.0
    result["fico_midpoint"] = fico_midpoint
    result["fico_band"] = pd.cut(
        fico_midpoint,
        bins=[0, 580, 670, 740, 800, np.inf],
        labels=["poor", "fair", "good", "very_good", "exceptional"],
        right=False,
    )
    return result


def add_loan_to_income_ratio(loans: pd.DataFrame) -> pd.DataFrame:
    """Add loan amount divided by annual income.

    TODO:
        Add an explicit missing income indicator if EDA shows it is predictive.
    """
    required_columns = {"loan_amnt", "annual_inc"}
    missing = required_columns.difference(loans.columns)
    if missing:
        raise KeyError(f"Missing required loan-to-income columns: {sorted(missing)}")

    result = loans.copy()
    loan_amount = _coerce_numeric(result["loan_amnt"])
    annual_income = _coerce_numeric(result["annual_inc"])
    result["loan_to_income_ratio"] = loan_amount.div(annual_income.replace(0, np.nan))
    return result


def build_feature_matrix(loans: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return model-ready features.

    TODO:
        Add config-driven feature selection from `configs/feature_config.yaml`.
    """
    result = loans.copy()
    result = add_dti_feature(result)
    result = add_credit_utilization_feature(result)
    fico_columns = {"fico_range_low", "fico_range_high"}
    if fico_columns.issubset(result.columns):
        result = add_fico_band_feature(result)
    elif fico_columns.intersection(result.columns):
        missing = fico_columns.difference(result.columns)
        raise KeyError(f"Missing required FICO columns: {sorted(missing)}")
    result = add_loan_to_income_ratio(result)
    return result
