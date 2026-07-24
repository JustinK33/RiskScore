"""Data loading utilities for Lending Club credit risk modeling.

The functions in this module should keep raw ingestion separate from feature
engineering and modeling.
"""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

CLOSED_LOAN_STATUSES: set[str] = {
    "Fully Paid",
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
}


def load_lending_club_data(
    path: str | Path,
    *,
    closed_statuses: Iterable[str] = CLOSED_LOAN_STATUSES,
) -> pd.DataFrame:
    """Load Lending Club loan data and filter to closed loans only.

    Parameters
    ----------
    path:
        Path to a raw Lending Club CSV or parquet file.
    closed_statuses:
        Loan statuses that represent terminal outcomes suitable for supervised
        default modeling.

    Returns
    -------
    pandas.DataFrame
        Raw loan records filtered to closed loan outcomes.

    TODO:
        Add richer schema validation, dtype normalization, and structured
        logging once the exact Lending Club extract is selected.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file does not exist: {data_path}")

    if data_path.suffix.lower() == ".csv":
        loans = pd.read_csv(data_path, low_memory=False)
    elif data_path.suffix.lower() in {".parquet", ".pq"}:
        loans = pd.read_parquet(data_path)
    else:
        raise ValueError("Supported raw data formats are CSV and parquet.")

    if "loan_status" not in loans.columns:
        raise KeyError("Expected Lending Club column `loan_status` is missing.")

    normalized_statuses = {status.strip().lower() for status in closed_statuses}
    status = loans["loan_status"].astype("string").str.strip().str.lower()
    return loans.loc[status.isin(normalized_statuses)].copy()


def create_default_target(
    loans: pd.DataFrame,
    *,
    status_column: str = "loan_status",
    target_column: str = "default_flag",
) -> pd.DataFrame:
    """Create a binary default target from terminal loan statuses.

    Ambiguous or non-terminal statuses are left as missing so callers can decide
    whether to filter or audit them.

    TODO:
        Expand the mapping if a specific Lending Club vintage contains
        additional terminal statuses.
    """
    if status_column not in loans.columns:
        raise KeyError(f"Expected status column `{status_column}` is missing.")

    default_statuses = {
        "charged off",
        "default",
        "does not meet the credit policy. status:charged off",
    }
    paid_statuses = {
        "fully paid",
        "does not meet the credit policy. status:fully paid",
    }

    result = loans.copy()
    normalized_status = result[status_column].astype("string").str.strip().str.lower()
    result[target_column] = pd.NA
    result.loc[normalized_status.isin(default_statuses), target_column] = 1
    result.loc[normalized_status.isin(paid_statuses), target_column] = 0
    result[target_column] = result[target_column].astype("Int64")
    return result
