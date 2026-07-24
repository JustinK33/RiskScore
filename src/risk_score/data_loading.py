"""Data loading utilities for Lending Club credit risk modeling.

The functions in this module should keep raw ingestion separate from feature
engineering and modeling.
"""

from pathlib import Path
from typing import Iterable

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
        Add robust file type handling, schema validation, dtype normalization,
        and logging.
    """
    raise NotImplementedError("Load raw Lending Club data and filter to closed loans.")


def create_default_target(
    loans: pd.DataFrame,
    *,
    status_column: str = "loan_status",
    target_column: str = "default_flag",
) -> pd.DataFrame:
    """Create a binary default target from terminal loan statuses.

    TODO:
        Map charged-off and default-like statuses to 1, fully paid statuses to 0,
        and document any ambiguous statuses that are excluded.
    """
    raise NotImplementedError("Create default target from terminal loan status.")
