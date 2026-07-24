"""Leakage controls for Lending Club credit risk modeling.

This module documents which fields are available at origination time and which
fields are only known after origination.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ColumnLeakageMetadata:
    """Metadata describing whether a column is safe for origination-time modeling."""

    column: str
    available_at_origination: bool
    reason: str


ORIGINATION_TIME_COLUMNS: tuple[str, ...] = (
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
)


POST_ORIGINATION_COLUMNS: dict[str, str] = {
    "loan_status": "Terminal outcome used to construct the target.",
    "pymnt_plan": "Can reflect servicing behavior after origination.",
    "out_prncp": "Outstanding principal is only known after payments occur.",
    "out_prncp_inv": "Outstanding investor principal is post-origination.",
    "total_pymnt": "Total payments are accumulated after origination.",
    "total_pymnt_inv": "Investor payments are accumulated after origination.",
    "total_rec_prncp": "Principal received is post-origination payment history.",
    "total_rec_int": "Interest received is post-origination payment history.",
    "total_rec_late_fee": "Late fees reveal repayment behavior.",
    "recoveries": "Recoveries occur after default or charge-off.",
    "collection_recovery_fee": "Collection fees occur after delinquency or default.",
    "last_pymnt_d": "Last payment date is post-origination.",
    "last_pymnt_amnt": "Last payment amount is post-origination.",
    "next_pymnt_d": "Next payment date is servicing information.",
    "last_credit_pull_d": "Credit pulls after origination may leak future state.",
    "last_fico_range_high": "Updated FICO score is measured after origination.",
    "last_fico_range_low": "Updated FICO score is measured after origination.",
    "collections_12_mths_ex_med": "May be reported after origination depending on extract timing.",
    "policy_code": "May encode Lending Club policy changes rather than borrower risk.",
}


def build_leakage_metadata() -> list[ColumnLeakageMetadata]:
    """Return documented leakage metadata for known Lending Club columns.

    TODO:
        Expand this list after inspecting the exact dataset vintage being used.
    """
    metadata = [
        ColumnLeakageMetadata(
            column=column,
            available_at_origination=True,
            reason="Borrower or loan attribute available at underwriting time.",
        )
        for column in ORIGINATION_TIME_COLUMNS
    ]
    metadata.extend(
        ColumnLeakageMetadata(
            column=column,
            available_at_origination=False,
            reason=reason,
        )
        for column, reason in POST_ORIGINATION_COLUMNS.items()
    )
    return metadata


def get_post_origination_columns(columns: list[str] | pd.Index) -> list[str]:
    """Return columns that should be excluded because they are post-origination.

    TODO:
        Add optional warnings for unknown columns and dataset-specific overrides
        from `configs/feature_config.yaml`.
    """
    known_leaky = set(POST_ORIGINATION_COLUMNS)
    return [column for column in columns if column in known_leaky]


def exclude_leaky_columns(loans: pd.DataFrame) -> pd.DataFrame:
    """Drop known post-origination leakage columns from a loan dataframe.

    TODO:
        Preserve target columns only when explicitly requested by the modeling
        pipeline and log every excluded field for reproducibility.
    """
    raise NotImplementedError("Exclude post-origination leakage columns.")
