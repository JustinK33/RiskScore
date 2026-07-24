"""Schema normalization for Lending Club-like credit risk datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

DEFAULT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "loan_status": ("loan_status", "loanStatus", "status", "n"),
    "issue_d": ("issue_d", "issue_month", "issueDate", "issueMonth", "issue_d_month"),
    "loan_amnt": ("loan_amnt", "loan_amount", "loanAmnt", "funded_amnt", "fundedAmnt"),
    "annual_inc": ("annual_inc", "annual_income", "annualInc", "annualIncome"),
    "dti": ("dti", "debt_to_income", "debtToIncome"),
    "revol_util": ("revol_util", "revolUtil", "revol_utilization"),
    "fico_range_low": ("fico_range_low", "ficoRangeLow"),
    "fico_range_high": ("fico_range_high", "ficoRangeHigh"),
    "term": ("term",),
    "grade": ("grade",),
    "sub_grade": ("sub_grade", "subGrade"),
    "home_ownership": ("home_ownership", "homeownership", "homeOwnership"),
    "verification_status": ("verification_status", "verified_income", "isIncV"),
    "purpose": ("purpose", "loan_purpose"),
    "addr_state": ("addr_state", "state", "addrState"),
    "delinq_2yrs": ("delinq_2yrs", "delinq_2y", "delinq2Yrs"),
    "earliest_cr_line": ("earliest_cr_line", "earliest_credit_line", "earliestCrLine"),
    "inq_last_6mths": ("inq_last_6mths", "inqLast6Mths"),
    "open_acc": ("open_acc", "open_credit_lines", "openAcc"),
    "pub_rec": ("pub_rec", "pubRec"),
    "revol_bal": ("revol_bal", "revolBal"),
    "total_acc": ("total_acc", "total_credit_lines", "totalAcc"),
}


def build_alias_lookup(
    column_aliases: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return a case-insensitive mapping from source names to canonical names."""
    aliases = {column: list(values) for column, values in DEFAULT_COLUMN_ALIASES.items()}
    if column_aliases:
        for canonical, values in column_aliases.items():
            if isinstance(values, str):
                aliases.setdefault(canonical, []).append(values)
            else:
                aliases.setdefault(canonical, []).extend(values)

    lookup: dict[str, str] = {}
    for canonical, values in aliases.items():
        for value in (canonical, *values):
            lookup[str(value).strip().lower()] = canonical
    return lookup


def normalize_column_names(
    loans: pd.DataFrame,
    *,
    column_aliases: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Rename supported source columns to the pipeline's canonical names."""
    lookup = build_alias_lookup(column_aliases)
    rename_map: dict[str, str] = {}
    claimed: set[str] = set()

    for column in loans.columns:
        canonical = lookup.get(str(column).strip().lower())
        if canonical is None or canonical in claimed:
            continue
        rename_map[column] = canonical
        claimed.add(canonical)

    return loans.rename(columns=rename_map).copy()


def normalize_issue_date(series: pd.Series) -> pd.Series:
    """Normalize common issue date formats to pandas timestamps."""
    text = series.astype("string").str.strip()
    return pd.to_datetime(text, errors="coerce", format="mixed")


def normalize_credit_schema(
    loans: pd.DataFrame,
    *,
    column_aliases: Mapping[str, Any] | None = None,
    date_column: str = "issue_d",
) -> pd.DataFrame:
    """Return loans with canonical names and normalized date values."""
    result = normalize_column_names(loans, column_aliases=column_aliases)
    if date_column in result.columns:
        result[date_column] = normalize_issue_date(result[date_column])
    return result
