"""Raw ingestion: read the extract, filter to usable outcomes, build the label.

Everything here happens before any feature exists. The three jobs are reading
only the columns the pipeline needs, restricting rows to loans whose outcome is
actually known, and turning a status string into a binary label.

The second job is the subtle one. Filtering to closed statuses looks obviously
correct and is quietly wrong, because "closed" is measured against the extract's
snapshot date. A 36-month loan issued in 2016 has closed by a 2018 snapshot only
if it *defaulted early*; the ones still paying on schedule read as ``Current``
and get dropped. Measured default rate by vintage in ``data/raw/1/loan.csv``
climbs 15.6% (2013) -> 18.5% -> 20.2% -> 24.3% (2016) and then falls back to
14.7% (2018) - a shape produced entirely by the snapshot, not by credit quality.
:func:`apply_outcome_maturity_embargo` removes the censored vintages and
flattens it to a steady ~14.9%. See
``docs/decisions/0004-outcome-maturity-embargo.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from risk_score.features import COLUMN_REGISTRY, ParseKind, alias_lookup, columns_to_read
from risk_score.schema import DEFAULT_DATE_FORMATS, SchemaReport, normalize_credit_schema

#: Statuses representing a terminal outcome. A `frozenset` rather than a `set`
#: because it is a default argument value: a caller who mutated it would change
#: the default for every other caller in the process.
CLOSED_LOAN_STATUSES: frozenset[str] = frozenset(
    {
        "Fully Paid",
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off",
    }
)

#: Statuses that mean the borrower stopped paying.
DEFAULT_STATUSES: frozenset[str] = frozenset(
    {
        "charged off",
        "default",
        "does not meet the credit policy. status:charged off",
    }
)

#: Statuses that mean the borrower repaid in full.
PAID_STATUSES: frozenset[str] = frozenset(
    {
        "fully paid",
        "does not meet the credit policy. status:fully paid",
    }
)


@dataclass(frozen=True)
class EmbargoResult:
    """Rows kept by the maturity embargo, plus what it removed and why.

    The per-vintage rates are the point of the whole exercise, so they travel
    with the data into the run manifest instead of being printed and lost.
    """

    loans: pd.DataFrame
    snapshot: pd.Timestamp
    rows_in: int
    rows_out: int
    rows_immature: int
    rows_unknown_maturity: int
    default_rate_before: pd.Series
    """Default rate by issue year *before* the embargo, index = year."""
    default_rate_after: pd.Series
    """Default rate by issue year *after* the embargo, index = year."""

    def summary(self) -> str:
        return (
            f"embargo snapshot={self.snapshot.date()} "
            f"kept={self.rows_out}/{self.rows_in} "
            f"immature={self.rows_immature} unknown_maturity={self.rows_unknown_maturity}"
        )


def _projected_read_plan(
    header: Iterable[str],
    *,
    column_aliases: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Decide which source columns to read and which to force to text.

    Returns ``(usecols, dtypes)``, both keyed by *source* name, which is why the
    header has to be read first: the dtype map cannot be written until the
    extract's own spelling of each column is known.
    """
    lookup = alias_lookup(column_aliases)
    wanted = set(columns_to_read())

    usecols: list[str] = []
    dtypes: dict[str, str] = {}
    for name in header:
        canonical = lookup.get(str(name).strip().lower())
        if canonical is None or canonical not in wanted:
            continue
        usecols.append(str(name))
        # Text-ish columns are declared so the C parser never has to guess, and
        # so a percent string can't be half-read as a float in one chunk and an
        # object in the next. Genuinely numeric columns are left to inference:
        # declaring float64 would make the whole read *fail* on a single 'n/a'
        # cell, and to_numeric downstream handles that case per value instead.
        if COLUMN_REGISTRY[canonical].parse is not ParseKind.NUMERIC:
            dtypes[str(name)] = "string"
    return usecols, dtypes


def read_raw_loans(
    path: str | Path,
    *,
    column_aliases: Mapping[str, Any] | None = None,
    project: bool = True,
    date_formats: Iterable[str] = DEFAULT_DATE_FORMATS,
) -> tuple[pd.DataFrame, SchemaReport]:
    """Read a raw extract and canonicalize it, reading only what is needed.

    The Lending Club extract has 145 columns and this pipeline uses about 27.
    The previous ``pd.read_csv(path, low_memory=False)`` read all of them, which
    projects to roughly 5x the peak memory on the 1.8M-row file - the difference
    between a run that finishes and one the kernel kills (audit P01).

    ``project=False`` reads every column, which is only useful for auditing an
    unfamiliar extract; the report's ``unknown_columns`` covers the normal case.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file does not exist: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        if project:
            # nrows=0 reads the header alone - cheap, and the only way to learn
            # the extract's column spelling before committing to a dtype map.
            header = pd.read_csv(data_path, nrows=0).columns
            usecols, dtypes = _projected_read_plan(header, column_aliases=column_aliases)
            if not usecols:
                raise ValueError(
                    f"None of the {len(header)} columns in {data_path} matched a known "
                    f"column. Found: {list(header)[:10]}. Add aliases under "
                    f"`column_aliases` in configs/run.yaml."
                )
            loans = pd.read_csv(data_path, usecols=usecols, dtype=dtypes)
        else:
            loans = pd.read_csv(data_path, low_memory=False)
    elif suffix in {".parquet", ".pq"}:
        # Column projection is native here, but it needs canonical names, which
        # for parquet is safe: this project only ever writes parquet itself.
        loans = pd.read_parquet(data_path)
    else:
        raise ValueError(
            f"Unsupported raw data format {suffix!r} for {data_path}. Expected .csv or .parquet."
        )

    return normalize_credit_schema(
        loans, column_aliases=column_aliases, date_formats=tuple(date_formats)
    )


def filter_to_closed_loans(
    loans: pd.DataFrame,
    *,
    closed_statuses: Iterable[str] = CLOSED_LOAN_STATUSES,
    status_column: str = "loan_status",
) -> pd.DataFrame:
    """Keep only loans whose status is terminal.

    Necessary but not sufficient - see :func:`apply_outcome_maturity_embargo`,
    which removes the vintages where "terminal" and "defaulted early" are the
    same set of rows.
    """
    if status_column not in loans.columns:
        raise KeyError(
            f"Expected status column `{status_column}` is missing. "
            f"Present columns: {sorted(loans.columns)[:15]}."
        )

    normalized_targets = {status.strip().lower() for status in closed_statuses}
    status = loans[status_column].astype("string").str.strip().str.lower()
    return loans.loc[status.isin(normalized_targets)]


def term_months(loans: pd.DataFrame, *, term_column: str = "term") -> pd.Series:
    """The loan term as a nullable integer month count.

    ``term`` arrives as ``' 36 months'`` in one extract and ``'36 months'`` in
    the other, so the digits are extracted rather than the string split: both
    spellings, and any future one that still writes the number, resolve the same
    way. A value with no digits becomes NA rather than raising, because the two
    callers below want to *count* those rows, not abort on them.
    """
    if term_column not in loans.columns:
        raise KeyError(
            f"Expected term column `{term_column}` is missing. "
            f"Present columns: {sorted(loans.columns)[:15]}."
        )
    return pd.to_numeric(
        loans[term_column].astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )


def filter_to_terms(
    loans: pd.DataFrame,
    *,
    terms: Iterable[int],
    term_column: str = "term",
) -> pd.DataFrame:
    """Keep only loans whose term is one of ``terms``. An empty ``terms`` keeps all.

    This exists because of the embargo, not independently of it. Once immature
    loans are removed, 60-month loans survive only in the earliest vintages - in
    the shipped configuration they are 28% of the 2013 rows and 0% of every later
    one. Training on that mix and scoring a term the model never sees in
    validation or test is a train/serve mismatch dressed up as more data, so the
    default restricts to 36-month loans and the drift report states the cliff
    that justifies it.

    Rows whose term does not parse are dropped, on the same reasoning as the
    embargo: a loan that cannot be shown to be in scope is not in scope.
    """
    wanted = {int(term) for term in terms}
    if not wanted:
        return loans
    return loans.loc[term_months(loans, term_column=term_column).isin(wanted)]


def load_lending_club_data(
    path: str | Path,
    *,
    closed_statuses: Iterable[str] = CLOSED_LOAN_STATUSES,
    column_aliases: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Read an extract and filter it to closed loans.

    Convenience wrapper over :func:`read_raw_loans` and
    :func:`filter_to_closed_loans` for callers that do not need the schema
    report. The pipeline uses the two functions directly, because the report
    goes into the run manifest.
    """
    loans, _report = read_raw_loans(path, column_aliases=column_aliases)
    return filter_to_closed_loans(loans, closed_statuses=closed_statuses)


def create_default_target(
    loans: pd.DataFrame,
    *,
    status_column: str = "loan_status",
    target_column: str = "default_flag",
) -> pd.Series:
    """Map a terminal status to 1 (default), 0 (paid), or NA (neither).

    Returns the label as a Series rather than a copy of the whole frame: the
    previous version copied every column to add one, which on 1.8M rows is a
    gigabyte of churn for a single Int64 column (audit P02).

    Non-terminal statuses become NA rather than 0. Treating ``Current`` as
    "did not default" is the same survivorship mistake the embargo exists to
    prevent, one level down.
    """
    if status_column not in loans.columns:
        raise KeyError(f"Expected status column `{status_column}` is missing.")

    normalized_status = loans[status_column].astype("string").str.strip().str.lower()
    # NA-first, then two positive assignments, so a status in neither set stays NA
    # by construction. The alternative - `isin(DEFAULT_STATUSES).astype(int)` -
    # collapses "paid" and "unknown" into the same 0 and is exactly the bug the
    # docstring above warns about. The two sets are disjoint, so the order of
    # these two lines does not matter; `tests/test_data_loading.py` pins that.
    target = pd.Series(pd.NA, index=loans.index, dtype="Int64", name=target_column)
    target[normalized_status.isin(DEFAULT_STATUSES)] = 1
    target[normalized_status.isin(PAID_STATUSES)] = 0
    return target


def _default_rate_by_vintage(loans: pd.DataFrame, *, date_column: str = "issue_d") -> pd.Series:
    """Default rate per issue year. Empty Series when the label cannot be built."""
    if date_column not in loans.columns or "loan_status" not in loans.columns:
        return pd.Series(dtype="float64", name="default_rate")
    target = create_default_target(loans)
    # `dropna` covers both columns at once, and both matter: a non-terminal status
    # is NA rather than 0 (see `create_default_target`) and an unparseable issue
    # date has no vintage to attribute. Either one silently included would make
    # this rate disagree with the label the model is actually trained on, which is
    # the one number this function exists to let a reader check by eye.
    frame = pd.DataFrame({"year": loans[date_column].dt.year, "default_flag": target}).dropna()
    if frame.empty:
        return pd.Series(dtype="float64", name="default_rate")
    rate = frame.groupby("year")["default_flag"].mean().astype("float64")
    rate.name = "default_rate"
    return rate


def apply_outcome_maturity_embargo(
    loans: pd.DataFrame,
    *,
    snapshot: str | pd.Timestamp,
    date_column: str = "issue_d",
    term_column: str = "term",
) -> EmbargoResult:
    """Keep only loans whose full term had elapsed by the snapshot date.

    The rule is ``issue_d + term_months <= snapshot``. A loan that fails it can
    only appear in a closed-loan filter by having defaulted, so including it
    biases the label upward - and the bias grows with vintage, which makes a
    time-based split look like genuine credit-quality drift.

    Rows with an unparseable date or term are removed and counted separately:
    they cannot be shown to be mature, and quietly keeping them would defeat the
    purpose. The counts go into the manifest so a run that discards half its
    input does not look identical to one that discards none.
    """
    for required in (date_column, term_column):
        if required not in loans.columns:
            raise KeyError(
                f"The outcome-maturity embargo needs `{required}`, which is missing. "
                f"Present columns: {sorted(loans.columns)[:15]}."
            )

    if not pd.api.types.is_datetime64_any_dtype(loans[date_column]):
        # Re-parsing here would reintroduce audit B28 in a second place, with a
        # different format policy from schema.py. Dates arrive already parsed.
        raise TypeError(
            f"`{date_column}` must already be datetime; got "
            f"{loans[date_column].dtype}. Run risk_score.schema."
            f"normalize_credit_schema first, which parses it with one explicit "
            f"whole-column format."
        )

    snapshot_ts = pd.Timestamp(snapshot)
    if snapshot_ts.tzinfo is not None:
        # Comparing a tz-aware cutoff against tz-naive issue dates raises inside
        # pandas with a message that does not mention the snapshot argument.
        raise ValueError(
            f"Snapshot {snapshot!r} is timezone-aware but issue dates are naive. "
            f"Pass a naive date such as '2018-12-01'."
        )

    rate_before = _default_rate_by_vintage(loans, date_column=date_column)

    issued = loans[date_column]
    months = term_months(loans, term_column=term_column)

    known = issued.notna() & months.notna()
    # DateOffset arithmetic per row is slow on millions of rows; converting the
    # month count to a period offset keeps it vectorized.
    #
    # `fillna(0)` is not a maturity decision, it is what `astype("int64")` needs:
    # a nullable Int64 with NA in it will not cast. Those rows say "matured at
    # issue", which would be the wrong answer, and they never reach it - `known`
    # is ANDed in below and they are counted as `rows_unknown_maturity`.
    matured_by = issued.dt.to_period("M") + months.fillna(0).astype("int64")
    # Both sides are truncated to month start, so the comparison is at the
    # precision the data actually has: `issue_d` is `Dec-2015` in the extract and
    # the day is an artifact of parsing. It also makes the rule insensitive to how
    # a caller spells the snapshot - '2018-12-01' and '2018-12-31' keep the same
    # rows, rather than one silently admitting a vintage the other embargoes.
    mature = known & (matured_by.dt.to_timestamp() <= snapshot_ts.to_period("M").to_timestamp())

    kept = loans.loc[mature]
    result = EmbargoResult(
        loans=kept,
        snapshot=snapshot_ts,
        rows_in=len(loans),
        rows_out=len(kept),
        rows_immature=int((known & ~mature).sum()),
        rows_unknown_maturity=int((~known).sum()),
        default_rate_before=rate_before,
        default_rate_after=_default_rate_by_vintage(kept, date_column=date_column),
    )
    if result.rows_out == 0:
        observed = issued.dropna()
        span = (
            f"{observed.min().date()} to {observed.max().date()}" if not observed.empty else "none"
        )
        raise ValueError(
            f"The outcome-maturity embargo removed every row. Snapshot is "
            f"{snapshot_ts.date()} and issue dates span {span}. Either the "
            f"snapshot predates the data or the term column did not parse."
        )
    return result
