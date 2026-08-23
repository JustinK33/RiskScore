"""Parsers and derived features, as pure Series functions.

Two things changed here, and both were structural rather than arithmetic.

**Every function returned a copy of the whole frame (audit P02).** The old
``add_*`` functions each took a DataFrame, called ``.copy()``, appended one
column, and returned the frame; ``build_feature_matrix`` copied once more on
entry. That is seven full copies of a 1.8M-row frame to produce four columns.
Each builder here returns a single Series, and ``build_feature_matrix`` does one
``assign`` and one ``drop``.

**Nothing dropped the raw column it replaced (audit B01).** ``add_dti_feature``
added ``dti_clean`` and left ``dti`` in place, so both reached the preprocessor -
and since the preprocessor routed columns by runtime dtype, the raw percent
string went to ``OneHotEncoder``. Which column each feature *consumes* is now
declared in :data:`risk_score.features.ENGINEERED_FEATURES` and honored here.

Parsing is by *declared* :class:`~risk_score.features.ParseKind`, never by
sniffing the sample. A one-row ``/predict`` request has no distribution to sniff,
so any sample-dependent rule would make serving disagree with training on the
same applicant. Where a sanity check needs a sample it is an assertion that
raises, never a branch that changes the output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from risk_score.features import ENGINEERED_FEATURES, ParseKind

_LOGGER = logging.getLogger(__name__)

#: Numeric sentinels Lending Club uses for "not available". They are real values
#: in the CSV and would otherwise be modelled as genuine extreme observations.
MISSING_SENTINELS: frozenset[float] = frozenset({-1.0, 9999.0, 999999.0})

#: Utilization above this is treated as an outlier and winsorized rather than
#: dropped. Real ``revol_util`` runs past 800% in the full extract - genuinely
#: over-limit borrowers - but a single 8.9 in a scaled feature dominates the
#: coefficient, and the risk signal saturates long before 200%.
MAX_CREDIT_UTILIZATION = 2.0

#: A dti this large is a sentinel or a units error, not a borrower. The extract's
#: legitimate range reaches the low hundreds for joint applications.
MAX_PLAUSIBLE_DTI = 1000.0

#: Sample size below which distribution-based sanity checks are skipped. A
#: single-applicant scoring request cannot support them, and firing on one
#: unusual input would take the service down for a valid request.
MIN_ROWS_FOR_SANITY_CHECK = 100


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Parse a numeric column to plain ``float64``, sentinels and infinities to NaN.

    ``float64`` rather than a nullable extension dtype (audit B24): ``Int64`` and
    ``Float64`` propagate through arithmetic and then reach scikit-learn, which
    converts them to ``object`` and fails deep inside a transformer with a
    message that does not name the column.
    """
    if pd.api.types.is_numeric_dtype(series) and not isinstance(
        series.dtype, pd.api.extensions.ExtensionDtype
    ):
        values = series.astype("float64")
    else:
        # Thousands separators and stray currency symbols appear in re-exported
        # extracts; % is handled here too so parse_percent can reuse this.
        cleaned = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        values = pd.to_numeric(cleaned, errors="coerce").astype("float64")

    # inf survives to_numeric ('inf' parses) and then breaks StandardScaler with
    # an error about NaN, which sends you looking in the wrong place.
    return values.replace([np.inf, -np.inf], np.nan).mask(values.isin(MISSING_SENTINELS))


def parse_percent(series: pd.Series, *, column_name: str = "percent") -> pd.Series:
    """Parse a percent-denominated column to a fraction.

    The divisor is fixed at 100 because the registry declares the column's units.
    Inferring the scale from the data would let a batch of low-utilization
    applicants be read as already-fractional and silently scored 100x too low.
    A sample-based check guards the declaration instead of replacing it.
    """
    fraction = coerce_numeric(series) / 100.0
    observed = fraction.dropna()
    if len(observed) >= MIN_ROWS_FOR_SANITY_CHECK and float(observed.max()) <= 0.02:
        # Everything under 2% after dividing means the source was already a
        # fraction, so this divided it twice. Loud, because the model would still
        # fit and just be quietly wrong.
        raise ValueError(
            f"Column `{column_name}` is declared as a percentage but its maximum "
            f"value after dividing by 100 is {float(observed.max()):.5f}, which "
            f"means the source was already a fraction. Change the column's "
            f"ParseKind in risk_score.features to NUMERIC."
        )
    return fraction


def parse_term_months(series: pd.Series) -> pd.Series:
    """Parse ``' 36 months'`` to ``36.0``.

    Digit extraction rather than strip-and-split: the two real extracts differ in
    leading whitespace and one writes ``'36 months'`` without it.
    """
    if pd.api.types.is_numeric_dtype(series):
        return coerce_numeric(series)
    digits = series.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(digits, errors="coerce").astype("float64")


def parse_employment_years(series: pd.Series) -> pd.Series:
    """Parse ``'10+ years'`` -> 10, ``'< 1 year'`` -> 0, ``'3 years'`` -> 3.

    ``'10+'`` is censored at 10 in the source, so 10 means "10 or more". Nothing
    here can recover the true value; the ceiling is the data's, not a choice.
    """
    if pd.api.types.is_numeric_dtype(series):
        return coerce_numeric(series)
    text = series.astype("string").str.strip()
    # '< 1 year' has no digit other than the 1, which would read as one year of
    # employment rather than under one, so it is matched before digit extraction.
    under_one = text.str.contains("<", na=False)
    digits = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce")
    return digits.mask(under_one, 0).astype("float64")


def parse_column(series: pd.Series, kind: ParseKind, *, column_name: str = "column") -> pd.Series:
    """Apply the parser the registry declares for a column."""
    if kind is ParseKind.NUMERIC:
        return coerce_numeric(series)
    if kind is ParseKind.PERCENT:
        return parse_percent(series, column_name=column_name)
    if kind is ParseKind.TERM_MONTHS:
        return parse_term_months(series)
    if kind is ParseKind.EMP_LENGTH_YEARS:
        return parse_employment_years(series)
    # CATEGORY, TEXT, and MONTH_DATE are not this function's job: dates are
    # parsed in schema.py with one whole-column format, and categories are
    # normalized by the encoder.
    raise ValueError(
        f"parse_column does not handle ParseKind.{kind.name} (column `{column_name}`). "
        f"Dates are parsed in risk_score.schema; categories in the encoder."
    )


def build_dti_clean(loans: pd.DataFrame) -> pd.Series:
    """Debt-to-income as a float, with sentinels and impossible values removed."""
    dti = coerce_numeric(loans["dti"])
    # A negative dti is the extract's -1 sentinel surviving a different spelling;
    # an enormous one is a units error. Both become missing so the imputer
    # handles them, rather than anchoring the scaler.
    return dti.mask((dti < 0) | (dti > MAX_PLAUSIBLE_DTI))


def build_credit_utilization(loans: pd.DataFrame) -> pd.Series:
    """Revolving utilization as a fraction in ``[0, MAX_CREDIT_UTILIZATION]``.

    Falls back to ``total_credit_utilized / total_credit_limit``, which the
    ``loans_full_schema`` extract carries instead of ``revol_util``. Both paths
    produce a fraction, which the old code's two branches did not: one divided by
    100 and the other did not, so the feature's units depended on which extract
    was loaded.
    """
    if "revol_util" in loans.columns:
        utilization = parse_percent(loans["revol_util"], column_name="revol_util")
    elif {"total_credit_utilized", "total_credit_limit"}.issubset(loans.columns):
        utilized = coerce_numeric(loans["total_credit_utilized"])
        limit = coerce_numeric(loans["total_credit_limit"])
        # A zero limit is "no revolving account", not "infinite utilization".
        utilization = utilized.div(limit.where(limit > 0))
    else:
        raise KeyError(
            "credit_utilization needs `revol_util`, or both `total_credit_utilized` "
            f"and `total_credit_limit`. Present: {sorted(loans.columns)[:15]}."
        )
    # Fix B20: negative utilization is impossible, and the old code applied no
    # bound at all, so a genuine 8.92 dominated every scaled coefficient.
    return utilization.mask(utilization < 0).clip(upper=MAX_CREDIT_UTILIZATION)


def build_loan_to_income_ratio(loans: pd.DataFrame) -> pd.Series:
    """Requested principal divided by annual income.

    Zero *and negative* income both become missing. The old version guarded only
    zero, so a negative income - which the extract does contain - produced a
    negative ratio that the model read as a very low risk of default.
    """
    loan_amount = coerce_numeric(loans["loan_amnt"])
    annual_income = coerce_numeric(loans["annual_inc"])
    # Fix B21: the old guard was `.replace(0, np.nan)`, which let a negative
    # income through as a negative ratio the model read as very low risk.
    return loan_amount.div(annual_income.where(annual_income > 0))


def build_credit_history_months(loans: pd.DataFrame) -> pd.Series:
    """Months from the borrower's first credit line to origination.

    This is the feature ``earliest_cr_line`` exists to produce. Without it the
    raw date string reached the preprocessor and was one-hot encoded into 655
    columns (audit B01) - discarding an ordered quantity to produce noise.
    """
    earliest = loans["earliest_cr_line"]
    issued = loans["issue_d"]
    for name, column in (("earliest_cr_line", earliest), ("issue_d", issued)):
        if not pd.api.types.is_datetime64_any_dtype(column):
            raise TypeError(
                f"`{name}` must already be datetime, got {column.dtype}. "
                f"Run risk_score.schema.normalize_credit_schema first."
            )
    # Whole months from the calendar fields directly. `(issued - earliest).dt.days
    # / 30.44` would invent a fractional part the source does not have, and
    # period subtraction returns offset objects that need a per-row .apply.
    # NaT yields NaN through .dt.year, so missing dates propagate for free.
    months = (issued.dt.year - earliest.dt.year) * 12 + (issued.dt.month - earliest.dt.month)
    months = months.astype("float64")
    # A credit line that starts after origination is a data error, not a
    # borrower with negative history.
    return months.mask(months < 0)


def build_fico_midpoint(loans: pd.DataFrame) -> pd.Series:
    """Midpoint of the origination FICO band.

    Absent from both real extracts, which is worth knowing before reading a
    feature-importance chart that does not mention FICO.
    """
    low = coerce_numeric(loans["fico_range_low"])
    high = coerce_numeric(loans["fico_range_high"])
    return (low + high) / 2.0


def build_fico_band(loans: pd.DataFrame) -> pd.Series:
    """Interpretable FICO bucket. For reporting, not for accuracy.

    Binning discards information a linear model can already use, so this exists
    to make a reason code readable ("fair credit") rather than to improve the
    score.
    """
    # right=False so a score of exactly 670 is 'good', matching how the bands are
    # published rather than how pd.cut defaults.
    return pd.cut(
        loans["fico_midpoint"],
        bins=[0, 580, 670, 740, 800, np.inf],
        labels=["poor", "fair", "good", "very_good", "exceptional"],
        right=False,
    )


#: Builder per engineered feature name. Keyed by the same names as
#: :data:`risk_score.features.ENGINEERED_FEATURES`, and checked against it at
#: import so a feature cannot be declared without a builder or vice versa.
FEATURE_BUILDERS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "dti_clean": build_dti_clean,
    "credit_utilization": build_credit_utilization,
    "loan_to_income_ratio": build_loan_to_income_ratio,
    "credit_history_months": build_credit_history_months,
    "fico_midpoint": build_fico_midpoint,
    "fico_band": build_fico_band,
}

_declared = {feature.name for feature in ENGINEERED_FEATURES}
if _declared != set(FEATURE_BUILDERS):
    raise RuntimeError(
        f"ENGINEERED_FEATURES and FEATURE_BUILDERS disagree: "
        f"declared-only {sorted(_declared - set(FEATURE_BUILDERS))}, "
        f"built-only {sorted(set(FEATURE_BUILDERS) - _declared)}."
    )


def build_feature_matrix(loans: pd.DataFrame, *, require_all: bool = False) -> pd.DataFrame:
    """Add every buildable derived feature and drop the raw columns they replace.

    A feature whose inputs are absent is skipped rather than raising, because the
    two real extracts carry different columns: ``fico_range_*`` exists in neither
    and ``revol_util`` in only one. ``require_all=True`` turns that into an error,
    which is what the training pipeline wants once the extract is known.

    One ``assign`` and one ``drop``, in that order. Building into a dict first
    means a feature can consume a column that a later feature also reads - the
    drop happens after every builder has run.
    """
    built: dict[str, pd.Series] = {}
    consumed: set[str] = set()
    skipped: list[str] = []

    for feature in ENGINEERED_FEATURES:
        available = set(loans.columns) | set(built)
        missing = [name for name in feature.requires if name not in available]
        if missing:
            if require_all:
                raise KeyError(
                    f"Cannot build `{feature.name}`: missing {missing}. "
                    f"Present columns: {sorted(loans.columns)[:15]}."
                )
            skipped.append(feature.name)
            continue
        # Builders that read a previously built feature (fico_band reads
        # fico_midpoint) need it visible, so the frame grows as we go. This is
        # the one place the declared build order in ENGINEERED_FEATURES matters.
        source = loans.assign(**built) if built else loans
        built[feature.name] = FEATURE_BUILDERS[feature.name](source)
        consumed.update(feature.consumes)

    if skipped:
        # Named, not silent: a run that quietly built four of six features looks
        # identical in the metrics to one that built all six.
        _LOGGER.info("skipped derived features (inputs absent): %s", ", ".join(skipped))

    result = loans.assign(**built) if built else loans
    to_drop = [name for name in consumed if name in result.columns]
    return result.drop(columns=to_drop) if to_drop else result
