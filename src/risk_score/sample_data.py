"""Synthetic Lending Club-shaped loan data for tests, demos, and CI.

Why this module exists
----------------------
The real Lending Club extract is 1.19 GB and cannot be committed, so without a
generator there is no way to prove the pipeline works in CI, and no way for
someone cloning the repo to see a result without a Kaggle download.

The output is deliberately *raw*: original Lending Club column names, percent
strings, ``' 36 months'`` term text, ``'Aug-2003'`` credit-line dates, junk
columns, and realistic missingness. Anything that hands the pipeline
pre-cleaned data would test the pipeline against a fiction.

The one non-obvious design choice is :func:`make_synthetic_loans` reproducing
**outcome-maturity survivorship bias** rather than sampling labels directly.
Each loan gets a latent default probability, a draw, and - for defaulters - a
default *month*. Status is then derived from what an observer at ``snapshot``
would actually see. A loan that will default in month 30 of a 36-month term but
is only 18 months old still reads as ``Current``, while one that defaulted in
month 6 already reads ``Charged Off``.

That single mechanism is what makes recent vintages look far riskier than they
are once you filter to closed loans, which is the effect
``risk_score.data_loading.apply_outcome_maturity_embargo`` exists to correct.
Tests can therefore assert the embargo works on data that has the bias for the
same reason the real data does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

# Terminal and non-terminal statuses as they appear in the real extract,
# including the two "Does not meet the credit policy" legacy prefixes, so the
# status vocabulary in tests matches what data_loading must handle.
STATUS_FULLY_PAID = "Fully Paid"
STATUS_CHARGED_OFF = "Charged Off"
STATUS_CURRENT = "Current"
STATUS_LATE = "Late (31-120 days)"
STATUS_GRACE = "In Grace Period"
STATUS_POLICY_PAID = "Does not meet the credit policy. Status:Fully Paid"
STATUS_POLICY_OFF = "Does not meet the credit policy. Status:Charged Off"

_EMP_LENGTHS = (
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
)
_PURPOSES = (
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "other",
    "major_purchase",
    "medical",
    "small_business",
    "car",
    "moving",
    "vacation",
    "house",
    "wedding",
    "renewable_energy",
    "educational",
)
# Weights roughly track the real purpose distribution: debt consolidation and
# credit card together are about 75% of originations.
_PURPOSE_WEIGHTS = (
    0.58,
    0.22,
    0.06,
    0.04,
    0.03,
    0.02,
    0.015,
    0.01,
    0.008,
    0.006,
    0.005,
    0.003,
    0.002,
    0.002,
)
_HOME_OWNERSHIP = ("MORTGAGE", "RENT", "OWN", "OTHER")
_HOME_WEIGHTS = (0.49, 0.40, 0.10, 0.01)
_VERIFICATION = ("Source Verified", "Verified", "Not Verified")
_VERIFICATION_WEIGHTS = (0.40, 0.32, 0.28)
_STATES = (
    "CA",
    "NY",
    "TX",
    "FL",
    "IL",
    "NJ",
    "PA",
    "OH",
    "GA",
    "VA",
    "NC",
    "MI",
    "MD",
    "AZ",
    "WA",
    "MA",
    "CO",
)
_GRADE_LETTERS = ("A", "B", "C", "D", "E", "F", "G")
# Upper interest-rate bound for each grade, from the real grade/rate mapping.
# Used to derive grade from the priced rate so the two stay mutually consistent
# the way they are in the source data.
_GRADE_RATE_EDGES = (8.0, 11.5, 15.0, 19.0, 23.0, 26.5, np.inf)


def _normalized(weights: tuple[float, ...]) -> npt.NDArray[np.float64]:
    """Rescale hand-written weights to sum to exactly 1.

    ``Generator.choice`` rejects probabilities that miss 1.0 by more than a
    tolerance, so hand-edited weight tuples above would otherwise break the
    generator the moment someone adjusts one category.
    """
    array = np.asarray(weights, dtype=np.float64)
    return np.asarray(array / array.sum(), dtype=np.float64)


def _sigmoid(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Numerically stable logistic transform."""
    return np.asarray(1.0 / (1.0 + np.exp(-values)), dtype=np.float64)


def _solve_intercept(linear_scores: npt.NDArray[np.float64], target_rate: float) -> float:
    """Return the intercept that makes the mean default probability match a target.

    The feature coefficients below are chosen for a realistic *ranking* of risk,
    which leaves the overall level wherever it lands. Rather than hand-tuning a
    magic intercept every time a coefficient changes, solve for it: the mean of
    ``sigmoid(scores + b)`` is strictly increasing in ``b``, so 60 bisection
    steps pin it to well under a basis point.
    """
    low, high = -20.0, 20.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if float(_sigmoid(linear_scores + mid).mean()) < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _monthly_installment(
    principal: npt.NDArray[np.floating[Any]],
    annual_rate_pct: npt.NDArray[np.floating[Any]],
    term_months: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.float64]:
    """Standard amortizing payment, so `installment` is consistent with its inputs.

    Deriving it instead of drawing it independently matters: `installment` is a
    LENDER_PRICED feature, and its predictive power comes from being an exact
    function of the priced rate. A random `installment` would understate how
    much leakage the lender-priced tier actually carries.
    """
    monthly_rate = annual_rate_pct / 100.0 / 12.0
    growth = (1.0 + monthly_rate) ** term_months
    payment = principal * monthly_rate * growth / (growth - 1.0)
    return np.asarray(np.round(payment, 2), dtype=np.float64)


def _format_month(periods: pd.PeriodIndex) -> pd.Series:
    """Format monthly periods the way the raw extract does, e.g. ``Aug-2003``."""
    return pd.Series(periods.strftime("%b-%Y"), dtype="object")


def make_synthetic_loans(
    *,
    n_rows: int = 4000,
    seed: int = 20130101,
    start_month: str = "2013-01",
    end_month: str = "2016-12",
    # 2019-06 against a 2013-01..2016-12 origination window reproduces the real
    # vintage bias closely: closed-only default rate runs 14.7% / 17.0% / 17.7%
    # / 31.8% by year, against a true lifetime rate of 14.1%. The real extract
    # shows 15.6 / 18.5 / 20.2 / 24.3 for the same reason.
    snapshot: str = "2019-06",
    target_default_rate: float = 0.15,
    percent_strings: bool = True,
    include_fico: bool = True,
    include_post_origination: bool = True,
    include_junk_columns: bool = True,
) -> pd.DataFrame:
    """Build a raw Lending Club-shaped frame with a learnable, biased outcome.

    Parameters
    ----------
    n_rows:
        Number of loans. 4000 is enough for a tri-split to have positives in
        every partition while staying fast enough for the default test run.
    seed:
        Seeds a ``numpy.random.default_rng``. Output is fully deterministic.
    start_month, end_month:
        Inclusive origination window, ``YYYY-MM``.
    snapshot:
        The month the extract is "pulled". Loans whose outcome is not yet known
        at this point read as ``Current``, which is what creates the
        survivorship bias the embargo has to remove.
    target_default_rate:
        Lifetime default rate across *all* originations. The rate measured on
        closed loans only will be higher, by design.
    percent_strings:
        When True, ``int_rate`` and ``revol_util`` are strings like
        ``'13.56%'``. When False they are floats. Both real extracts exist, so
        tests parametrize over this.
    include_fico:
        The `data/raw/1/loan.csv` extract has no ``fico_range_*`` columns at
        all, so the pipeline has to work without them.
    include_post_origination:
        Adds servicing columns that the leakage filter must remove.
    include_junk_columns:
        Adds ``id``, ``url``, ``desc``, ``zip_code`` and similar, so the
        unknown-column audit has something to report.

    Returns
    -------
    pandas.DataFrame
        Unsorted raw rows. Deliberately unsorted: the split must not rely on
        input order.
    """
    if n_rows < 1:
        raise ValueError("`n_rows` must be at least 1.")
    if not 0.0 < target_default_rate < 1.0:
        raise ValueError("`target_default_rate` must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    issue_range = pd.period_range(start=start_month, end=end_month, freq="M")
    if len(issue_range) == 0:
        raise ValueError("`start_month` must not be after `end_month`.")
    snapshot_period = pd.Period(snapshot, freq="M")

    # --- 1. origination attributes -------------------------------------------
    issue_periods = pd.PeriodIndex(rng.choice(issue_range, size=n_rows), freq="M")
    # 36-month loans dominate, and 60-month is only offered above a size floor.
    term_months = np.where(rng.random(n_rows) < 0.72, 36, 60)

    annual_inc = np.round(rng.lognormal(mean=11.0, sigma=0.55, size=n_rows), 2)
    annual_inc = np.clip(annual_inc, 4_000.0, 2_500_000.0)
    # Loan size tracks income but is capped at the real $40k program maximum and
    # quantized to $25 the way the platform does.
    loan_amnt = np.clip(annual_inc * rng.uniform(0.05, 0.45, n_rows), 1_000.0, 40_000.0)
    loan_amnt = np.round(loan_amnt / 25.0) * 25.0

    dti = np.round(np.clip(rng.gamma(shape=6.0, scale=3.1, size=n_rows), 0.0, 60.0), 2)
    revol_util_pct = np.round(np.clip(rng.beta(2.2, 2.0, n_rows) * 128.0, 0.0, 145.0), 1)
    revol_bal = np.round(np.clip(rng.lognormal(9.2, 1.05, n_rows), 0.0, 900_000.0), 0)

    fico_mid = np.clip(np.round(rng.normal(698.0, 31.0, n_rows)), 660.0, 850.0)
    delinq_2yrs = rng.poisson(0.28, n_rows)
    inq_last_6mths = rng.poisson(0.72, n_rows)
    pub_rec = rng.poisson(0.18, n_rows)
    open_acc = np.clip(rng.poisson(11.4, n_rows), 1, None)
    total_acc = open_acc + rng.poisson(13.0, n_rows)
    emp_index = rng.integers(0, len(_EMP_LENGTHS), n_rows)
    # '10+ years' is by far the most common value in the real data; bias toward it.
    emp_index = np.where(rng.random(n_rows) < 0.33, len(_EMP_LENGTHS) - 1, emp_index)
    credit_history_months = np.clip(rng.normal(200.0, 78.0, n_rows), 24.0, 560.0).astype(int)

    # --- 2. latent risk ------------------------------------------------------
    # Coefficients set the ranking of risk; the intercept is solved so the mean
    # matches `target_default_rate`. Signs are the ones credit risk actually
    # shows: utilization and DTI up is worse, FICO and tenure up is better.
    loan_to_income = loan_amnt / annual_inc
    linear = (
        1.85 * (revol_util_pct / 100.0)
        + 0.030 * dti
        - 0.0115 * (fico_mid - 698.0)
        + 1.10 * loan_to_income
        + 0.26 * delinq_2yrs
        + 0.17 * inq_last_6mths
        + 0.21 * pub_rec
        - 0.045 * emp_index
        - 0.0016 * (credit_history_months - 200.0)
        # 60-month loans are riskier at equal observable quality.
        + 0.34 * (term_months == 60)
        + rng.normal(0.0, 0.62, n_rows)
    )
    linear += _solve_intercept(linear, target_default_rate)
    default_probability = _sigmoid(linear)
    will_default = rng.random(n_rows) < default_probability

    # --- 3. lender pricing (the LENDER_PRICED tier) --------------------------
    # The lender sees the same risk and prices it. This is exactly why int_rate,
    # grade, sub_grade, and installment are near-label proxies rather than
    # ordinary features: they encode the underwriter's own risk estimate.
    priced_risk = _sigmoid(linear + rng.normal(0.0, 0.30, n_rows))
    int_rate = np.round(np.clip(5.32 + 23.0 * priced_risk, 5.32, 30.99), 2)
    grade_index = np.searchsorted(np.asarray(_GRADE_RATE_EDGES), int_rate, side="left")
    grade_index = np.clip(grade_index, 0, len(_GRADE_LETTERS) - 1)
    grade = np.asarray(_GRADE_LETTERS)[grade_index]
    sub_grade = np.char.add(grade, rng.integers(1, 6, n_rows).astype(str))
    installment = _monthly_installment(loan_amnt, int_rate, term_months)

    # --- 4. observed status at the snapshot ----------------------------------
    maturity_periods = issue_periods + term_months
    # Charge-off timing is front-loaded: Beta(2, 3) puts the mass in the first
    # 40% of the term, matching Lending Club's observed default curve.
    default_offset = np.ceil(rng.beta(2.0, 3.0, n_rows) * term_months).astype(int)
    default_periods = issue_periods + default_offset

    matured = maturity_periods <= snapshot_period
    default_visible = will_default & (default_periods <= snapshot_period)

    status = np.full(n_rows, STATUS_CURRENT, dtype=object)
    status[default_visible] = STATUS_CHARGED_OFF
    status[~will_default & matured] = STATUS_FULLY_PAID
    # A small slice of the legacy "does not meet the credit policy" statuses,
    # which the closed-status filter and the target map both have to recognize.
    legacy = rng.random(n_rows) < 0.012
    status[legacy & (status == STATUS_FULLY_PAID)] = STATUS_POLICY_PAID
    status[legacy & (status == STATUS_CHARGED_OFF)] = STATUS_POLICY_OFF
    # Loans mid-delinquency at the snapshot: non-terminal, so they must be
    # dropped rather than scored as either outcome.
    delinquent = will_default & ~default_visible & (rng.random(n_rows) < 0.30)
    status[delinquent & (rng.random(n_rows) < 0.5)] = STATUS_LATE
    status[delinquent & (status == STATUS_CURRENT)] = STATUS_GRACE

    # --- 5. assemble the raw frame -------------------------------------------
    frame: dict[str, object] = {
        "loan_amnt": loan_amnt,
        "funded_amnt": loan_amnt,
        # Term is text with a leading space in the real CSV, which is why the
        # parser strips before matching.
        "term": [f" {value} months" for value in term_months],
        "int_rate": [f"{value:.2f}%" for value in int_rate] if percent_strings else int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": np.asarray(_EMP_LENGTHS)[emp_index],
        "home_ownership": rng.choice(_HOME_OWNERSHIP, n_rows, p=_normalized(_HOME_WEIGHTS)),
        "annual_inc": annual_inc,
        "verification_status": rng.choice(
            _VERIFICATION, n_rows, p=_normalized(_VERIFICATION_WEIGHTS)
        ),
        "issue_d": _format_month(issue_periods),
        "loan_status": status,
        "purpose": rng.choice(_PURPOSES, n_rows, p=_normalized(_PURPOSE_WEIGHTS)),
        "addr_state": rng.choice(_STATES, n_rows),
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "earliest_cr_line": _format_month(issue_periods - credit_history_months),
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": (
            [f"{value:.1f}%" for value in revol_util_pct] if percent_strings else revol_util_pct
        ),
        "total_acc": total_acc,
    }
    if include_fico:
        # The real columns are a 4- or 5-point band, never a single score.
        frame["fico_range_low"] = fico_mid - 2.0
        frame["fico_range_high"] = fico_mid + 2.0
    if include_post_origination:
        repaid_fraction = np.where(will_default, rng.uniform(0.05, 0.7, n_rows), 1.0)
        total_pymnt = np.round(installment * term_months * repaid_fraction, 2)
        frame["out_prncp"] = np.where(matured, 0.0, np.round(loan_amnt * 0.35, 2))
        frame["total_pymnt"] = total_pymnt
        frame["total_rec_prncp"] = np.round(total_pymnt * 0.82, 2)
        frame["total_rec_int"] = np.round(total_pymnt * 0.18, 2)
        frame["recoveries"] = np.where(will_default, np.round(loan_amnt * 0.06, 2), 0.0)
        frame["last_pymnt_d"] = _format_month(
            pd.PeriodIndex(np.minimum(default_periods, maturity_periods), freq="M")
        )
        frame["last_pymnt_amnt"] = installment
        frame["last_credit_pull_d"] = _format_month(
            pd.PeriodIndex(np.full(n_rows, snapshot_period), freq="M")
        )
        # Post-origination FICO is the single most leaky column in the extract:
        # it drops ~100 points at charge-off.
        frame["last_fico_range_high"] = np.where(will_default, fico_mid - 95.0, fico_mid + 18.0)
        frame["last_fico_range_low"] = np.asarray(frame["last_fico_range_high"]) - 4.0
    if include_junk_columns:
        frame["id"] = np.arange(1_000_000, 1_000_000 + n_rows)
        frame["member_id"] = np.full(n_rows, np.nan)
        frame["url"] = [f"https://example.invalid/loan/{index}" for index in range(n_rows)]
        frame["desc"] = np.full(n_rows, np.nan)
        frame["zip_code"] = [f"{rng.integers(100, 999)}xx" for _ in range(n_rows)]
        frame["policy_code"] = np.ones(n_rows)

    loans = pd.DataFrame(frame)

    # --- 6. missingness ------------------------------------------------------
    # Rates taken from the real extract. These are what make the imputer and
    # the missing-indicator columns worth having, so they are not optional.
    for column, rate in (
        ("revol_util", 0.005),
        ("emp_length", 0.058),
        ("dti", 0.002),
        ("annual_inc", 0.001),
        ("earliest_cr_line", 0.001),
        ("inq_last_6mths", 0.001),
    ):
        if column not in loans.columns:
            continue
        blanks = rng.random(n_rows) < rate
        if blanks.any():
            # `mask` preserves the column's dtype where it can; integer columns
            # widen to float exactly as a real `read_csv` would when a value is
            # blank, so the frame stays faithful to the source.
            loans[column] = loans[column].mask(blanks, other=np.nan)

    # Shuffle so nothing downstream can accidentally depend on rows arriving in
    # date order.
    return loans.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_sample_dataset(
    output_path: str | Path,
    **kwargs: object,
) -> Path:
    """Write a synthetic raw CSV and return the path.

    Backs ``riskscore make-sample-data`` so the demo and the CI smoke test need
    no external download.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    loans = make_synthetic_loans(**kwargs)  # type: ignore[arg-type]
    loans.to_csv(destination, index=False)
    return destination
