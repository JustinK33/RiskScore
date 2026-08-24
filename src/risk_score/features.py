"""The canonical column registry: one source of truth for every column.

Why this module exists
----------------------
Column knowledge used to be spread across three files that each held part of the
answer and none of it authoritatively:

* ``schema.py`` had ``DEFAULT_COLUMN_ALIASES`` - names, no types, no policy.
* ``leakage_check.py`` had ``ORIGINATION_TIME_COLUMNS`` and
  ``POST_ORIGINATION_COLUMNS`` - policy, no names for aliases, no types.
* ``configs/feature_config.yaml`` had numeric/categorical lists - and was loaded
  by nothing at all.

They disagreed. ``configs/feature_config.yaml`` listed ``fico_range_low`` as a
model feature; the real extract has no such column. ``leakage_check.py`` treated
``int_rate`` as an ordinary origination-time feature; it is the lender's own
risk estimate. And because nothing declared a *type*, the preprocessor inferred
one at runtime from whatever pandas happened to parse, which is how
``earliest_cr_line`` - a date string with 655 distinct values - ended up
one-hot encoded.

Every column now has exactly one entry here, carrying its aliases, its parsing
rule, and its feature tier. Downstream code asks this registry rather than
guessing, so a column cannot be a feature in one module and leakage in another.

Feature tiers
-------------
The tier is a *policy* statement, not a data type. See
``docs/decisions/0005-lender-priced-feature-tier.md`` for the full argument; in
short:

``BORROWER`` / ``LOAN_REQUEST``
    Included by default. Borrower attributes and what the applicant asked for.

``LENDER_PRICED``
    **Excluded by default.** ``int_rate``, ``grade``, ``sub_grade``, and
    ``installment`` are all available at origination, so a leakage check based
    on timing waves them through. But they are the underwriter's own risk
    estimate, already fitted to default. A model built on them largely
    reproduces an existing decision instead of predicting from borrower
    attributes, and it cannot be used to score an applicant the lender has not
    already priced. Opt in with ``include_lender_priced=True``; both variants
    get reported so the difference is measured, not asserted.

``POST_ORIGINATION``
    Never a feature, at any setting. Known only after the loan was funded.

``IDENTIFIER``
    Never a feature: row ids, URLs, free text, and constant policy codes.

``TARGET_SOURCE`` / ``TIMELINE``
    Needed by the pipeline (to build the label, to split by date) but never fed
    to the model.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum


class FeatureTier(StrEnum):
    """Why a column is or is not allowed to reach the model."""

    BORROWER = "borrower"
    LOAN_REQUEST = "loan_request"
    LENDER_PRICED = "lender_priced"
    POST_ORIGINATION = "post_origination"
    IDENTIFIER = "identifier"
    TARGET_SOURCE = "target_source"
    TIMELINE = "timeline"


class ParseKind(StrEnum):
    """How a raw value becomes a usable one.

    The raw extract stores numbers as text in several different disguises, and
    the disguise is a property of the column, not something to sniff at runtime.
    """

    NUMERIC = "numeric"
    """Plain number, possibly quoted or thousands-separated."""

    PERCENT = "percent"
    """``'13.56%'`` or a bare float. Both real extracts exist."""

    TERM_MONTHS = "term_months"
    """``' 36 months'`` -> ``36``."""

    EMP_LENGTH_YEARS = "emp_length_years"
    """``'10+ years'`` -> ``10``, ``'< 1 year'`` -> ``0``."""

    MONTH_DATE = "month_date"
    """``'Aug-2003'`` or ``'2003-08-01'`` -> a timestamp."""

    CATEGORY = "category"
    """Low-cardinality string, one-hot encoded."""

    TEXT = "text"
    """Free text or an identifier. Never a feature."""


#: Parse kinds that produce a numeric column, so the preprocessor scales them.
NUMERIC_PARSE_KINDS: frozenset[ParseKind] = frozenset(
    {
        ParseKind.NUMERIC,
        ParseKind.PERCENT,
        ParseKind.TERM_MONTHS,
        ParseKind.EMP_LENGTH_YEARS,
    }
)

#: Tiers whose columns may be fed to the model at some setting.
MODELABLE_TIERS: frozenset[FeatureTier] = frozenset(
    {FeatureTier.BORROWER, FeatureTier.LOAN_REQUEST, FeatureTier.LENDER_PRICED}
)

#: Shortest alias we will accept. The original registry mapped ``'n'`` to
#: ``loan_status``, so any dataset with a column literally named ``n`` had its
#: target silently replaced by unrelated data. Requiring three characters makes
#: that class of collision impossible to reintroduce by accident.
MIN_ALIAS_LENGTH = 3


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Everything the pipeline knows about one canonical column."""

    name: str
    tier: FeatureTier
    parse: ParseKind
    aliases: tuple[str, ...] = ()
    required: bool = False
    """If True, the run aborts when the column is absent rather than imputing it."""
    description: str = ""

    @property
    def is_numeric(self) -> bool:
        """Whether this column reaches the model as a scaled numeric feature."""
        return self.parse in NUMERIC_PARSE_KINDS

    def match_keys(self) -> tuple[str, ...]:
        """Lookup keys in priority order, canonical name first.

        Order is the whole point. When a frame contains both ``loan_amnt`` and
        ``funded_amnt``, the canonical name must win; resolving in arbitrary
        column order is what produced two columns both named ``loan_amnt``
        (audit B02).
        """
        return (self.name, *self.aliases)


# --- the registry --------------------------------------------------------------
# Ordering is documentation: target and timeline first, then what the applicant
# asked for, then borrower attributes, then the lender's own pricing, then
# everything that must never reach the model.
_SPECS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        name="loan_status",
        tier=FeatureTier.TARGET_SOURCE,
        parse=ParseKind.CATEGORY,
        # 'n' deliberately removed - see MIN_ALIAS_LENGTH.
        aliases=("loanstatus", "status"),
        required=True,
        description="Terminal or in-flight loan outcome. The label is derived from this.",
    ),
    ColumnSpec(
        name="issue_d",
        tier=FeatureTier.TIMELINE,
        parse=ParseKind.MONTH_DATE,
        aliases=("issue_month", "issuedate", "issuemonth", "issue_d_month"),
        required=True,
        description="Origination month. Defines the time split and the vintage breakdown.",
    ),
    ColumnSpec(
        name="loan_amnt",
        tier=FeatureTier.LOAN_REQUEST,
        parse=ParseKind.NUMERIC,
        aliases=("loan_amount", "loanamnt", "funded_amnt", "fundedamnt"),
        required=True,
        description="Requested principal in dollars.",
    ),
    ColumnSpec(
        name="term",
        tier=FeatureTier.LOAN_REQUEST,
        parse=ParseKind.TERM_MONTHS,
        aliases=("term_months",),
        required=True,
        description="Loan term in months. Also drives the outcome-maturity embargo.",
    ),
    ColumnSpec(
        name="purpose",
        tier=FeatureTier.LOAN_REQUEST,
        parse=ParseKind.CATEGORY,
        aliases=("loan_purpose",),
        description="Borrower-stated use of funds.",
    ),
    ColumnSpec(
        name="annual_inc",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("annual_income", "annualinc", "annualincome"),
        required=True,
        description="Self-reported annual income.",
    ),
    ColumnSpec(
        name="emp_length",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.EMP_LENGTH_YEARS,
        aliases=("emp_length_years", "employment_length", "emplength"),
        description="Years at current employer, capped at 10 by the source encoding.",
    ),
    ColumnSpec(
        name="home_ownership",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.CATEGORY,
        aliases=("homeownership",),
        description="RENT, MORTGAGE, OWN, or OTHER.",
    ),
    ColumnSpec(
        name="verification_status",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.CATEGORY,
        aliases=("verified_income", "isincv"),
        description="Whether the platform verified stated income.",
    ),
    ColumnSpec(
        name="addr_state",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.CATEGORY,
        aliases=("state", "addrstate"),
        description="Two-letter state of residence.",
    ),
    ColumnSpec(
        name="dti",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("debt_to_income", "debttoincome"),
        description="Debt-to-income ratio excluding the requested loan, as a percentage.",
    ),
    ColumnSpec(
        name="revol_util",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.PERCENT,
        aliases=("revolutil", "revol_utilization"),
        description="Revolving line utilization. Exceeds 100% in the real data.",
    ),
    ColumnSpec(
        name="revol_bal",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("revolbal",),
        description="Total revolving balance.",
    ),
    ColumnSpec(
        name="delinq_2yrs",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("delinq_2y", "delinq2yrs"),
        description="30+ day delinquencies in the last two years.",
    ),
    ColumnSpec(
        name="inq_last_6mths",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("inqlast6mths",),
        description="Credit inquiries in the last six months, excluding auto and mortgage.",
    ),
    ColumnSpec(
        name="open_acc",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("open_credit_lines", "openacc"),
        description="Open credit lines.",
    ),
    ColumnSpec(
        name="total_acc",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("total_credit_lines", "totalacc"),
        description="Total credit lines ever opened.",
    ),
    ColumnSpec(
        name="pub_rec",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("pubrec",),
        description="Derogatory public records.",
    ),
    ColumnSpec(
        name="earliest_cr_line",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.MONTH_DATE,
        aliases=("earliest_credit_line", "earliestcrline"),
        description=(
            "First credit line opened. Used only to derive credit_history_months; "
            "never a feature itself, because as a raw string it has 655 distinct "
            "values in the real extract (audit B01)."
        ),
    ),
    ColumnSpec(
        name="total_credit_utilized",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("totalcreditutilized",),
        description="Bureau-style total balance; a fallback source for utilization.",
    ),
    ColumnSpec(
        name="total_credit_limit",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("totalcreditlimit",),
        description="Bureau-style total limit; the denominator for the utilization fallback.",
    ),
    ColumnSpec(
        name="fico_range_low",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("ficorangelow",),
        description=(
            "Lower bound of the origination FICO band. Absent from both real "
            "extracts in data/raw/, so the pipeline must work without it."
        ),
    ),
    ColumnSpec(
        name="fico_range_high",
        tier=FeatureTier.BORROWER,
        parse=ParseKind.NUMERIC,
        aliases=("ficorangehigh",),
        description="Upper bound of the origination FICO band.",
    ),
    ColumnSpec(
        name="int_rate",
        tier=FeatureTier.LENDER_PRICED,
        parse=ParseKind.PERCENT,
        aliases=("interest_rate", "intrate"),
        description="Assigned interest rate: the lender's own risk estimate, priced.",
    ),
    ColumnSpec(
        name="grade",
        tier=FeatureTier.LENDER_PRICED,
        parse=ParseKind.CATEGORY,
        description="Assigned letter grade. A deterministic function of int_rate.",
    ),
    ColumnSpec(
        name="sub_grade",
        tier=FeatureTier.LENDER_PRICED,
        parse=ParseKind.CATEGORY,
        aliases=("subgrade",),
        description="Assigned sub-grade, a finer slice of the same pricing decision.",
    ),
    ColumnSpec(
        name="installment",
        tier=FeatureTier.LENDER_PRICED,
        parse=ParseKind.NUMERIC,
        description="Monthly payment: an exact function of amount, rate, and term.",
    ),
    # --- post-origination: never features ------------------------------------
    ColumnSpec(
        name="out_prncp",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Outstanding principal, known only once payments occur.",
    ),
    ColumnSpec(
        name="out_prncp_inv",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Outstanding investor principal.",
    ),
    ColumnSpec(
        name="total_pymnt",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Payments received to date.",
    ),
    ColumnSpec(
        name="total_pymnt_inv",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Investor payments received to date.",
    ),
    ColumnSpec(
        name="total_rec_prncp",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Principal received to date.",
    ),
    ColumnSpec(
        name="total_rec_int",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Interest received to date.",
    ),
    ColumnSpec(
        name="total_rec_late_fee",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Late fees charged, which directly reveal repayment behavior.",
    ),
    ColumnSpec(
        name="recoveries",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Post-charge-off recoveries. Non-zero implies the label.",
    ),
    ColumnSpec(
        name="collection_recovery_fee",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Collection fee, charged only after delinquency.",
    ),
    ColumnSpec(
        name="last_pymnt_d",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.MONTH_DATE,
        description="Last payment date.",
    ),
    ColumnSpec(
        name="last_pymnt_amnt",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Last payment amount.",
    ),
    ColumnSpec(
        name="next_pymnt_d",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.MONTH_DATE,
        description="Next scheduled payment date: servicing state.",
    ),
    ColumnSpec(
        name="last_credit_pull_d",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.MONTH_DATE,
        description="Most recent credit pull, which may postdate origination.",
    ),
    ColumnSpec(
        name="last_fico_range_high",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description=(
            "Updated FICO. The single leakiest column in the extract: it falls "
            "roughly 100 points at charge-off."
        ),
    ),
    ColumnSpec(
        name="last_fico_range_low",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Updated FICO lower bound. Same problem.",
    ),
    ColumnSpec(
        name="collections_12_mths_ex_med",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.NUMERIC,
        description="Collections in 12 months; reporting window may postdate origination.",
    ),
    ColumnSpec(
        name="pymnt_plan",
        tier=FeatureTier.POST_ORIGINATION,
        parse=ParseKind.CATEGORY,
        description="Payment plan flag, set during servicing.",
    ),
    ColumnSpec(
        name="policy_code",
        tier=FeatureTier.IDENTIFIER,
        parse=ParseKind.NUMERIC,
        description="Constant 1.0 in every public extract. Encodes nothing.",
    ),
    ColumnSpec(
        name="id",
        tier=FeatureTier.IDENTIFIER,
        parse=ParseKind.TEXT,
        description="Row identifier. Correlates with time, so it leaks the vintage.",
    ),
    ColumnSpec(
        name="member_id",
        tier=FeatureTier.IDENTIFIER,
        parse=ParseKind.TEXT,
        description="Fully null in every public extract.",
    ),
    ColumnSpec(
        name="url",
        tier=FeatureTier.IDENTIFIER,
        parse=ParseKind.TEXT,
        description="Listing URL. Contains the row id.",
    ),
    ColumnSpec(
        name="desc",
        tier=FeatureTier.IDENTIFIER,
        parse=ParseKind.TEXT,
        description="Free-text borrower description. Out of scope for this model.",
    ),
    ColumnSpec(
        name="zip_code",
        tier=FeatureTier.IDENTIFIER,
        parse=ParseKind.TEXT,
        description=(
            "Truncated ZIP. Excluded on fair-lending grounds: it proxies for "
            "protected characteristics far more tightly than addr_state."
        ),
    ),
)


def _validate_registry(specs: Iterable[ColumnSpec]) -> dict[str, ColumnSpec]:
    """Index the registry by canonical name, rejecting anything ambiguous.

    Run once at import. A registry that maps one alias to two canonical columns
    resolves by dict insertion order, which is a silent, order-dependent bug -
    so it is a hard error at import time instead.
    """
    by_name: dict[str, ColumnSpec] = {}
    seen_keys: dict[str, str] = {}
    for spec in specs:
        if spec.name in by_name:
            raise ValueError(f"Duplicate canonical column in registry: {spec.name!r}.")
        by_name[spec.name] = spec
        # The length floor applies to aliases only. A canonical name is the
        # column's literal name in the source extract - 'id' is genuinely two
        # characters - whereas an alias is a guess about what some other extract
        # might have called it, and a two-character guess will eventually match
        # something unrelated.
        _reject_short_aliases(spec)
        for key in spec.match_keys():
            normalized = key.strip().lower()
            owner = seen_keys.get(normalized)
            if owner is not None and owner != spec.name:
                raise ValueError(f"Alias {key!r} is claimed by both {owner!r} and {spec.name!r}.")
            seen_keys[normalized] = spec.name
    return by_name


def _reject_short_aliases(spec: ColumnSpec) -> None:
    """Reject aliases too short to be unambiguous (audit B03)."""
    for alias in spec.aliases:
        if len(alias.strip()) < MIN_ALIAS_LENGTH:
            raise ValueError(
                f"Alias {alias!r} for {spec.name!r} is shorter than "
                f"{MIN_ALIAS_LENGTH} characters. Short aliases collide with "
                f"unrelated columns; 'n' once hijacked loan_status this way."
            )


COLUMN_REGISTRY: dict[str, ColumnSpec] = _validate_registry(_SPECS)


def column_spec(name: str) -> ColumnSpec:
    """Look up one canonical column, with a message that lists the near misses."""
    try:
        return COLUMN_REGISTRY[name]
    except KeyError:
        candidates = [known for known in COLUMN_REGISTRY if name.lower() in known]
        hint = f" Did you mean one of {candidates}?" if candidates else ""
        raise KeyError(f"Unknown canonical column {name!r}.{hint}") from None


def alias_lookup(
    extra_aliases: Mapping[str, str | Iterable[str]] | None = None,
) -> dict[str, str]:
    """Return a lowercase source-name -> canonical-name map.

    ``extra_aliases`` comes from ``configs/run.yaml`` and lets a new
    extract be onboarded without a code change. It is validated to the same
    standard as the built-in registry: too-short aliases and cross-canonical
    collisions are rejected rather than resolved by luck.
    """
    lookup: dict[str, str] = {}
    for spec in COLUMN_REGISTRY.values():
        for key in spec.match_keys():
            lookup[key.strip().lower()] = spec.name

    if not extra_aliases:
        return lookup

    for canonical, values in extra_aliases.items():
        if canonical not in COLUMN_REGISTRY:
            raise KeyError(
                f"Config declares aliases for unknown canonical column {canonical!r}. "
                f"Add a ColumnSpec to risk_score.features first."
            )
        candidates = (values,) if isinstance(values, str) else tuple(values)
        for value in candidates:
            normalized = str(value).strip().lower()
            if len(normalized) < MIN_ALIAS_LENGTH:
                raise ValueError(
                    f"Configured alias {value!r} for {canonical!r} is shorter than "
                    f"{MIN_ALIAS_LENGTH} characters."
                )
            owner = lookup.get(normalized)
            if owner is not None and owner != canonical:
                raise ValueError(
                    f"Configured alias {value!r} for {canonical!r} is already claimed by {owner!r}."
                )
            lookup[normalized] = canonical
    return lookup


def alias_priority(canonical: str) -> tuple[str, ...]:
    """Lookup keys for one column in priority order, canonical name first."""
    return column_spec(canonical).match_keys()


def model_feature_columns(*, include_lender_priced: bool = False) -> tuple[str, ...]:
    """Raw canonical columns eligible to become features, in registry order.

    Excludes ``LENDER_PRICED`` unless asked, and always excludes
    ``earliest_cr_line``: it is a date that exists to produce
    ``credit_history_months``, and feeding the raw string to a one-hot encoder is
    audit bug B01.
    """
    allowed = set(MODELABLE_TIERS)
    if not include_lender_priced:
        allowed.discard(FeatureTier.LENDER_PRICED)
    return tuple(
        spec.name
        for spec in COLUMN_REGISTRY.values()
        if spec.tier in allowed and spec.parse is not ParseKind.MONTH_DATE
    )


def columns_in_tier(tier: FeatureTier) -> tuple[str, ...]:
    """Every canonical column in one tier, in registry order."""
    return tuple(spec.name for spec in COLUMN_REGISTRY.values() if spec.tier is tier)


def required_columns() -> tuple[str, ...]:
    """Columns whose absence aborts the run rather than being imputed."""
    return tuple(spec.name for spec in COLUMN_REGISTRY.values() if spec.required)


def columns_to_read(*, include_lender_priced: bool = False) -> tuple[str, ...]:
    """Canonical columns the pipeline needs, for projection at read time.

    The real extract has 145 columns and the pipeline uses about 30. Reading
    only these cuts peak memory by roughly 5x, which is the difference between
    a run that completes and one that is killed (audit P01).

    ``LENDER_PRICED`` columns are always read even when excluded from the model,
    because the leakage-cost comparison needs to fit both variants from one
    read. Post-origination columns are read too - they are needed to *audit*
    that they were dropped, and reading nothing would make the audit vacuous.
    """
    keep = {
        FeatureTier.TARGET_SOURCE,
        FeatureTier.TIMELINE,
        FeatureTier.BORROWER,
        FeatureTier.LOAN_REQUEST,
        FeatureTier.LENDER_PRICED,
    }
    _ = include_lender_priced  # read regardless; the tier filter happens later
    return tuple(spec.name for spec in COLUMN_REGISTRY.values() if spec.tier in keep)


@dataclass(frozen=True, slots=True)
class EngineeredFeature:
    """A derived feature: what it needs, and what it makes redundant."""

    name: str
    requires: tuple[str, ...] = ()
    """Canonical columns that must *all* be present for this feature to be built."""
    requires_any: tuple[tuple[str, ...], ...] = ()
    """Alternative input groups; at least one group must be fully present.

    This exists because ``credit_utilization`` has two sources: the standard
    extract carries ``revol_util`` and ``loans_full_schema`` carries
    ``total_credit_utilized`` / ``total_credit_limit``. Declaring only the first
    as ``requires`` made the builder's own fallback unreachable - the feature was
    skipped on the very extract the fallback was written for.
    """
    consumes: tuple[str, ...] = ()
    """Raw columns dropped once the feature exists, because it replaces them."""
    numeric: bool = True
    description: str = ""

    def satisfied_by(self, available: Collection[str]) -> bool:
        """Whether every input this feature needs is present."""
        if not all(name in available for name in self.requires):
            return False
        return not self.requires_any or any(
            all(name in available for name in group) for group in self.requires_any
        )

    def unmet(self, available: Collection[str]) -> str:
        """What is missing, phrased for an error message. Empty when buildable."""
        parts: list[str] = []
        missing = [name for name in self.requires if name not in available]
        if missing:
            parts.append(f"missing {missing}")
        if self.requires_any and not any(
            all(name in available for name in group) for group in self.requires_any
        ):
            options = " or ".join(str(list(group)) for group in self.requires_any)
            parts.append(f"needs one of {options}")
        return "; ".join(parts)


#: Derived features, in build order. ``consumes`` is the fix for audit B01:
#: previously every ``add_*`` function appended a cleaned column and left the raw
#: source in place, so ``dti`` and ``dti_clean`` both reached the preprocessor and
#: the raw string version was one-hot encoded.
ENGINEERED_FEATURES: tuple[EngineeredFeature, ...] = (
    EngineeredFeature(
        name="dti_clean",
        requires=("dti",),
        consumes=("dti",),
        description="Debt-to-income parsed to a float and bounded to a plausible range.",
    ),
    EngineeredFeature(
        name="credit_utilization",
        requires_any=(("revol_util",), ("total_credit_utilized", "total_credit_limit")),
        # Only revol_util is consumed: it *is* this quantity, rescaled. The
        # bureau-style balance and limit are kept, because the levels carry
        # signal the ratio does not.
        consumes=("revol_util",),
        description=(
            "Revolving utilization as a fraction. Falls back to "
            "total_credit_utilized / total_credit_limit when revol_util is absent."
        ),
    ),
    EngineeredFeature(
        name="loan_to_income_ratio",
        requires=("loan_amnt", "annual_inc"),
        # Neither source is consumed: loan size and income each carry signal
        # beyond their ratio.
        description="Requested principal divided by annual income.",
    ),
    EngineeredFeature(
        name="credit_history_months",
        requires=("earliest_cr_line", "issue_d"),
        consumes=("earliest_cr_line",),
        description=(
            "Months between first credit line and origination. Replaces a raw "
            "date string that would otherwise be one-hot encoded into 655 columns."
        ),
    ),
    EngineeredFeature(
        name="fico_midpoint",
        requires=("fico_range_low", "fico_range_high"),
        consumes=("fico_range_low", "fico_range_high"),
        description="Midpoint of the origination FICO band. Absent from both real extracts.",
    ),
    EngineeredFeature(
        name="fico_band",
        requires=("fico_midpoint",),
        numeric=False,
        description="Interpretable FICO bucket, for reporting rather than accuracy.",
    ),
)

ENGINEERED_BY_NAME: dict[str, EngineeredFeature] = {
    feature.name: feature for feature in ENGINEERED_FEATURES
}


def resolvable_engineered_features(
    available: Iterable[str],
) -> tuple[tuple[EngineeredFeature, ...], tuple[EngineeredFeature, ...]]:
    """Split the declared features into ``(buildable, blocked)`` for one extract.

    Walked in declared order with each built feature added to what is available,
    because ``fico_band`` reads the ``fico_midpoint`` built one step earlier.

    Both the training pipeline and :func:`risk_score.transformers.build_feature_spec`
    call this. Two separate walks would eventually disagree, and the symptom
    would be a served row with a different feature set from the fitted model.
    """
    present = set(available)
    buildable: list[EngineeredFeature] = []
    blocked: list[EngineeredFeature] = []
    for feature in ENGINEERED_FEATURES:
        if feature.satisfied_by(present):
            buildable.append(feature)
            present.add(feature.name)
        else:
            blocked.append(feature)
    return tuple(buildable), tuple(blocked)
