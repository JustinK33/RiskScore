"""Leakage control: prove, per run, that nothing unpermitted reached the model.

The previous version was two hand-maintained lists and a drop. It had three
problems, and the third is the one that mattered.

1. It duplicated column knowledge that ``schema.py`` and
   ``configs/feature_config.yaml`` also held, and the three disagreed. All of it
   now comes from :mod:`risk_score.features`.
2. The deny-list pass was redundant with the allow-list pass that ran
   immediately after it in the pipeline, so the first one dropped columns the
   second was about to drop anyway.
3. **An unrecognized column was silently kept.** A deny list can only reject
   what someone thought to name. A future extract adding
   ``settlement_amount`` - post-charge-off, a near-perfect predictor of default -
   would have sailed through as an ordinary feature. The audit here is
   allow-list-first: a column the registry does not classify is a *finding*, not
   a feature.

The output is a :class:`LeakageAudit` that goes into the run manifest, so
"leakage was controlled" is a checkable record of which columns were admitted
and which were refused for what reason, rather than a claim in a README.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from risk_score.features import (
    COLUMN_REGISTRY,
    ENGINEERED_BY_NAME,
    FeatureTier,
    model_feature_columns,
)

#: Tiers that are never features, with the reason a reader needs. Kept as a
#: mapping rather than a set because the reason is what makes an audit line
#: reviewable: "dropped 17 columns" is not.
REFUSAL_REASONS: dict[FeatureTier, str] = {
    FeatureTier.POST_ORIGINATION: (
        "Known only after the loan was funded, so it encodes the outcome."
    ),
    FeatureTier.IDENTIFIER: "Row identifier, URL, or free text - no borrower risk content.",
    FeatureTier.TARGET_SOURCE: "The label is derived from it.",
    FeatureTier.TIMELINE: "Used to split by time; feeding it in would fit the calendar.",
    FeatureTier.LENDER_PRICED: (
        "The lender's own risk estimate, already fitted to default. Available at "
        "origination but not usable to score an applicant nobody has priced yet. "
        "Opt in with include_lender_priced=True."
    ),
}


@dataclass(frozen=True)
class LeakageAudit:
    """Which columns were admitted as features, and why the rest were not."""

    admitted: tuple[str, ...]
    """Columns that reached the model, in frame order."""

    refused: dict[str, str] = field(default_factory=dict)
    """Column -> reason it was excluded."""

    unclassified: tuple[str, ...] = ()
    """Columns the registry does not know. Blocked, and worth a look."""

    include_lender_priced: bool = False
    """The tier setting this audit was run under."""

    @property
    def is_clean(self) -> bool:
        """True when every column in the frame was classified.

        An unclassified column is not proof of leakage, but it is proof that
        nobody has decided, which is the state this module exists to surface.
        """
        return not self.unclassified

    def summary(self) -> str:
        return (
            f"features={len(self.admitted)} refused={len(self.refused)} "
            f"unclassified={len(self.unclassified)} "
            f"lender_priced={'on' if self.include_lender_priced else 'off'}"
        )


def audit_columns(
    columns: Iterable[str],
    *,
    include_lender_priced: bool = False,
    keep_columns: Iterable[str] = (),
) -> LeakageAudit:
    """Classify every column as admitted, refused with a reason, or unrecognized.

    ``keep_columns`` are passed through untouched - the target and any join key
    the caller needs downstream. They are not reported as features.
    """
    allowed = set(model_feature_columns(include_lender_priced=include_lender_priced))
    engineered = set(ENGINEERED_BY_NAME)
    protected = set(keep_columns)

    admitted: list[str] = []
    refused: dict[str, str] = {}
    unclassified: list[str] = []

    for column in columns:
        name = str(column)
        if name in protected:
            continue
        if name in allowed or name in engineered:
            admitted.append(name)
            continue
        spec = COLUMN_REGISTRY.get(name)
        if spec is None:
            # Allow-list-first: an unknown column is refused *and* flagged. A
            # deny list would have admitted it.
            unclassified.append(name)
            continue
        refused[name] = REFUSAL_REASONS.get(
            spec.tier, f"Tier {spec.tier.value} is not admitted as a feature."
        )

    return LeakageAudit(
        admitted=tuple(admitted),
        refused=refused,
        unclassified=tuple(unclassified),
        include_lender_priced=include_lender_priced,
    )


def select_model_features(
    loans: pd.DataFrame,
    *,
    include_lender_priced: bool = False,
    keep_columns: Iterable[str] = ("default_flag", "issue_d"),
    strict: bool = False,
) -> tuple[pd.DataFrame, LeakageAudit]:
    """Reduce a frame to admitted features plus ``keep_columns``.

    One allow-list pass, replacing the old deny-then-allow pair where the first
    pass dropped columns the second was about to drop anyway (audit B26).

    ``issue_d`` is kept by default because the time-based split needs it, and
    :func:`audit_columns` does not count it as a feature. It stays in the frame
    all the way into the fitted ``Pipeline`` - ``credit_history_months`` is
    derived from it - and what keeps it away from the estimator is that
    :class:`~risk_score.transformers.FeatureSpec` does not declare it as one.

    ``strict=True`` raises on an unclassified column instead of dropping it.
    That is the training pipeline's setting: onboarding a new extract should be a
    deliberate act, not something a run does quietly.
    """
    audit = audit_columns(
        loans.columns, include_lender_priced=include_lender_priced, keep_columns=keep_columns
    )
    if strict and not audit.is_clean:
        raise ValueError(
            f"Unclassified columns present: {list(audit.unclassified)}. Add a "
            f"ColumnSpec for each in risk_score.features - deciding a column's "
            f"tier is how post-origination leakage stays out."
        )

    keep = [
        column
        for column in loans.columns
        if column in audit.admitted or column in set(keep_columns)
    ]
    return loans.loc[:, keep], audit
