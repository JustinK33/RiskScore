"""The human-readable outputs: the model card, the variant comparison, the shapes
the dashboard reads.

Three things live here because all three answer "what does a person see", and
none of them may compute a number of their own. Every value in a model card is
read out of ``metrics.json`` and ``manifest.json``; if a figure is not in one of
those files it does not belong on the card, because a card that derives its own
arithmetic is a second implementation of the run.

Why the card is generated in code rather than filled into a template
--------------------------------------------------------------------
A template plus a substitution pass has one failure mode that matters: a
placeholder nobody filled ships as ``$auc_roc`` in a document whose whole purpose
is to be trusted. The plan's own acceptance check for this feature was "no
``$placeholder`` survives in the card", which is a check that only exists because
templates leak. Building the markdown from typed accessors makes the failure
impossible rather than detectable: a missing key is a ``-`` placed deliberately
by :func:`format_metric`, and a missing *section* is a diff.

The static prose - intended use, out of scope, ethical considerations, the
standing limitations - is here as module constants. It is genuinely static: it
describes the modelling decisions, not one run's numbers, and a per-run copy of
it would drift between runs of the same code.

Why the comparison is quantified rather than asserted
-----------------------------------------------------
"Excluding the lender's own price costs a little accuracy" is the sort of claim
that is repeated until it is believed. :func:`comparison_table` fits the same
split under two tiers and prints the delta, so the cost of the leakage policy in
ADR 0005 is a number a reader can disagree with. The same function does the
LR-versus-XGBoost comparison, because "two variants of one split, one row each"
is the same table either way.

Where ``comparison.json`` lives, and why it is not inside a run directory
------------------------------------------------------------------------
At the report root, beside ``registry.json``, not in ``runs/<run_id>/``. A run
directory is published by one atomic rename and is immutable afterwards, and a
comparison is only complete once every variant has been published - so writing it
into one variant's directory would mean either reopening a published run or
declaring one variant the owner of a document about all of them. Root level also
matches what the file is: a statement about several runs, like the registry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from risk_score.artifacts import RunMetadata, feature_tier
from risk_score.drift import PSI_MODERATE, PSI_SIGNIFICANT
from risk_score.explain import feature_label
from risk_score.features import FeatureTier, columns_in_tier

#: The card, next to the metrics it summarizes.
MODEL_CARD_FILENAME = "model_card.md"

#: The comparison, at the report root rather than inside a run (see the module
#: docstring).
COMPARISON_FILENAME = "comparison.json"

#: Compared side by side, with the direction that counts as an improvement. The
#: direction is data rather than a comment because the table renders a delta
#: column and a reader must not have to remember that a lower Brier is better.
COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("auc_roc", "higher"),
    ("average_precision", "higher"),
    ("ks_statistic", "higher"),
    ("brier_score", "lower"),
    ("expected_calibration_error", "lower"),
    ("selected_threshold_total_cost", "lower"),
    ("psi_score", "lower"),
    # Neither better nor worse: it is a policy consequence, and a lender reading
    # a comparison needs to see that variant B's better AUC came with a different
    # share of the book approved.
    ("approval_rate", "neutral"),
)

#: Below these, a metric is a description of a handful of rows rather than of a
#: model. 1000 rows and 50 defaults is the point at which an AUC's standard error
#: drops under roughly 0.04, which is smaller than the differences this project
#: reports between variants. The committed artifact this project replaced had
#: **116 rows and one positive** and the dashboard rendered its 0.07 AUC as a
#: result, which is why these are published warnings and not a comment.
SANITY_MIN_TEST_ROWS = 1000
SANITY_MIN_TEST_POSITIVES = 50

_INTENDED_USE = """\
Screening applications for the product this extract describes: unsecured
consumer instalment loans, scored **at origination** from what an applicant
states plus a credit bureau pull.

The output is a calibrated probability of default over the full term and one
approve/decline recommendation against the published threshold. It is decision
*support*. The threshold encodes one cost ratio, stated below, and a lender whose
costs differ should reselect it on their own validation data rather than inherit
this one."""

_OUT_OF_SCOPE = """\
- **Pricing.** This model estimates risk; turning a risk into a rate is a
  separate decision with separate regulation.
- **Servicing and collections.** Only origination-time information is used, so
  the model knows nothing about how an existing loan has been paid.
- **Any other product or population.** A different term, a secured product, or a
  population unlike the training window is out of scope by construction, and the
  drift table is how that is detected rather than assumed.
- **Unreviewed adverse action.** The reason codes are exact and are meant to be
  read by a person before a decline is communicated."""

_ETHICAL = """\
**No protected attribute is a feature.** Race, colour, religion, national
origin, sex, marital status and age appear nowhere in this extract, so the model
cannot use them directly. That is not the same as fairness, and it is important
not to read it as such: in the United States, geography correlates with race, and
`addr_state` is a feature here.

**No fairness metric is reported, because none can be computed from these
columns.** Saying so is more useful than substituting a proxy and calling the
question answered. A deployment subject to ECOA needs a disparate-impact test
against attributes this dataset does not contain, on data that does.

**The explanations are exact, not indicative.** Each reason code is a SHAP value
in the model's own log-odds, and the full set sums to the score, so a reviewer
can check a decline rather than take it on trust."""

_STATIC_LIMITS = (
    "One time-ordered split and no cross-validation, so every number here "
    "carries the sampling noise of a single test window.",
    "Test was scored once. Any change made after reading these numbers makes "
    "them optimistic, and there is no second held-out window left.",
    "The calibrator was fitted on validation, so the validation calibration "
    "error is in-sample. It is the floor the test number should be read "
    "against, not a second result.",
    "Labels come from the extract's own status column at the snapshot date. A "
    "charge-off recorded after the snapshot reads as repaid.",
    "Reason codes assume feature independence (interventional SHAP) and are in "
    "uncalibrated log-odds. Calibration is monotone, so it moves the "
    "probability without reordering the reasons.",
    "A high PSI says a feature moved, not that the model got worse. Read it "
    "beside the SHAP summary: a feature the model barely uses can drift hard "
    "and change nothing.",
)


# --- small formatters ----------------------------------------------------------


def format_metric(value: Any, digits: int = 4) -> str:
    """A metric for a table cell. A missing one reads as ``-``, never ``0.0000``.

    An absent number rendered as zero is read as a catastrophic model rather than
    as missing data, and every surface this project has - the card, the CLI
    tables, the dashboard - needs the same answer, so there is one function.
    ``bool`` is excluded because ``True`` is an ``int`` and would print ``1.0000``,
    and ``nan`` is excluded because a single-class vintage legitimately has no AUC.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "-"
    return "-" if not np.isfinite(value) else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A GitHub-flavoured markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown``, which needs ``tabulate`` -
    a dependency for six lines, in a project whose install is already a jump.
    """
    lines = [list(headers), ["---"] * len(headers)]
    lines += [["" if cell is None else str(cell) for cell in row] for row in rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def _megabytes(size: Any) -> str:
    return f"{int(size) / 1e6:.1f} MB" if isinstance(size, int | float) else "-"


# --- the sanity check ----------------------------------------------------------


def sanity_warnings(payload: Mapping[str, Any]) -> list[str]:
    """Everything about this run that a reader must not miss, in one list.

    Returned as sentences with the numbers in them, so the dashboard banner, the
    model card, and the CLI all say the same thing and none of them decides what
    counts as alarming. An empty list means the run passed every check here, not
    that it is good.
    """
    rows_test = payload.get("rows_test")
    positives = payload.get("positives_test")
    warnings: list[str] = []

    if isinstance(rows_test, int) and rows_test < SANITY_MIN_TEST_ROWS:
        warnings.append(
            f"Test window holds {rows_test} rows, under {SANITY_MIN_TEST_ROWS}. "
            "Every metric below is a description of a small sample."
        )
    if isinstance(positives, int) and positives < SANITY_MIN_TEST_POSITIVES:
        warnings.append(
            f"Test window holds {positives} default(s), under "
            f"{SANITY_MIN_TEST_POSITIVES}. Ranking metrics are dominated by "
            "which individual loans defaulted."
        )
    if payload.get("psi_score_band") == "significant":
        warnings.append(
            f"Score PSI is {format_metric(payload.get('psi_score'))} against the "
            f"{PSI_SIGNIFICANT} band: the population scored in the test window is "
            "not the population the model was fitted on."
        )
    # The one check that says the calibration step did not earn its place. Both
    # numbers are on test, so this is a like-for-like comparison.
    calibrated, uncalibrated = payload.get("brier_score"), payload.get("brier_score_uncalibrated")
    if (
        isinstance(calibrated, float)
        and isinstance(uncalibrated, float)
        and calibrated > uncalibrated
    ):
        warnings.append(
            f"Calibration made the Brier score worse ({format_metric(uncalibrated)} "
            f"uncalibrated to {format_metric(calibrated)} calibrated), which usually "
            "means the validation window was too small to fit a correction on."
        )
    if payload.get("include_lender_priced"):
        warnings.append(
            "This model uses lender-priced features "
            f"({', '.join(columns_in_tier(FeatureTier.LENDER_PRICED))}), so it "
            "cannot score an applicant nobody has priced yet. See ADR 0005."
        )
    return warnings


# --- the model card ------------------------------------------------------------


def render_model_card(
    payload: Mapping[str, Any],
    metadata: RunMetadata,
    *,
    vintages: pd.DataFrame | None = None,
) -> str:
    """The run's model card, as markdown. Published as ``model_card.md``.

    Takes the two artifacts a run already wrote - the ``metrics.json`` payload and
    the manifest - plus the vintage table, which is the one report whose value is
    the *series* and cannot be summarized into a scalar. Nothing is recomputed
    here; see the module docstring.
    """
    sections = [
        f"# Model card: `{metadata.run_id}`",
        "Generated by `riskscore` from `metrics.json` and `manifest.json` in this "
        "run directory.\nEvery number below was measured by the run that wrote "
        "those files. Nothing here is maintained by hand.",
        _at_a_glance(payload, metadata),
        _warnings_section(payload),
        "## Intended use\n\n" + _INTENDED_USE,
        "## Out of scope\n\n" + _OUT_OF_SCOPE,
        _data_section(payload, metadata),
        _features_section(payload, metadata),
        _metrics_section(payload),
        _vintage_section(vintages),
        _drift_section(payload),
        _explanation_section(payload),
        _limits_section(payload),
        "## Ethical considerations\n\n" + _ETHICAL,
        _provenance_section(metadata),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def _at_a_glance(payload: Mapping[str, Any], metadata: RunMetadata) -> str:
    threshold = payload.get("selected_threshold")
    return "## At a glance\n\n" + _table(
        ["Field", "Value"],
        [
            ["run id", f"`{metadata.run_id}`"],
            ["created (UTC)", metadata.created_at],
            ["model", f"`{metadata.model_type}`"],
            ["feature tier", f"`{metadata.feature_tier}`"],
            ["git commit", f"`{metadata.git_commit}`"],
            [
                "extract",
                f"`{metadata.dataset_path}` (sha256 `{metadata.dataset_sha256}`, "
                f"{_megabytes(metadata.dataset_bytes)})",
            ],
            [
                "rows train / validation / test",
                f"{payload.get('rows_train', '-')} / "
                f"{payload.get('rows_validation', '-')} / {payload.get('rows_test', '-')}",
            ],
            [
                "decision threshold",
                f"{format_metric(threshold)} "
                f"(selected on {payload.get('threshold_selected_on', '-')})",
            ],
            ["AUC (test)", format_metric(payload.get("auc_roc"))],
            ["Brier (test, calibrated)", format_metric(payload.get("brier_score"))],
            ["calibration error (test)", format_metric(payload.get("expected_calibration_error"))],
            ["approval rate (test)", format_metric(payload.get("approval_rate"))],
        ],
    )


def _warnings_section(payload: Mapping[str, Any]) -> str:
    """Directly under the headline numbers, or not at all.

    Placement is the point: a caveat below the limitations section is a caveat
    nobody read before quoting the AUC.
    """
    warnings = sanity_warnings(payload)
    if not warnings:
        return ""
    return "## Read this first\n\n" + "\n".join(f"- **{line}**" for line in warnings)


def _data_section(payload: Mapping[str, Any], metadata: RunMetadata) -> str:
    embargo = metadata.embargo or {}
    before = embargo.get("default_rate_by_vintage_before", {})
    after = embargo.get("default_rate_by_vintage_after", {})
    vintage_rows = [
        [year, format_metric(before.get(year), 3), format_metric(after.get(year), 3)]
        for year in sorted(set(before) | set(after))
    ]
    return "\n\n".join(
        [
            "## Training data",
            f"**Target.** {metadata.target_definition}",
            "**Rows at each stage.** A run that discarded 40% of its input must "
            "not look like one that discarded none.\n\n"
            + _table(
                ["Stage", "Rows"],
                [[stage, count] for stage, count in metadata.rows.items()],
            ),
            "**Time partitions.** The split is by origination date, and only the "
            "training partition was fitted on.\n\n"
            + _table(
                ["Partition", "Window"],
                [[name, window] for name, window in metadata.split_windows.items()],
            ),
            "**Outcome-maturity embargo.** Filtering to closed loans alone keeps a "
            "young loan only when it *defaulted*, so the measured default rate "
            "climbs with vintage for a reason that is arithmetic rather than "
            "credit. Loans whose full term had not elapsed by the snapshot are "
            f"removed: {embargo.get('summary', '-')}\n\n"
            + (
                _table(["Vintage", "Default rate before", "After"], vintage_rows)
                if vintage_rows
                else "_No per-vintage counts recorded._"
            ),
            f"**Term filter.** {payload.get('term_months_in', '-')} months. "
            "Downstream of the embargo rather than an independent choice: "
            "60-month loans survive the embargo only in the earliest vintages, so "
            "admitting them trains on a term mix that validation never sees.",
        ]
    )


def _features_section(payload: Mapping[str, Any], metadata: RunMetadata) -> str:
    features = metadata.features or {}
    numeric = list(features.get("numeric", ()))
    categorical = list(features.get("categorical", ()))
    tier_note = (
        "This run **admits** the lender-priced tier "
        f"({', '.join(columns_in_tier(FeatureTier.LENDER_PRICED))}). Those columns "
        "are the lender's own price for the loan, so they are excellent predictors "
        "and unavailable for an applicant nobody has priced. The model is "
        "therefore a benchmark, not a screening model."
        if payload.get("include_lender_priced")
        else "This run **excludes** the lender-priced tier "
        f"({', '.join(columns_in_tier(FeatureTier.LENDER_PRICED))}), which is the "
        "default. Those columns encode the lender's own risk assessment, so a "
        "model using them cannot score an applicant who has not been priced yet. "
        "See ADR 0005; `comparison.json` quantifies what excluding them costs."
    )
    return "\n\n".join(
        [
            "## Features",
            tier_note,
            f"`{features.get('summary', payload.get('features', '-'))}`",
            _table(
                ["Feature", "Kind", "Meaning"],
                [
                    [f"`{name}`", kind, feature_label(name)]
                    for kind, names in (("numeric", numeric), ("categorical", categorical))
                    for name in names
                ],
            ),
            f"**Leakage audit.** {features.get('leakage', payload.get('leakage', '-'))}. "
            "The audit is a record, not a filter: what keeps a post-origination "
            "column out of the model is that the feature spec never contains one.",
        ]
    )


def _metrics_section(payload: Mapping[str, Any]) -> str:
    false_negative = payload.get("false_negative_cost")
    false_positive = payload.get("false_positive_cost")
    ratio = (
        f"{false_negative / false_positive:.1f}:1"
        if isinstance(false_negative, int | float)
        and isinstance(false_positive, int | float)
        and false_positive
        else "-"
    )
    return "\n\n".join(
        [
            "## Metrics",
            "Measured on the test window, scored once, with the threshold and the "
            "calibrator frozen beforehand.",
            _table(
                ["Metric", "Value", "What it is"],
                [
                    ["AUC ROC", format_metric(payload.get("auc_roc")), "ranking, all thresholds"],
                    [
                        "average precision",
                        format_metric(payload.get("average_precision")),
                        "ranking, weighted towards the defaults",
                    ],
                    [
                        "KS statistic",
                        format_metric(payload.get("ks_statistic")),
                        "largest separation between the two score distributions",
                    ],
                    [
                        "Brier score",
                        format_metric(payload.get("brier_score")),
                        "calibrated probability accuracy, lower is better",
                    ],
                    [
                        "Brier score (uncalibrated)",
                        format_metric(payload.get("brier_score_uncalibrated")),
                        "the same model before the correction",
                    ],
                    [
                        "calibration error",
                        format_metric(payload.get("expected_calibration_error")),
                        "mean gap between stated and observed default rate",
                    ],
                    [
                        "calibration error (validation, in-sample)",
                        format_metric(
                            payload.get("expected_calibration_error_validation_in_sample")
                        ),
                        "the floor, not a result: fitted on these rows",
                    ],
                    [
                        "default rate",
                        format_metric(payload.get("default_rate")),
                        "the base rate of the test window",
                    ],
                    [
                        "approval rate",
                        format_metric(payload.get("approval_rate")),
                        "share approved at the threshold",
                    ],
                ],
            ),
            f"**Calibration.** `{payload.get('calibration_method', '-')}`, fitted on "
            f"{payload.get('calibration_fitted_on', '-')} and applied to every "
            "score this bundle produces. Measured *and* applied: a reported "
            "calibration curve whose correction is thrown away describes a model "
            "nobody would ship.",
            f"**Decision rule.** Approve when the calibrated probability is below "
            f"{format_metric(payload.get('selected_threshold'))}. Chosen on "
            f"{payload.get('threshold_selected_on', '-')} by minimizing expected "
            f"cost with a funded default costed at {false_negative} against "
            f"{false_positive} for a declined good loan, a ratio of {ratio}. "
            f"Validation cost at the selected threshold: "
            f"{format_metric(payload.get('selected_threshold_total_cost'), 1)}.",
        ]
    )


def _vintage_section(vintages: pd.DataFrame | None) -> str:
    """The per-vintage table, verbatim.

    Not summarized into a scalar, because the finding is the shape of the series:
    a default rate flat across vintages is the embargo working, and an AUC that
    falls in the last year is the thing a single headline number hides.
    """
    if vintages is None or vintages.empty:
        return ""
    columns = [
        column
        for column in (
            "partition",
            "vintage",
            "rows",
            "defaults",
            "default_rate",
            "auc_roc",
            "brier_score",
            "approval_rate",
        )
        if column in vintages.columns
    ]
    rows = [
        [
            value if isinstance(value, str | int | np.integer) else format_metric(value, 3)
            for value in record
        ]
        for record in vintages[columns].itertuples(index=False, name=None)
    ]
    return (
        "## Performance by origination year\n\n"
        "One AUC over a multi-year window hides whether the model works in every "
        "year of it.\n\n" + _table(columns, rows)
    )


def _drift_section(payload: Mapping[str, Any]) -> str:
    unstable = list(payload.get("psi_features_unstable", ()))
    return "\n\n".join(
        [
            "## Population stability",
            f"Population Stability Index, {payload.get('drift_reference', '-')} "
            f"against {payload.get('drift_comparison', '-')}. Bands are the "
            f"conventional {PSI_MODERATE} and {PSI_SIGNIFICANT}. PSI needs no "
            "labels, which is why it is the number to watch between now and the "
            "next outcome.",
            _table(
                ["Measure", "PSI", "Band"],
                [
                    [
                        "score",
                        format_metric(payload.get("psi_score")),
                        payload.get("psi_score_band", "-"),
                    ],
                    [
                        f"worst feature (`{payload.get('psi_feature_worst', '-')}`)",
                        format_metric(payload.get("psi_feature_worst_value")),
                        "see `psi_features.csv`",
                    ],
                ],
            ),
            (
                "Features outside the stable band: " + ", ".join(f"`{name}`" for name in unstable)
                if unstable
                else "No feature is outside the stable band."
            ),
        ]
    )


def _explanation_section(payload: Mapping[str, Any]) -> str:
    top = list(payload.get("top_features", ()))
    if not top:
        return ""
    return "\n\n".join(
        [
            "## What the model uses",
            f"Ranked by mean absolute SHAP contribution on the test window, "
            f"computed by the `{payload.get('explainer', '-')}` path. Full table in "
            "`shap_summary.csv`; per-applicant reason codes come from the same "
            "values, so a global ranking and an individual explanation cannot "
            "disagree.",
            _table(
                ["Rank", "Feature", "Meaning"],
                [
                    [index, f"`{name}`", feature_label(name)]
                    for index, name in enumerate(top, start=1)
                ],
            ),
        ]
    )


def _limits_section(payload: Mapping[str, Any]) -> str:
    limits = list(_STATIC_LIMITS)
    rows_test = payload.get("rows_test")
    if isinstance(rows_test, int):
        limits.append(
            f"The test window is {rows_test} rows. Read every metric above as a "
            "statement about that many loans."
        )
    return "## Limitations\n\n" + "\n".join(f"- {line}" for line in limits)


def _provenance_section(metadata: RunMetadata) -> str:
    """Library versions, because the same code on two version sets is two models."""
    return "## Reproducing this run\n\n" + _table(
        ["Component", "Version"],
        [
            ["bundle schema", metadata.bundle_schema_version],
            *([name, version] for name, version in metadata.library_versions.items()),
        ],
    )


# --- comparing variants --------------------------------------------------------


def variant_label(payload: Mapping[str, Any]) -> str:
    """``logistic_regression / origination_only``. The row name in a comparison.

    Built from the two facts that distinguish variants of one split, in the same
    order a run id spells them, so a label and a run id read as the same thing.
    """
    return (
        f"{payload.get('model_type', 'unknown')} / "
        f"{feature_tier(bool(payload.get('include_lender_priced')))}"
    )


def comparison_table(payloads: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per variant, baseline first, with deltas against the baseline.

    The baseline is the **first** payload, not the best one: a comparison exists
    to answer "what does moving away from the default buy", and letting the
    winner define the baseline inverts every sign whenever the winner changes.

    Deltas are signed differences, ``variant - baseline``, in the metric's own
    units. Whether a positive delta is good depends on the metric, which is what
    :data:`COMPARISON_METRICS` records.
    """
    if not payloads:
        raise ValueError("A comparison needs at least one variant.")
    frame = pd.DataFrame(
        [
            {
                "variant": variant_label(payload),
                "run_id": payload.get("run_id"),
                "model_type": payload.get("model_type"),
                "feature_tier": feature_tier(bool(payload.get("include_lender_priced"))),
                "rows_train": payload.get("rows_train"),
                "rows_test": payload.get("rows_test"),
                **{name: payload.get(name) for name, _direction in COMPARISON_METRICS},
            }
            for payload in payloads
        ]
    )
    # Deltas only where they mean something. `approval_rate` is neutral, so a
    # delta column for it would invite reading a difference as an improvement.
    for name, direction in COMPARISON_METRICS:
        if direction != "neutral":
            frame[f"{name}_delta"] = frame[name] - frame[name].iloc[0]
    return frame


def comparison_payload(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``comparison.json``: the table, the winner, and the leakage cost.

    ``lender_priced_delta`` is the point of the exercise. For every model type
    fitted under both tiers it reports what admitting the lender's own price adds
    to the AUC, so ADR 0005's policy is defended with a measurement instead of an
    argument. It is absent when only one tier was fitted, rather than reported as
    zero.
    """
    table = comparison_table(payloads)
    ranked = table.dropna(subset=["auc_roc"]).sort_values("auc_roc", ascending=False)
    by_variant = {
        (str(row["model_type"]), str(row["feature_tier"])): row for _, row in table.iterrows()
    }

    deltas = []
    for model_type in dict.fromkeys(table["model_type"]):
        excluded = by_variant.get((str(model_type), feature_tier(False)))
        included = by_variant.get((str(model_type), feature_tier(True)))
        if excluded is None or included is None:
            continue
        deltas.append(
            {
                "model_type": str(model_type),
                "auc_roc_origination_only": _float_or_none(excluded["auc_roc"]),
                "auc_roc_with_lender_priced": _float_or_none(included["auc_roc"]),
                "auc_roc_gain": _float_or_none(included["auc_roc"] - excluded["auc_roc"]),
                "run_id_origination_only": excluded["run_id"],
                "run_id_with_lender_priced": included["run_id"],
            }
        )

    return {
        "baseline": table["variant"].iloc[0],
        # Published so a reader (or the dashboard) knows which direction is an
        # improvement without a hardcoded copy of that knowledge.
        "metrics": dict(COMPARISON_METRICS),
        "variants": [
            {key: _json_safe_value(value) for key, value in record.items()}
            for record in table.to_dict(orient="records")
        ],
        # By AUC, because it is the only metric here that is comparable across
        # variants without also agreeing on a threshold. Absent rather than
        # guessed if no variant produced one.
        "best_by_auc_roc": (
            {
                "variant": ranked["variant"].iloc[0],
                "run_id": ranked["run_id"].iloc[0],
                "auc_roc": _float_or_none(ranked["auc_roc"].iloc[0]),
            }
            if not ranked.empty
            else None
        ),
        "lender_priced_delta": deltas,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return float(value)


# --- shapes for the dashboard --------------------------------------------------


def _json_safe_value(value: Any) -> Any:
    """One cell, in a type ``json.dumps`` can write and ``JSON.parse`` can read.

    NaN is the case that matters. ``json.dumps`` emits a bare ``NaN`` token by
    default, which is not valid JSON and makes ``JSON.parse`` throw - so a single
    missing metric in a table would take out a whole dashboard panel. Every
    absent value becomes ``null``, which the formatter renders as ``-``.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    # A numpy scalar is not JSON-serializable, and unwrapping it first means the
    # finite check below sees a plain float either way.
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def columnar(frame: pd.DataFrame) -> dict[str, list[Any]]:
    """A table as one list per column, not one object per row.

    The report tables are dense and narrow: the threshold cost table is 99 rows of
    six numbers, and serving it as ``[{"threshold": ..., "total_cost": ...}, ...]``
    repeats every key 99 times, which is most of the bytes. Columnar is also the
    shape a chart wants - one array per axis - so the client does no reshaping.
    """
    return {
        str(name): [_json_safe_value(value) for value in frame[name].tolist()]
        for name in frame.columns
    }
