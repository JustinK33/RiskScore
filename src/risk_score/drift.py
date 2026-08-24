"""Has the population moved? PSI on the score and on every feature.

A model is fitted on one population and then used on another. Everything in
``evaluation.py`` answers "how well did it rank the rows it was tested on"; this
module answers the question that decides whether that number is still worth
anything next quarter: **are the rows still the same shape**.

The Population Stability Index is the standard credit-risk answer. Bin the
reference population, count what share of each population falls in each bin, and
sum

    PSI = sum_over_bins (share_new - share_old) * ln(share_new / share_old)

which is the symmetrized Kullback-Leibler divergence between the two binned
distributions. It needs no labels, which is the entire point: a lender learns
whether the score has drifted long before it learns whether the loans defaulted.

The bands
---------
The 0.10 / 0.25 cut-offs (:data:`PSI_MODERATE`, :data:`PSI_SIGNIFICANT`) are
industry convention, not a derivation, and they are reported as *bands* rather
than as a pass/fail so nobody reads 0.099 and 0.101 as different findings. Under
0.10 is noise at these sample sizes; over 0.25 the reference population is no
longer the population being scored and the model needs refitting.

Why this project has a drift section at all
-------------------------------------------
Because it has a *known* drift, and it is a real one. With the outcome-maturity
embargo applied, 60-month loans only survive in the earliest vintages, so a split
that admitted both terms would train on 13.9% 60-month loans and validate on
0.0%. The default configuration therefore restricts to 36-month terms - and the
PSI table on ``term`` is how that restriction stays a measured decision rather
than a comment in a config file.

Two conventions worth knowing before reading a table from here
--------------------------------------------------------------
**Missingness is a bin.** A feature that goes from 2% missing to 40% missing has
drifted, however stable its observed values are, so NaN is counted in its own
bucket rather than dropped. Non-finite numbers are counted as missing too: an
infinity is not a value the reference bins can place.

**An unseen category is a bin.** A level absent from the reference population
gets the ``__unseen__`` bucket, whose reference share is zero and whose PSI
contribution is therefore large - which is the correct reading. The model's
one-hot encoder maps that level to nothing at all.

Zero-count bins would otherwise make PSI infinite. Each population's shares are
floored at half an observation, ``0.5 / n``, which is the ordinary continuity
correction and has the property that the same empty bin counts for less on a
small sample - where an empty bin is not yet surprising - than on a large one.
The floored shares are deliberately *not* renormalized, because spreading the
correction back across the bins would change the bins that had data in them.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from risk_score.evaluation import (
    as_metric_arrays,
    compute_auc_roc,
    compute_brier_score,
    compute_ks_statistic,
)
from risk_score.explain import feature_label
from risk_score.transformers import FeatureSpec

LOGGER = logging.getLogger(__name__)

#: Quantile bins cut from the reference population. Ten is the credit-risk
#: convention and it is a genuine trade-off, not a default: more bins detect a
#: narrower shift but make every bin's share noisier, and PSI is a sum over bins
#: so that noise accumulates rather than averaging out.
PSI_BUCKETS = 10

#: The two conventional band edges. Below the first, no action; above the second,
#: the reference population is not the population being scored.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

#: Bucket names that are not values. Dunder-ish on purpose: they have to be
#: impossible to confuse with a real Lending Club category such as ``other``,
#: which is an actual value of ``purpose`` and ``home_ownership``.
MISSING_BUCKET = "__missing__"
UNSEEN_BUCKET = "__unseen__"


def psi_band(psi: float) -> str:
    """``"stable"``, ``"moderate"``, or ``"significant"`` for one PSI value.

    A band rather than a boolean, because the thresholds are convention: 0.099
    and 0.101 are the same finding and a pass/fail column would present them as
    opposites.
    """
    if not np.isfinite(psi):
        raise ValueError(f"PSI must be a finite number; got {psi!r}.")
    if psi < PSI_MODERATE:
        return "stable"
    return "moderate" if psi < PSI_SIGNIFICANT else "significant"


def _shares(counts: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Bin counts as shares, with empty bins floored at half an observation."""
    total = int(counts.sum())
    return np.maximum(counts / total, 0.5 / total)


def _numeric_bin_edges(values: npt.NDArray[np.float64], buckets: int) -> npt.NDArray[np.float64]:
    """Quantile edges from the reference population, open at both ends.

    Open ends matter: a comparison row beyond the reference's observed range is
    the single most interesting thing a drift check can find, and closed edges
    would leave it in no bin at all - silently reducing the comparison's total and
    understating the drift.

    Duplicate quantiles are collapsed, which is not a rare case: a column like
    ``pub_rec`` is zero for most borrowers, so its 10th through 70th percentiles
    are all 0 and asking for ten bins yields four. The table reports how many
    bins it actually used rather than pretending to the requested resolution.
    """
    if buckets < 2:
        raise ValueError(f"`buckets` must be at least 2; got {buckets}.")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, buckets + 1)))
    if edges.size < 2:
        # A constant reference column: one bin, and any comparison value that is
        # not that constant still lands in it. PSI is then 0 by construction,
        # which is the honest answer - a constant reference cannot detect a shift.
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _as_float_array(values: pd.Series) -> npt.NDArray[np.float64]:
    """Float view of a possibly nullable numeric column.

    ``na_value`` is required: a pandas nullable ``Int64`` holding ``pd.NA`` raises
    on a plain ``to_numpy(dtype=float)`` rather than producing NaN.
    """
    return np.asarray(values.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)


def _numeric_buckets(
    reference: pd.Series, comparison: pd.Series, buckets: int
) -> tuple[list[str], list[float], list[float], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Bin both populations against the reference's quantiles, missingness last."""
    reference_values = _as_float_array(reference)
    comparison_values = _as_float_array(comparison)
    edges = _numeric_bin_edges(reference_values, buckets)

    def counted(values: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        finite = np.isfinite(values)
        placed, _ = np.histogram(values[finite], bins=edges)
        return np.append(placed, int((~finite).sum())).astype(np.int64)

    labels = [f"[{low:.4g}, {high:.4g})" for low, high in itertools.pairwise(edges)]
    return (
        [*labels, MISSING_BUCKET],
        [*edges[:-1].tolist(), float("nan")],
        [*edges[1:].tolist(), float("nan")],
        counted(reference_values),
        counted(comparison_values),
    )


def _categorical_buckets(
    reference: pd.Series, comparison: pd.Series
) -> tuple[list[str], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """One bucket per reference level, plus unseen levels, plus missingness.

    Levels are compared as strings so that a column read as ``object`` in one
    partition and as ``category`` in another - or ``1`` against ``"1"`` - does not
    register as total drift for a reason that is about dtypes rather than about
    borrowers.
    """
    reference_levels = reference.dropna().astype(str).value_counts()
    comparison_levels = comparison.dropna().astype(str).value_counts()
    levels = [str(level) for level in reference_levels.index]

    reference_counts = [int(reference_levels[level]) for level in levels]
    comparison_counts = [int(comparison_levels.get(level, 0)) for level in levels]
    # Everything the reference never showed the encoder, collapsed into one
    # bucket: individually they are each tiny, together they are the finding.
    unseen = int(comparison_levels.drop(labels=levels, errors="ignore").sum())

    return (
        [*levels, UNSEEN_BUCKET, MISSING_BUCKET],
        np.array([*reference_counts, 0, int(reference.isna().sum())], dtype=np.int64),
        np.array([*comparison_counts, unseen, int(comparison.isna().sum())], dtype=np.int64),
    )


def psi_table(
    reference: pd.Series,
    comparison: pd.Series,
    *,
    buckets: int = PSI_BUCKETS,
) -> pd.DataFrame:
    """The per-bin working for one PSI value.

    The scalar is a sum, and a sum tells you nothing about which end of the
    distribution moved. This table is what a reader looks at after seeing a PSI of
    0.31 - it is published as ``psi_score.csv`` for exactly that reason.

    Numeric columns are binned by the reference's quantiles; anything else is
    binned by its levels, so ``lower`` and ``upper`` are NaN there.
    """
    reference = pd.Series(reference)
    comparison = pd.Series(comparison)
    if reference.empty or comparison.empty:
        raise ValueError(
            f"Both populations must be non-empty; got {len(reference)} reference "
            f"and {len(comparison)} comparison rows."
        )

    numeric = pd.api.types.is_numeric_dtype(reference) and pd.api.types.is_numeric_dtype(comparison)
    if numeric:
        labels, lower, upper, reference_counts, comparison_counts = _numeric_buckets(
            reference, comparison, buckets
        )
    else:
        labels, reference_counts, comparison_counts = _categorical_buckets(reference, comparison)
        lower = upper = [float("nan")] * len(labels)

    reference_share = _shares(reference_counts)
    comparison_share = _shares(comparison_counts)
    return pd.DataFrame(
        {
            "bucket": labels,
            "lower": lower,
            "upper": upper,
            "reference_count": reference_counts,
            "comparison_count": comparison_counts,
            "reference_share": reference_share,
            "comparison_share": comparison_share,
            "psi_contribution": (comparison_share - reference_share)
            * np.log(comparison_share / reference_share),
        }
    )


def population_stability_index(
    reference: pd.Series,
    comparison: pd.Series,
    *,
    buckets: int = PSI_BUCKETS,
) -> float:
    """PSI between two populations of one variable. Never negative.

    Each bin's contribution ``(p - q) * ln(p / q)`` has the sign of ``p - q``
    twice over, so every term is non-negative and the sum is zero only when the
    two binned distributions agree exactly.
    """
    return float(psi_table(reference, comparison, buckets=buckets)["psi_contribution"].sum())


def feature_drift(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    spec: FeatureSpec,
    buckets: int = PSI_BUCKETS,
) -> pd.DataFrame:
    """PSI per model feature, worst first. Published as ``psi_features.csv``.

    Takes **engineered** frames - the output of the pipeline's ``engineer`` step -
    not the design matrix. A PSI per one-hot column would report fifty numbers
    about ``addr_state`` and none about ``addr_state``, and the whole value of this
    table is that a human can read a row of it and know what to do.

    Driven by ``spec.model_features``, so a column the model does not use cannot
    raise an alarm and a column it does use cannot be quietly left out.
    """
    rows = []
    for feature in spec.model_features:
        for frame, side in ((reference, "reference"), (comparison, "comparison")):
            if feature not in frame.columns:
                raise KeyError(
                    f"Feature {feature!r} is declared in the spec but missing from "
                    f"the {side} frame. Both frames must come from the pipeline's "
                    f"`engineer` step, whose output is the declared feature set."
                )
        table = psi_table(reference[feature], comparison[feature], buckets=buckets)
        psi = float(table["psi_contribution"].sum())
        rows.append(
            {
                "feature": feature,
                "label": feature_label(feature),
                "kind": "numeric" if feature in spec.numeric_features else "categorical",
                "psi": psi,
                "band": psi_band(psi),
                # The bin count is reported because a coarse binning is a weaker
                # test, and a reader comparing `pub_rec`'s 0.02 to `dti_clean`'s
                # 0.02 should be able to see that one had four bins and the other
                # had ten.
                "buckets": len(table),
                "missing_rate_reference": float(reference[feature].isna().mean()),
                "missing_rate_comparison": float(comparison[feature].isna().mean()),
            }
        )

    drift = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
    return drift.assign(rank=np.arange(1, len(drift) + 1))


def metrics_by_vintage(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    dates: pd.Series,
    threshold: float,
    partition: str | None = None,
) -> pd.DataFrame:
    """Discrimination and default rate per origination year.

    The single table this project most needs to publish. One headline AUC over a
    three-year test window hides whether the model works in every year of it or
    works in one year and is carried by the label trend across the rest - and the
    default rate column is where the embargo's correction becomes visible as a
    flat series rather than the climb from 15.6% to 24.3% that the uncorrected
    data shows.

    Vintages too small or single-class get NaN in the rank-based columns rather
    than a raised error: a partition boundary lands mid-year often enough that
    refusing to produce the table would mean never producing it. ``rows`` and
    ``defaults`` are always there, so a NaN is explicable rather than mysterious.
    """
    labels, scores = as_metric_arrays(y_true, y_score)
    if not pd.Series(dates).index.equals(pd.Series(y_true).index):
        # Aligning these by position would silently pair a 2013 loan's date with a
        # 2015 loan's outcome, and every number below would still look plausible.
        raise ValueError("`dates` must share an index with `y_true`.")

    vintages = pd.to_datetime(pd.Series(dates), errors="coerce").dt.year
    frame = pd.DataFrame({"label": labels, "score": scores, "vintage": vintages.to_numpy()})
    unknown = int(frame["vintage"].isna().sum())
    if unknown:
        # Should be impossible downstream of a time split, which cannot place a row
        # without a date. Dropped and counted rather than left to become a NaN
        # vintage group that reads like a real cohort.
        LOGGER.warning(
            "dropping %d row(s) with no origination date from the vintage table", unknown
        )
        frame = frame.dropna(subset=["vintage"])

    rows: list[dict[str, Any]] = []
    for vintage, group in frame.groupby(frame["vintage"].astype("int64"), sort=True):
        positives = int(group["label"].sum())
        both_classes = 0 < positives < len(group)
        rows.append(
            {
                **({"partition": partition} if partition is not None else {}),
                "vintage": int(vintage),
                "rows": len(group),
                "defaults": positives,
                "default_rate": positives / len(group),
                "auc_roc": (
                    compute_auc_roc(group["label"], group["score"])
                    if both_classes
                    else float("nan")
                ),
                "ks_statistic": (
                    compute_ks_statistic(group["label"], group["score"])
                    if both_classes
                    else float("nan")
                ),
                # Calibration is measurable on one class, unlike a ranking metric:
                # "you said 8% and nobody defaulted" is a real, usable finding.
                "brier_score": compute_brier_score(group["label"], group["score"]),
                "approval_rate": float((group["score"] < threshold).mean()),
            }
        )
    return pd.DataFrame(rows)
