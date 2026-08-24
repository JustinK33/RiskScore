"""Metrics, the cost-sensitive threshold search, and the guard around it.

Three of this project's four headline numbers were computed here, and two of them
were wrong in ways that made the model look better than it was.

**KS was direction-blind.** The committed artifact reported ``auc_roc: 0.0696``
beside ``ks_statistic: 0.9304``. An AUC far below 0.5 next to a near-perfect KS
is the signature of inverted labels: the model separates the classes almost
completely and ranks them backwards. Because the old implementation took
``.abs()`` of the gap between the two cumulative curves, backwards separation
scored as well as correct separation, and the one number that could have
contradicted the AUC agreed with it instead (audit B06). KS here is now signed
in the direction a credit score is supposed to run - higher score, more defaults
- so an inverted model scores near zero and the two metrics can no longer tell
opposite stories.

**KS was also tie-blind.** Ranking rows and cumulating one at a time splits a
block of equal scores in whatever order the sort produced, so the running gap can
peak *inside* a tie - at a cut-off no threshold can actually implement. With a
tree model emitting a few hundred distinct probabilities over a million rows,
those blocks are large. Ties are now aggregated to their last row before the
maximum is taken, so the statistic is the largest gap achievable at a real
threshold (audit B07).

**The threshold search minimized without constraint.** It also broke ties toward
the *lowest* threshold, so where several cut-offs cost the same it chose the one
that declined the most applicants - and at a 5:1 false-negative ratio the true
minimum is often "decline everybody", which costs nothing in missed defaults
(audit B10). Selection now refuses thresholds below a minimum approval rate and
breaks ties toward the higher threshold.

**And it was quadratic in the grid.** 99 candidate thresholds each ran a full
``confusion_matrix`` over every row: 99 passes over the array to answer 99
questions that are all cumulative sums of one sorted pass (audit P03). The table
is now built from one argsort, one cumsum, and one ``searchsorted``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.metrics import precision_recall_curve as sklearn_precision_recall_curve

#: The default candidate cut-offs: every whole percentage point from 1% to 99%.
#: Two decimal places because a threshold is a published lending policy, and
#: 0.14 is a policy while 0.1372549 is a fitting artifact.
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(
    round(float(value), 2) for value in np.arange(0.01, 1.0, 0.01)
)

#: A selected threshold must approve at least this share of applicants.
#:
#: Not a safety margin - a definition. With a 5:1 false-negative cost and a 15%
#: default rate, declining every applicant costs 0.85 units per row while
#: approving every applicant costs 0.75, so the unconstrained minimum can sit at
#: the very bottom of the grid, and "decline everyone" is the one operating point
#: guaranteed to be useless. One in five is the loosest rate at which a portfolio
#: still exists; a run that cannot meet it is telling you the cost ratio is wrong,
#: which is why falling below it raises rather than silently returning the corner.
DEFAULT_MIN_APPROVAL_RATE = 0.20


@dataclass(frozen=True)
class ClassificationMetrics:
    """Core binary classification metrics for credit default risk models."""

    auc_roc: float
    average_precision: float
    ks_statistic: float
    brier_score: float
    default_rate: float
    approval_rate: float


@dataclass(frozen=True, slots=True)
class CostMatrix:
    """What each kind of mistake costs, in whatever unit the caller chooses.

    A false negative is a default that was approved; a false positive is a good
    borrower who was declined. Only the *ratio* affects the selected threshold,
    so the units are free - 5.0 and 1.0 says "a missed default hurts five times
    as much as a lost customer".
    """

    false_negative_cost: float
    false_positive_cost: float

    def __post_init__(self) -> None:
        for name, cost in (
            ("false_negative_cost", self.false_negative_cost),
            ("false_positive_cost", self.false_positive_cost),
        ):
            if not np.isfinite(cost) or cost < 0:
                raise ValueError(f"`{name}` must be a finite, non-negative number; got {cost!r}.")
        # Both zero makes every threshold cost exactly the same, and the search
        # would return whichever the tie-break preferred - a decision with no
        # input, reported as if it had one.
        if self.false_negative_cost == 0 and self.false_positive_cost == 0:
            raise ValueError("At least one of the two costs must be non-zero.")


@dataclass(frozen=True, slots=True)
class ValidationScores:
    """Scores from the validation partition, and the type that says so.

    Every fitted decision in this project - the calibrator, the threshold - is
    fitted on validation, and the reason is in
    ``docs/decisions/0003-train-validation-test-split.md``. This wrapper is how
    that rule stops being a convention.

    A convention that lives in a docstring survives until the first refactor that
    passes ``y_test`` to a function expecting ``y_true``, at which point nothing
    raises and every metric simply improves. Requiring this type turns that into
    a ``TypeError`` at the call site.

    The check is at runtime, not only in the annotation, and deliberately so:
    pandas ships no type stubs here, so a bare ``pd.Series`` is ``Any`` to mypy
    and would pass static checking untouched. Annotations catch the mistake for
    typed callers; the ``isinstance`` check catches it for everyone.

    Wrapping test scores in this is of course still possible. It is now a
    deliberate, greppable sentence rather than an argument in the wrong position.
    """

    y_true: pd.Series
    y_score: pd.Series

    def __post_init__(self) -> None:
        if len(self.y_true) != len(self.y_score):
            raise ValueError(
                f"Labels and scores must be the same length; "
                f"got {len(self.y_true)} and {len(self.y_score)}."
            )
        if not self.y_true.index.equals(self.y_score.index):
            # Both come from the same partition, so a mismatched index means one
            # of them was reindexed or re-sorted on the way here, and pandas would
            # align them silently into a frame of NaN.
            raise ValueError("Labels and scores must share an index.")


def as_metric_arrays(
    y_true: pd.Series, y_score: pd.Series
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Validate one label/score pair and hand back plain numpy arrays.

    Public, and shared with :mod:`risk_score.calibration`, because every number
    this project reports about a partition should have been through the same door.

    Every metric in this module starts here, so the guards are stated once. They
    all describe conditions under which a metric would still return a number:
    an empty partition gives NaN, a NaN score sorts unpredictably, and a label
    column of ``{0, 1, 2}`` silently redefines what a positive is.
    """
    labels = np.asarray(pd.Series(y_true).to_numpy(), dtype=np.int64)
    scores = np.asarray(pd.Series(y_score).to_numpy(), dtype=np.float64)

    if labels.size != scores.size:
        raise ValueError(
            f"Labels and scores must be the same length; got {labels.size} and {scores.size}."
        )
    if labels.size == 0:
        raise ValueError("Cannot evaluate an empty partition.")
    unexpected = sorted(int(value) for value in set(np.unique(labels)) - {0, 1})
    if unexpected:
        raise ValueError(f"Labels must be 0 or 1; found {unexpected}.")
    if not np.isfinite(scores).all():
        raise ValueError(
            f"Scores must all be finite; found {int((~np.isfinite(scores)).sum())} "
            f"NaN or infinite value(s). A NaN score has no place in a ranking."
        )
    return labels, scores


def compute_auc_roc(y_true: pd.Series, y_score: pd.Series) -> float:
    """Area under the ROC curve: the probability a random default outranks a random non-default."""
    labels, scores = as_metric_arrays(y_true, y_score)
    _require_both_classes(labels, "AUC-ROC")
    return float(roc_auc_score(labels, scores))


def compute_average_precision(y_true: pd.Series, y_score: pd.Series) -> float:
    """Area under the precision-recall curve.

    Computed directly rather than by integrating the curve from
    :func:`compute_precision_recall`. The old code built the whole curve to read
    one scalar out of a column repeated on every row of it, which is both slower
    and an invitation to read row 7 by mistake.
    """
    labels, scores = as_metric_arrays(y_true, y_score)
    _require_both_classes(labels, "average precision")
    return float(average_precision_score(labels, scores))


def compute_brier_score(y_true: pd.Series, y_score: pd.Series) -> float:
    """Mean squared error of the predicted probabilities.

    Unlike AUC this one is not rank-invariant: it is the number that notices a
    model whose ordering is fine and whose probabilities are inflated, which is
    exactly what ``class_weight="balanced"`` used to produce (audit B05).
    """
    labels, scores = as_metric_arrays(y_true, y_score)
    return float(brier_score_loss(labels, scores))


def compute_precision_recall(y_true: pd.Series, y_score: pd.Series) -> pd.DataFrame:
    """The precision-recall curve, one row per distinct operating point.

    sklearn returns one more precision/recall pair than thresholds - the final
    point is recall 0, precision 1, which no threshold produces - so the
    threshold column ends in NaN rather than being silently truncated to match.
    """
    labels, scores = as_metric_arrays(y_true, y_score)
    _require_both_classes(labels, "a precision-recall curve")
    precision, recall, thresholds = sklearn_precision_recall_curve(labels, scores)
    return pd.DataFrame(
        {
            "threshold": np.append(thresholds, np.nan),
            "precision": precision,
            "recall": recall,
        }
    )


def _require_both_classes(labels: npt.NDArray[np.int64], metric: str) -> None:
    """Refuse a metric that is undefined - not merely uninformative - on one class."""
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        raise ValueError(
            f"{metric} requires both classes; got {positives} positive(s) "
            f"out of {labels.size} rows."
        )


def compute_ks_statistic(y_true: pd.Series, y_score: pd.Series) -> float:
    """Kolmogorov-Smirnov separation, signed in the scoring direction.

    The largest gap, at any achievable threshold, between the share of defaults
    caught and the share of good loans wrongly caught with them. 0.35 means some
    cut-off declines 35 percentage points more of the defaults than of the
    non-defaults.

    Two departures from the textbook two-sample KS, both deliberate:

    * **Signed, not absolute.** The maximum is taken over
      ``F_default - F_non_default`` rather than its absolute value. A model that
      separates the classes perfectly *backwards* has a textbook KS near 1 and is
      worthless; here it scores near 0, in agreement with its AUC of near 0. The
      committed artifact's 0.070 AUC beside a 0.930 KS is the failure this closes
      (audit B06). The two definitions coincide for any model ranked the right way
      round, which is the only case where the number means anything anyway.
    * **Ties collapsed.** The gap is evaluated only at the last row of each block
      of equal scores, because a threshold cannot cut a tie in half. Cumulating
      row by row lets the maximum land inside a tied block, overstating separation
      by an amount that grows with the size of the blocks - and a boosted model
      over a million rows emits large ones (audit B07).

    Never returns a negative number: the two cumulative curves both end at 1, so
    their difference is 0 at the final threshold and the maximum is at least that.
    """
    labels, scores = as_metric_arrays(y_true, y_score)
    _require_both_classes(labels, "The KS statistic")

    # Descending, because a credit score's operating points run from "decline
    # only the very worst" downward, and the statistic should be read that way.
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    positives = int(sorted_labels.sum())
    negatives = sorted_labels.size - positives
    caught_defaults = np.cumsum(sorted_labels) / positives
    caught_non_defaults = np.cumsum(1 - sorted_labels) / negatives

    # The last index of every run of equal scores: the only places a threshold
    # can actually be put.
    boundaries = np.flatnonzero(np.diff(sorted_scores) != 0)
    achievable = np.append(boundaries, sorted_scores.size - 1)

    return float(np.max(caught_defaults[achievable] - caught_non_defaults[achievable]))


def compute_threshold_cost_table(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    cost_matrix: CostMatrix,
    thresholds: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Confusion counts, approval rate, and total cost at every candidate threshold.

    An applicant is **declined** when their score is at or above the threshold,
    so the approved population is ``score < threshold``. That convention is
    stated because every count below depends on it and the opposite convention
    produces a table that looks equally plausible.

    One sorted pass answers the whole grid. Sorting the scores once and taking a
    cumulative sum of the labels gives, for any threshold, the number of
    positives below it from a single ``searchsorted`` - so this is O(n log n + k)
    rather than the O(n * k) of one ``confusion_matrix`` call per candidate
    (audit P03). On 400k validation rows and 99 thresholds that is one pass
    instead of ninety-nine.
    """
    labels, scores = as_metric_arrays(y_true, y_score)
    grid = np.asarray(
        DEFAULT_THRESHOLD_GRID if thresholds is None else thresholds, dtype=np.float64
    )
    if grid.size == 0:
        raise ValueError("At least one threshold is required.")
    if not np.isfinite(grid).all():
        raise ValueError("Thresholds must all be finite.")

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # `cumulative_positives[k]` is the number of defaults among the k lowest
    # scores. The leading zero is what makes `k == 0` - a threshold below every
    # score, approving nobody - an ordinary lookup rather than a special case.
    cumulative_positives = np.concatenate(([0], np.cumsum(sorted_labels)))
    total = labels.size
    total_positives = int(cumulative_positives[-1])
    total_negatives = total - total_positives

    # side="left": the first index whose score is >= the threshold, which is also
    # the count of rows strictly below it, which is the count of approvals.
    approvals = np.searchsorted(sorted_scores, grid, side="left")
    # Approved and defaulted: the mistake the cost matrix weights most heavily.
    false_negatives = cumulative_positives[approvals]
    true_positives = total_positives - false_negatives
    true_negatives = approvals - false_negatives
    false_positives = total_negatives - true_negatives

    return pd.DataFrame(
        {
            "threshold": grid,
            "true_positives": true_positives.astype(np.int64),
            "false_positives": false_positives.astype(np.int64),
            "true_negatives": true_negatives.astype(np.int64),
            "false_negatives": false_negatives.astype(np.int64),
            "predicted_default_rate": (total - approvals) / total,
            "approval_rate": approvals / total,
            "total_cost": false_negatives * cost_matrix.false_negative_cost
            + false_positives * cost_matrix.false_positive_cost,
        }
    )


def select_threshold_by_cost(
    scores: ValidationScores,
    *,
    cost_matrix: CostMatrix,
    thresholds: Sequence[float] | None = None,
    min_approval_rate: float = DEFAULT_MIN_APPROVAL_RATE,
) -> float:
    """Choose the cheapest threshold that still approves somebody.

    Takes a :class:`ValidationScores`, not two Series, because selecting a
    threshold is *fitting* and the partition it is fitted on is part of the
    result's meaning. Choosing it on test and then reporting test's cost reports
    the minimum of the grid, not a performance (audit B04).

    Two corrections to the plain minimum:

    * Candidates approving less than ``min_approval_rate`` of applicants are
      excluded before the search, because the unconstrained minimum is frequently
      the bottom of the grid - decline everyone, miss no defaults (audit B10).
    * Ties are broken toward the **higher** threshold. Equal cost means equal
      confusion counts, and between two rules with identical outcomes the one
      that declines fewer applicants is the one to publish. The old code sorted
      thresholds ascending and so preferred the stricter rule for no reason.
    """
    if not isinstance(scores, ValidationScores):
        raise TypeError(
            "select_threshold_by_cost takes a ValidationScores, not bare arrays. "
            "The threshold is a fitted decision; wrap the validation partition "
            "explicitly so that fitting it on test cannot happen by accident."
        )
    if not 0.0 <= min_approval_rate < 1.0:
        raise ValueError(f"`min_approval_rate` must be in [0, 1); got {min_approval_rate!r}.")

    table = compute_threshold_cost_table(
        scores.y_true,
        scores.y_score,
        cost_matrix=cost_matrix,
        thresholds=thresholds,
    )
    eligible = table[table["approval_rate"] >= min_approval_rate]
    if eligible.empty:
        raise ValueError(
            f"No candidate threshold approves at least {min_approval_rate:.0%} of "
            f"applicants; the most permissive candidate approves "
            f"{table['approval_rate'].max():.1%}. Either the score distribution is "
            f"degenerate or the threshold grid does not reach far enough."
        )

    cheapest = eligible["total_cost"].min()
    # Highest threshold among the equally cheap: the most permissive rule with
    # that cost. `.max()` on the filtered frame, rather than a sort, so the
    # tie-break is the visible thing rather than an argument to `ascending=`.
    return float(eligible.loc[eligible["total_cost"] == cheapest, "threshold"].max())
