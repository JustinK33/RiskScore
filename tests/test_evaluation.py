"""Tests for the metrics, the threshold search, and the partition guard.

The old suite asserted ``0 <= result <= 1`` on the KS statistic. That is why the
inverted-label bug survived: 0.9304 is between 0 and 1, and so is the 0.35 it
should have been. Every assertion here is against a hand-computed constant, a
scipy or sklearn oracle, or a brute-force implementation of the same definition -
so a test that passes is evidence about the number, not about its type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix

from risk_score.evaluation import (
    DEFAULT_MIN_APPROVAL_RATE,
    ClassificationMetrics,
    CostMatrix,
    ValidationScores,
    compute_auc_roc,
    compute_average_precision,
    compute_brier_score,
    compute_ks_statistic,
    compute_precision_recall,
    compute_threshold_cost_table,
    select_threshold_by_cost,
)

COSTS = CostMatrix(false_negative_cost=5.0, false_positive_cost=1.0)


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def scored(labels: list[int], scores: list[float]) -> ValidationScores:
    """A validation partition with a shared index, the way the pipeline builds one."""
    index = pd.RangeIndex(len(labels))
    return ValidationScores(
        y_true=pd.Series(labels, index=index, dtype=int),
        y_score=pd.Series(scores, index=index, dtype=float),
    )


def ranked(rng: np.random.Generator, n: int = 400) -> tuple[pd.Series, pd.Series]:
    """A realistically noisy, correctly ordered score and its labels."""
    scores = rng.beta(2.0, 8.0, size=n)
    labels = rng.binomial(1, scores)
    if labels.sum() in {0, n}:  # pragma: no cover - vanishingly unlikely at n=400
        labels[0], labels[-1] = 0, 1
    return pd.Series(labels), pd.Series(scores)


# --- shared input guards -------------------------------------------------------


def test_an_empty_partition_is_refused_rather_than_returning_nan() -> None:
    with pytest.raises(ValueError, match="empty partition"):
        compute_auc_roc(series([]), series([]))


def test_a_nan_score_is_refused_because_it_has_no_place_in_a_ranking() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_ks_statistic(pd.Series([0, 1, 1]), series([0.1, np.nan, 0.9]))


def test_a_label_outside_zero_and_one_is_named() -> None:
    with pytest.raises(ValueError, match=r"found \[2\]"):
        compute_auc_roc(pd.Series([0, 1, 2]), series([0.1, 0.5, 0.9]))


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_brier_score(pd.Series([0, 1]), series([0.1, 0.5, 0.9]))


def test_a_single_class_partition_is_refused_by_the_metrics_that_need_two() -> None:
    with pytest.raises(ValueError, match="requires both classes; got 0 positive"):
        compute_auc_roc(pd.Series([0, 0, 0]), series([0.1, 0.5, 0.9]))
    with pytest.raises(ValueError, match="requires both classes; got 3 positive"):
        compute_ks_statistic(pd.Series([1, 1, 1]), series([0.1, 0.5, 0.9]))


def test_the_brier_score_still_works_on_one_class() -> None:
    """It is a squared error, not a ranking statistic, so one class is fine -
    and it is the only number a wholly-defaulted vintage can still report."""
    assert compute_brier_score(pd.Series([1, 1]), series([0.5, 1.0])) == pytest.approx(0.125)


# --- AUC, average precision, Brier ---------------------------------------------


def test_a_perfectly_ranked_score_has_auc_one() -> None:
    assert compute_auc_roc(pd.Series([0, 0, 1, 1]), series([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_a_perfectly_inverted_score_has_auc_zero() -> None:
    assert compute_auc_roc(pd.Series([1, 1, 0, 0]), series([0.1, 0.2, 0.8, 0.9])) == 0.0


def test_average_precision_matches_the_hand_computed_value() -> None:
    """Ranked 0.9(+), 0.8(-), 0.4(+), 0.1(-): precision at each recall step is
    1/1 and 2/3, and average precision averages them over the two positives."""
    result = compute_average_precision(pd.Series([0, 1, 0, 1]), series([0.1, 0.4, 0.8, 0.9]))

    assert result == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_the_brier_score_is_the_mean_squared_error() -> None:
    result = compute_brier_score(pd.Series([0, 0, 1, 1]), series([0.1, 0.2, 0.8, 0.9]))

    assert result == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)


def test_the_precision_recall_curve_ends_at_recall_zero_with_no_threshold() -> None:
    curve = compute_precision_recall(pd.Series([0, 1, 0, 1]), series([0.1, 0.4, 0.8, 0.9]))

    assert list(curve.columns) == ["threshold", "precision", "recall"]
    assert curve["recall"].iloc[-1] == 0.0
    assert curve["precision"].iloc[-1] == 1.0
    # The extra sklearn point is not implementable, so its threshold is NaN
    # rather than a fabricated value.
    assert np.isnan(curve["threshold"].iloc[-1])


def test_average_precision_does_not_come_from_the_curve() -> None:
    """The scalar used to be a column repeated on every row of the curve, which
    invited reading row 7. It is now its own function, and the curve has no
    such column at all."""
    y_true, y_score = ranked(np.random.default_rng(11))

    assert "average_precision" not in compute_precision_recall(y_true, y_score).columns


# --- KS ------------------------------------------------------------------------


def test_b06_a_perfectly_inverted_score_has_ks_near_zero_not_near_one() -> None:
    """The whole bug in one test. The committed artifact reported AUC 0.070 with
    KS 0.930; a textbook two-sample KS agrees with the 0.930, which is why this
    project's KS is signed in the scoring direction instead."""
    labels = pd.Series([1, 1, 1, 0, 0, 0])
    inverted = series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    assert compute_auc_roc(labels, inverted) == 0.0
    assert compute_ks_statistic(labels, inverted) == 0.0
    # The two-sample statistic scipy computes is 1.0 on the same input: it
    # measures separation, and separation is not the same claim as skill.
    assert ks_2samp(inverted[labels == 1], inverted[labels == 0]).statistic == 1.0


def test_ks_is_one_for_a_perfectly_separated_and_correctly_ordered_score() -> None:
    labels = pd.Series([0, 0, 0, 1, 1, 1])
    scores = series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    assert compute_ks_statistic(labels, scores) == 1.0


def test_ks_is_the_largest_gap_between_the_two_caught_shares() -> None:
    """Descending by score: +, -, +, -. After the first row 1/2 of defaults and
    0/2 of the good loans are declined, a gap of 0.5, and nothing later beats it."""
    labels = pd.Series([1, 0, 1, 0])
    scores = series([0.9, 0.8, 0.4, 0.1])

    assert compute_ks_statistic(labels, scores) == pytest.approx(0.5)


def test_b07_a_tied_block_cannot_be_cut_in_half() -> None:
    """Four rows all scored 0.5, one default among them, and one clean default
    above. No threshold can separate the tied rows, so the only achievable
    gaps are at 0.9 and at 0.5 - giving 0.5 and 0.0. Cumulating row by row
    would have found 0.5 + 0.5 - 0 = 1.0 in the middle of the tie, and reported
    perfect separation for a score with almost none."""
    labels = pd.Series([1, 1, 0, 0, 0])
    scores = series([0.9, 0.5, 0.5, 0.5, 0.5])

    assert compute_ks_statistic(labels, scores) == pytest.approx(0.5)


def test_ks_equals_the_scipy_two_sample_statistic_when_the_score_is_ranked_correctly() -> None:
    """The signed definition coincides with the textbook one for any model
    ordered the right way round - which is the only case where either number
    means anything."""
    y_true, y_score = ranked(np.random.default_rng(7))
    oracle = ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic

    assert compute_ks_statistic(y_true, y_score) == pytest.approx(oracle, abs=1e-12)


def test_ks_never_returns_a_negative_number() -> None:
    """Both cumulative curves end at 1, so the final gap is 0 and the maximum is
    at least that. A worthless score reports 0, not -0.4."""
    rng = np.random.default_rng(3)
    scores = pd.Series(rng.random(200))
    labels = pd.Series(rng.binomial(1, 0.2, size=200))

    assert compute_ks_statistic(labels, scores) >= 0.0


# --- the threshold cost table --------------------------------------------------


def test_the_cost_table_matches_a_confusion_matrix_at_every_threshold() -> None:
    """The vectorized table against the 99-call loop it replaced (audit P03).
    Same definition, computed two different ways, on noisy data with ties."""
    rng = np.random.default_rng(19)
    y_true, y_score = ranked(rng, n=300)
    y_score = y_score.round(2)  # deliberate ties, which is where indexing goes wrong

    table = compute_threshold_cost_table(y_true, y_score, cost_matrix=COSTS)

    for row in table.itertuples():
        declined = (y_score >= row.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, declined, labels=[0, 1]).ravel()
        assert (
            row.true_positives,
            row.false_positives,
            row.true_negatives,
            row.false_negatives,
        ) == (tp, fp, tn, fn)
        assert row.total_cost == pytest.approx(fn * 5.0 + fp * 1.0)
        assert row.approval_rate == pytest.approx((tn + fn) / len(y_true))


def test_an_applicant_at_exactly_the_threshold_is_declined() -> None:
    """The convention every count in the table depends on, asserted once."""
    table = compute_threshold_cost_table(
        pd.Series([0, 1]), series([0.5, 0.5]), cost_matrix=COSTS, thresholds=[0.5]
    )

    assert table["approval_rate"].iloc[0] == 0.0
    assert table["true_positives"].iloc[0] == 1
    assert table["false_positives"].iloc[0] == 1


def test_a_threshold_above_every_score_approves_everyone() -> None:
    table = compute_threshold_cost_table(
        pd.Series([0, 1, 1]), series([0.1, 0.2, 0.3]), cost_matrix=COSTS, thresholds=[0.99]
    )
    row = table.iloc[0]

    assert row["approval_rate"] == 1.0
    assert row["false_negatives"] == 2
    assert row["true_positives"] == 0
    assert row["total_cost"] == pytest.approx(10.0)


def test_the_default_grid_is_ninety_nine_whole_percentage_points() -> None:
    table = compute_threshold_cost_table(pd.Series([0, 1]), series([0.2, 0.8]), cost_matrix=COSTS)

    assert len(table) == 99
    assert table["threshold"].iloc[0] == 0.01
    assert table["threshold"].iloc[-1] == 0.99


def test_an_empty_threshold_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="At least one threshold"):
        compute_threshold_cost_table(
            pd.Series([0, 1]), series([0.2, 0.8]), cost_matrix=COSTS, thresholds=[]
        )


# --- selection -----------------------------------------------------------------


def test_selection_returns_a_threshold_of_minimum_cost() -> None:
    """Two defaults at 0.8/0.9 and two good loans at 0.1/0.2. Any cut-off in
    (0.2, 0.8] catches both defaults and no good loans, so the cheapest cost is
    zero - and the selected threshold is one of the cut-offs that achieve it."""
    scores = scored([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    selected = select_threshold_by_cost(scores, cost_matrix=COSTS)
    table = compute_threshold_cost_table(scores.y_true, scores.y_score, cost_matrix=COSTS)
    cost_at_selected = table.loc[table["threshold"] == selected, "total_cost"].iloc[0]

    assert cost_at_selected == 0.0
    assert 0.2 < selected <= 0.8


def test_b10_selection_refuses_to_decline_almost_everybody() -> None:
    """With a 5:1 cost ratio and a 40% default rate, declining everyone is the
    unconstrained minimum: 0 false negatives against 6 false positives is 6,
    while the best real rule costs more. The guard is what keeps the search from
    returning a policy no lender can run."""
    labels = [1] * 4 + [0] * 6
    # Defaults and good loans overlap heavily, so no threshold separates them.
    values = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    scores = scored(labels, values)

    unconstrained = compute_threshold_cost_table(scores.y_true, scores.y_score, cost_matrix=COSTS)
    cheapest_row = unconstrained.loc[unconstrained["total_cost"].idxmin()]
    assert cheapest_row["approval_rate"] == 0.0  # decline everyone, as advertised

    selected = select_threshold_by_cost(scores, cost_matrix=COSTS)
    approval_rate = float((scores.y_score < selected).mean())
    assert approval_rate >= DEFAULT_MIN_APPROVAL_RATE
    assert approval_rate == 1.0


def test_b10_ties_break_toward_the_higher_threshold() -> None:
    """Every threshold from 0.31 to 0.80 declines exactly the two defaults and
    nobody else, so all fifty cost the same. The most permissive of them - 0.80,
    which approves the 0.30 applicant with the most room to spare - is the one to
    publish. The old code sorted thresholds ascending and returned 0.31."""
    scores = scored([0, 0, 1, 1], [0.1, 0.3, 0.8, 0.9])

    assert select_threshold_by_cost(scores, cost_matrix=COSTS) == pytest.approx(0.80)


def test_a_higher_false_negative_cost_never_relaxes_the_threshold() -> None:
    """Monotonicity: making missed defaults more expensive can only tighten the
    rule. If this ever fails, the cost column and the counts have drifted apart."""
    rng = np.random.default_rng(23)
    y_true, y_score = ranked(rng, n=500)
    scores = ValidationScores(y_true=y_true, y_score=y_score)

    thresholds = [
        select_threshold_by_cost(
            scores, cost_matrix=CostMatrix(false_negative_cost=cost, false_positive_cost=1.0)
        )
        for cost in (1.0, 2.0, 5.0, 20.0)
    ]

    assert thresholds == sorted(thresholds, reverse=True)


def test_a_degenerate_score_distribution_raises_and_says_what_it_could_offer() -> None:
    scores = scored([0, 1], [0.999, 0.999])

    with pytest.raises(ValueError, match=r"most permissive candidate approves 0\.0%"):
        select_threshold_by_cost(scores, cost_matrix=COSTS)


def test_the_minimum_approval_rate_can_be_relaxed_deliberately() -> None:
    scores = scored([0, 1], [0.999, 0.999])

    assert select_threshold_by_cost(scores, cost_matrix=COSTS, min_approval_rate=0.0) > 0


def test_an_out_of_range_minimum_approval_rate_is_refused() -> None:
    scores = scored([0, 1], [0.2, 0.8])

    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        select_threshold_by_cost(scores, cost_matrix=COSTS, min_approval_rate=1.0)


# --- the partition guard -------------------------------------------------------


def test_b04_bare_arrays_are_refused_by_the_threshold_search() -> None:
    """A convention in a docstring survives until the first refactor. This is
    the same rule with a runtime consequence.

    The runtime check is load-bearing rather than belt-and-braces: pandas ships
    no type stubs in this project, so ``pd.Series`` is ``Any`` to mypy and the
    static check alone would let this call through.
    """
    with pytest.raises(TypeError, match="ValidationScores"):
        select_threshold_by_cost(pd.Series([0, 1]), cost_matrix=COSTS)


def test_validation_scores_refuse_a_mismatched_index() -> None:
    """pandas would align these into a frame of NaN without complaint."""
    with pytest.raises(ValueError, match="share an index"):
        ValidationScores(
            y_true=pd.Series([0, 1], index=[0, 1]),
            y_score=pd.Series([0.2, 0.8], index=[5, 6]),
        )


def test_validation_scores_refuse_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        ValidationScores(y_true=pd.Series([0, 1, 1]), y_score=series([0.2, 0.8]))


# --- the cost matrix -----------------------------------------------------------


def test_a_negative_cost_is_refused() -> None:
    with pytest.raises(ValueError, match="finite, non-negative"):
        CostMatrix(false_negative_cost=-5.0, false_positive_cost=1.0)


def test_two_zero_costs_are_refused_because_every_threshold_would_tie() -> None:
    with pytest.raises(ValueError, match="must be non-zero"):
        CostMatrix(false_negative_cost=0.0, false_positive_cost=0.0)


def test_the_metrics_container_holds_exactly_the_reported_numbers() -> None:
    """The dataclass is the contract between the pipeline and the metrics file."""
    metrics = ClassificationMetrics(
        auc_roc=0.75,
        average_precision=0.5,
        ks_statistic=0.4,
        brier_score=0.15,
        default_rate=0.2,
        approval_rate=0.8,
    )

    assert metrics.auc_roc == 0.75
    with pytest.raises(AttributeError):
        metrics.auc_roc = 0.99  # type: ignore[misc]
