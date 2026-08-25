"""Property-based tests for the five places a hand-picked example is not enough.

Every other suite in this project asserts against a value: a metric equals a
scipy oracle on one fixed frame, a parser turns one string into one number. That
catches a wrong formula. It does not catch a formula that is right on the input
somebody thought of, which is the shape of both KS bugs in the audit - the
inverted sign (B06) and the tied-block overstatement (B07) survived a suite that
checked ``0 <= ks <= 1`` on data with no ties in it.

So the five properties here are chosen by that standard: each one is a statement
whose falsification requires an input a person would not have written down.

* ``compute_threshold_cost_table`` against brute force. The fast version is one
  argsort and two cumsums replacing 99 ``confusion_matrix`` calls (P03); the
  brute-force version is four boolean masks and is obviously correct. They must
  agree on every row of every grid, including the ties the fast version's
  ``searchsorted`` boundary handling exists for.
* ``compute_ks_statistic`` against ``scipy.stats.ks_2samp``. Both of the audit's
  KS bugs are one-sided or tie-dependent, so the oracle has to be a real
  two-sample KS with a stated direction, on data with deliberate ties.
* The parsers, round-tripped. A parser is a function from a formatting decision
  back to a number, and the property is that the number survives.
* PSI's invariants: non-negative always, exactly zero against itself, and equal
  to the sum of the table it publishes.
* The selected threshold, monotone in the false-negative cost. The one property
  that says the cost matrix is *connected* to the decision - a threshold search
  that ignored its input would pass every value-based test in the suite.

Hypothesis rather than a loop over ``random``: the shrinking is the point. A
failure arrives as the smallest frame that reproduces it, which for a threshold
bug is the difference between a two-row counterexample and a 500-row one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.stats import ks_2samp

from risk_score.drift import population_stability_index, psi_table
from risk_score.evaluation import (
    CostMatrix,
    ValidationScores,
    compute_ks_statistic,
    compute_threshold_cost_table,
    select_threshold_by_cost,
)
from risk_score.feature_engineering import (
    coerce_numeric,
    parse_employment_years,
    parse_percent,
    parse_term_months,
)

# --- strategies ---------------------------------------------------------------

#: Scores drawn from a coarse grid as often as from the continuum, because ties
#: are the interesting case and uniform floats produce none. Both KS and the cost
#: table have tie-specific code - the achievable-threshold mask and the
#: `side="left"` searchsorted - and continuous data never reaches it.
_TIED_SCORES = st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.5, 0.75, 0.9, 1.0])
_ANY_SCORE = st.one_of(
    _TIED_SCORES,
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)

#: Finite, and bounded well away from the float extremes. The functions under
#: test are documented to reject NaN and infinity, so generating them would test
#: the guard rather than the property, and `tests/test_evaluation.py` already
#: covers the guard by name.
_FINITE = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False, width=32)


@st.composite
def _labelled_scores(draw: st.DrawFn, *, min_size: int = 2) -> tuple[pd.Series, pd.Series]:
    """A label/score pair with both classes present and a shuffled index.

    ``both classes`` because every metric here requires it and says so; asserting
    a property on a partition the function refuses to evaluate would test the
    error message.

    The index is deliberately not ``0..n-1``: pandas aligns on index, and a
    property suite that only ever passes default indexes cannot catch an
    implementation that reindexes.
    """
    size = draw(st.integers(min_value=min_size, max_value=200))
    labels = draw(st.lists(st.integers(0, 1), min_size=size, max_size=size))
    assume(0 < sum(labels) < size)
    scores = draw(st.lists(_ANY_SCORE, min_size=size, max_size=size))
    index = pd.RangeIndex(start=1000, stop=1000 + size)
    return pd.Series(labels, index=index), pd.Series(scores, index=index, dtype="float64")


# --- 1. the cost table against brute force ------------------------------------


@given(
    pair=_labelled_scores(),
    grid=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=12,
    ),
    fn_cost=st.floats(min_value=0.0, max_value=100.0),
    fp_cost=st.floats(min_value=0.0, max_value=100.0),
)
@settings(max_examples=200, deadline=None)
def test_cost_table_matches_brute_force(
    pair: tuple[pd.Series, pd.Series],
    grid: list[float],
    fn_cost: float,
    fp_cost: float,
) -> None:
    """Every row of the vectorized table equals four boolean masks.

    The fast path is one argsort and two cumulative sums answering the whole grid
    from `searchsorted` (audit P03). This is the slow, obviously-correct version
    of the same definition: declined is ``score >= threshold``, approved is
    ``score < threshold``, and the four counts follow directly.

    The two implementations share no code, which is the entire value: an off-by-one
    in the `side="left"` boundary would produce a table that is internally
    consistent, sums to n, and is wrong by one row at every tie.
    """
    assume(fn_cost > 0 or fp_cost > 0)
    y_true, y_score = pair
    costs = CostMatrix(false_negative_cost=fn_cost, false_positive_cost=fp_cost)

    table = compute_threshold_cost_table(y_true, y_score, cost_matrix=costs, thresholds=grid)
    labels = y_true.to_numpy()
    scores = y_score.to_numpy()

    for row in table.itertuples(index=False):
        declined = scores >= row.threshold
        approved = ~declined
        assert row.true_positives == int((declined & (labels == 1)).sum())
        assert row.false_positives == int((declined & (labels == 0)).sum())
        assert row.true_negatives == int((approved & (labels == 0)).sum())
        assert row.false_negatives == int((approved & (labels == 1)).sum())
        # Every applicant is in exactly one cell. A table that satisfies the four
        # counts above and not this one is arithmetically impossible, so this is
        # the cheap check that the assertions above are reading the columns they
        # think they are.
        assert (
            row.true_positives + row.false_positives + row.true_negatives + row.false_negatives
            == labels.size
        )
        assert row.approval_rate == pytest.approx(int(approved.sum()) / labels.size)
        assert row.approval_rate + row.predicted_default_rate == pytest.approx(1.0)
        assert row.total_cost == pytest.approx(
            row.false_negatives * fn_cost + row.false_positives * fp_cost
        )


# --- 2. KS against scipy ------------------------------------------------------


@given(pair=_labelled_scores())
@settings(max_examples=300, deadline=None)
def test_ks_matches_scipy_two_sample(pair: tuple[pd.Series, pd.Series]) -> None:
    """The signed, tie-collapsed KS equals a one-sided ``ks_2samp``.

    ``compute_ks_statistic`` maximizes ``F_non_default - F_default`` over the
    thresholds a cut-off can actually be placed at. scipy's ``alternative="greater"``
    statistic is ``max(cdf(data1) - cdf(data2))`` over the pooled sample points,
    which is the same maximum: evaluating the left-continuous ECDF at every
    distinct score is evaluating the right-continuous one at its predecessor, and
    both sets contain the final point where the difference is zero.

    So the correspondence is exact rather than approximate, and it pins both audit
    bugs at once. An absolute-valued KS (B06) would exceed this on any
    backwards-ranked draw, and a KS that cumulates row by row instead of collapsing
    ties (B07) would exceed it on any draw from the coarse grid.

    ``method="asymp"`` only to skip the exact p-value computation. The statistic
    does not depend on it.
    """
    y_true, y_score = pair
    labels = y_true.to_numpy()
    scores = y_score.to_numpy()

    expected = ks_2samp(
        scores[labels == 0], scores[labels == 1], alternative="greater", method="asymp"
    ).statistic
    assert compute_ks_statistic(y_true, y_score) == pytest.approx(expected, abs=1e-12)


@given(pair=_labelled_scores())
@settings(max_examples=100, deadline=None)
def test_the_two_ks_directions_recover_the_textbook_statistic(
    pair: tuple[pd.Series, pd.Series],
) -> None:
    """Both directions are non-negative, and the larger is the two-sided KS.

    This is the precise statement of what the signed version discards, and it is
    worth stating as an equality rather than a bound: the textbook two-sided
    statistic is exactly ``max(forward, backward)``, so a model ranked backwards
    keeps its textbook number and loses its reported one.

    Note what is *not* asserted, because the obvious stronger claim is false: both
    directions can be positive at once. That needs neither class to stochastically
    dominate the other, which an ECDF difference that changes sign gives you, and
    hypothesis produced one in about thirty examples.
    """
    y_true, y_score = pair
    forward = compute_ks_statistic(y_true, y_score)
    backward = compute_ks_statistic(y_true, -y_score)
    labels = y_true.to_numpy()
    scores = y_score.to_numpy()

    two_sided = ks_2samp(
        scores[labels == 0], scores[labels == 1], alternative="two-sided", method="asymp"
    ).statistic
    assert forward >= 0.0
    assert backward >= 0.0
    assert max(forward, backward) == pytest.approx(two_sided, abs=1e-12)


# --- 3. the parsers, round-tripped --------------------------------------------


@given(values=st.lists(_FINITE, min_size=1, max_size=50))
def test_coerce_numeric_survives_thousands_separators_and_currency(values: list[float]) -> None:
    """Formatting a float the way a re-exported extract does is reversible.

    ``$1,234.50`` is not hypothetical: it is what a spreadsheet writes when
    somebody opens the CSV and saves it, and the whole reason `coerce_numeric`
    strips those characters instead of failing on them.
    """
    assume(all(value not in {-1.0, 9999.0, 999999.0} for value in values))
    # `:,` and not `:,.6f`: a fixed precision rounds 0.0078125 to 0.007812 and the
    # round-trip fails on the formatting rather than on the parser. Python's
    # default float formatting is the shortest string that reads back identically,
    # which is what makes this a round-trip at all.
    formatted = pd.Series([f"${value:,}" for value in values])
    parsed = coerce_numeric(formatted)
    assert parsed.to_numpy() == pytest.approx(np.asarray(values, dtype=float), rel=1e-9)
    assert parsed.dtype == np.float64


@given(
    values=st.lists(
        st.floats(min_value=0.03, max_value=500.0, allow_nan=False),
        min_size=1,
        max_size=50,
    )
)
def test_parse_percent_divides_by_exactly_one_hundred(values: list[float]) -> None:
    """``'13.56%'`` -> ``0.1356``, and never a scale inferred from the data.

    The floor of 0.03 keeps the draw above the already-a-fraction guard, which
    fires when every value lands under 2% after dividing. That guard is checked by
    name in ``tests/test_feature_engineering.py``; the property here is that the
    divisor is fixed, because a divisor inferred per batch is how a set of
    low-utilization applicants gets scored a hundred times too low.
    """
    formatted = pd.Series([f"{value}%" for value in values])
    parsed = parse_percent(formatted, column_name="revol_util")
    assert parsed.to_numpy() == pytest.approx(np.asarray(values) / 100.0, rel=1e-9)


@given(months=st.integers(min_value=1, max_value=600), pad=st.sampled_from(["", " ", "  "]))
def test_parse_term_months_ignores_leading_whitespace(months: int, pad: str) -> None:
    """The two real extracts disagree about the leading space, and both must parse.

    This is a property rather than two examples because the padding and the number
    are independent, and the bug this shape catches - a `strip().split()` that
    works on one extract - only appears at one combination.
    """
    parsed = parse_term_months(pd.Series([f"{pad}{months} months"]))
    assert parsed.iloc[0] == float(months)


@given(years=st.integers(min_value=1, max_value=9))
def test_parse_employment_years_reads_under_one_as_zero_not_one(years: int) -> None:
    """``'< 1 year'`` is zero, and every ordinary ``'n years'`` is n.

    ``'< 1 year'`` contains the digit 1, so a bare digit extraction reads the
    shortest employment history in the data as a year of it. The censored ``'10+
    years'`` is asserted separately, outside the draw, because 10 means "10 or
    more" and there is nothing to parametrize.
    """
    assert parse_employment_years(pd.Series([f"{years} years"])).iloc[0] == float(years)
    assert parse_employment_years(pd.Series(["< 1 year"])).iloc[0] == 0.0
    assert parse_employment_years(pd.Series(["10+ years"])).iloc[0] == 10.0


# --- 4. PSI invariants --------------------------------------------------------


@given(
    reference=st.lists(_FINITE, min_size=2, max_size=200),
    comparison=st.lists(_FINITE, min_size=2, max_size=200),
)
@settings(max_examples=200, deadline=None)
def test_psi_is_non_negative_and_equals_its_own_table(
    reference: list[float], comparison: list[float]
) -> None:
    """PSI >= 0, and the scalar is the sum of the table published beside it.

    Each bin contributes ``(q - p) * ln(q / p)``, which carries the sign of
    ``q - p`` twice and is therefore non-negative term by term. Worth a property
    because the continuity floor (``0.5 / n`` on an empty bin) is applied to both
    populations independently and deliberately not renormalized - an arrangement
    that could plausibly produce a negative term and does not.

    The second half is the one that would catch a real divergence: ``psi_score.csv``
    is what a reader opens after seeing a PSI of 0.31, and a scalar computed
    separately from the table it explains is a scalar that can disagree with it.
    """
    left = pd.Series(reference, dtype="float64")
    right = pd.Series(comparison, dtype="float64")

    psi = population_stability_index(left, right)
    table = psi_table(left, right)

    assert np.isfinite(psi)
    assert psi >= 0.0
    assert (table["psi_contribution"] >= 0.0).all()
    assert psi == pytest.approx(float(table["psi_contribution"].sum()))
    # Shares are floored, never zeroed, which is what keeps the log finite.
    assert (table["reference_share"] > 0.0).all()
    assert (table["comparison_share"] > 0.0).all()


@given(values=st.lists(_FINITE, min_size=2, max_size=200))
@settings(max_examples=100, deadline=None)
def test_psi_against_itself_is_exactly_zero(values: list[float]) -> None:
    """Identical populations means zero, exactly, not approximately.

    Exactly, because identical counts give identical shares and ``ln(1) == 0`` in
    IEEE 754 - so any tolerance here would be hiding an asymmetry between how the
    two populations are binned. That asymmetry is the plausible bug: the edges come
    from the reference alone.
    """
    population = pd.Series(values, dtype="float64")
    assert population_stability_index(population, population) == 0.0


# --- 5. the threshold is monotone in the false-negative cost ------------------


@given(
    pair=_labelled_scores(min_size=20),
    costs=st.lists(st.floats(min_value=0.5, max_value=50.0, width=32), min_size=2, max_size=5),
)
@settings(max_examples=150, deadline=None)
def test_threshold_falls_as_missing_a_default_gets_more_expensive(
    pair: tuple[pd.Series, pd.Series], costs: list[float]
) -> None:
    """Raise the cost of a missed default and the selected threshold never rises.

    An applicant is declined at ``score >= threshold``, so a lower threshold
    declines more and approves fewer defaults. False negatives are therefore
    non-decreasing in the threshold and false positives non-increasing, which makes
    total cost the sum of an increasing function weighted by ``c_fn`` and a
    decreasing one weighted by ``c_fp``. Weighting the increasing part more heavily
    can only move the minimizer left, and taking the *highest* of the equally cheap
    thresholds - the documented tie-break - preserves that.

    This is the only property that asserts the cost matrix reaches the decision at
    all. A search that ignored its argument would satisfy every value-based test in
    the suite as long as the one committed cost ratio still produced the expected
    number.
    """
    y_true, y_score = pair
    scores = ValidationScores(y_true=y_true, y_score=y_score)

    selected = []
    for fn_cost in sorted(costs):
        try:
            selected.append(
                select_threshold_by_cost(
                    scores,
                    cost_matrix=CostMatrix(false_negative_cost=fn_cost, false_positive_cost=1.0),
                )
            )
        except ValueError:
            # No candidate clears the minimum approval rate. That is a documented
            # refusal about the score distribution, not about the cost, so it
            # cannot be a counterexample to a statement about the cost.
            assume(False)

    assert selected == sorted(selected, reverse=True)
