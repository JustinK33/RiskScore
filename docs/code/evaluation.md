# `src/risk_score/evaluation.py`

## Purpose

Compute the numbers the project reports, and choose the threshold it operates at.

Three of the four headline metrics came from this file, and two of them were wrong in the same direction: they made the model look better than it was.

**KS was direction-blind.**
The committed artifact reported `auc_roc: 0.0696` beside `ks_statistic: 0.9304`.
An AUC far below 0.5 next to a near-perfect KS is the signature of inverted labels, meaning the model separates the two classes almost completely and ranks them backwards.
The old implementation took `.abs()` of the gap between the two cumulative curves, so backwards separation scored exactly as well as correct separation.
The one number that could have contradicted the AUC agreed with it instead (audit B06).
KS here is signed in the direction a credit score is supposed to run, higher score meaning more defaults, so an inverted model now scores near zero and the two metrics can no longer tell opposite stories.

**KS was also tie-blind.**
Ranking rows and cumulating one at a time splits a block of equal scores in whatever order the sort happened to produce, so the running gap can peak *inside* a tie, at a cut-off no threshold can implement.
With a boosted model emitting a few hundred distinct probabilities over a million rows, those blocks are large.
Ties are now aggregated to their last row before the maximum is taken (audit B07).

**The threshold search minimized without constraint.**
At a 5:1 false-negative ratio the true unconstrained minimum is frequently "decline everybody", which costs nothing in missed defaults.
It also broke ties toward the *lowest* threshold, so where several cut-offs cost the same it published the one that declined the most applicants (audit B10).

**And the search was quadratic in the grid.**
99 candidate thresholds each ran a full `confusion_matrix` over every row: 99 passes over the array to answer 99 questions that are all cumulative sums of one sorted pass (audit P03).

## Public API

| Name | What it is |
| --- | --- |
| `ClassificationMetrics` | The frozen container the pipeline reports: AUC, average precision, KS, Brier, default rate, approval rate. |
| `CostMatrix` | What a false negative and a false positive each cost. Only the ratio matters, so the units are the caller's. |
| `ValidationScores` | A label/score pair tagged as coming from validation. The only thing `select_threshold_by_cost` accepts. |
| `as_metric_arrays` | The shared input gate: validate one label/score pair, return plain numpy arrays. |
| `compute_auc_roc` / `compute_average_precision` / `compute_brier_score` | Scalars, thin wrappers over sklearn behind the shared gate. |
| `compute_ks_statistic` | Signed, tie-collapsed KS. |
| `compute_precision_recall` | The full PR curve, one row per operating point. |
| `compute_threshold_cost_table` | Confusion counts, approval rate, and total cost at every candidate threshold. |
| `select_threshold_by_cost` | The cheapest threshold that still approves at least `min_approval_rate` of applicants. |
| `DEFAULT_THRESHOLD_GRID` | Every whole percentage point from 0.01 to 0.99. |
| `DEFAULT_MIN_APPROVAL_RATE` | `0.20`. |

## Inputs and outputs

Everything takes a pair of pandas Series and returns a float or a DataFrame.
Nothing here reads a file, writes a file, fits an estimator, or holds state between calls.

`as_metric_arrays` is the single door.
Every metric in the module starts there, and so does `risk_score.calibration.compute_calibration_curve`, because every number the project reports about a partition should have been through the same guards.
Those guards are:

- labels and scores the same length,
- the partition non-empty, because an empty one returns NaN rather than raising,
- labels a subset of `{0, 1}`, because a stray `2` silently redefines what a positive is,
- all scores finite, because a NaN score sorts unpredictably and has no place in a ranking.

The error names the offending values (`Labels must be 0 or 1; found [2]`), not just the condition.

`compute_threshold_cost_table` returns one row per candidate threshold with `threshold`, the four confusion counts, `predicted_default_rate`, `approval_rate`, and `total_cost`.

## Invariants and failure modes

**An applicant is declined when `score >= threshold`.**
So the approved population is `score < threshold`.
Every count in the cost table depends on that convention, and the opposite convention produces a table that looks equally plausible, which is why it is asserted directly rather than left implicit.

**KS is never negative, and never exceeds the largest achievable gap.**
Both cumulative curves end at 1, so their difference is 0 at the final threshold and the maximum is at least that.
A worthless score reports 0, not -0.4.
The signed definition coincides with the textbook two-sample statistic for any model ranked the right way round, which is the only case where either number means anything.

**A selected threshold approves at least `min_approval_rate` of applicants.**
This is a definition, not a safety margin.
With a 5:1 false-negative cost and a 15% default rate, declining every applicant costs 0.85 units per row while approving every applicant costs 0.75, so the unconstrained minimum can sit at the very bottom of the grid.
"Decline everyone" is the one operating point guaranteed to be useless.
Falling below the floor raises with the most permissive rate the grid could offer, rather than silently returning the corner.

**Ties in cost break toward the higher threshold.**
Equal cost means identical confusion counts, and between two rules with identical outcomes the one that declines fewer applicants is the one to publish.

**Metrics that are undefined on one class refuse rather than return NaN.**
AUC, average precision, KS, and the PR curve all require both classes present and say how many positives they found.
The Brier score deliberately does not: it is a squared error rather than a ranking statistic, and it is the only number a wholly-defaulted vintage can still report.

**`select_threshold_by_cost` refuses bare arrays.**
It takes a `ValidationScores` and raises `TypeError` on anything else.
The runtime check is load-bearing rather than belt-and-braces: pandas ships no type stubs in this project, so a bare `pd.Series` is `Any` to mypy and the annotation alone would let the call through.
Wrapping test scores in `ValidationScores` is of course still possible, but it is now a deliberate, greppable sentence rather than an argument in the wrong position.

**A `CostMatrix` with both costs zero raises.**
Every threshold would then cost the same and the search would return whatever the tie-break preferred, which is a decision with no input reported as if it had one.

**Performance.**
The cost table is one `argsort`, one `cumsum` with a prepended zero, and one `np.searchsorted(..., side="left")`, so it is O(n log n + k) rather than O(n·k).
On 400k validation rows and 99 thresholds that is one pass over the data instead of ninety-nine.

## What must NOT live here

- **Fitting anything.** Choosing a threshold is the one fitted decision in this file, and it is confined behind `ValidationScores` for that reason. A calibrator belongs in `calibration.py`.
- **Reading or writing files.** The pipeline decides where a CSV lands.
- **Plotting.** `calibration.py` owns the one figure, and no metric function should import matplotlib.
- **Drift and PSI.** They compare two partitions rather than describing one, and they get their own module.
- **The `ClassificationMetrics` to JSON mapping.** That shape is the artifact contract and belongs with the code that writes artifacts.

## Related tests

`tests/test_evaluation.py`, 36 tests, every assertion against a hand-computed constant, a scipy or sklearn oracle, or a second implementation of the same definition.
The old suite asserted `0 <= result <= 1`, which is exactly why the inverted-KS bug survived: 0.9304 is between 0 and 1 and so is the 0.35 it should have been.
That assertion style is banned.

Named audit regressions:

- `test_b06_a_perfectly_inverted_score_has_ks_near_zero_not_near_one` reproduces the artifact's AUC 0.070 / KS 0.930 pairing, and also asserts `ks_2samp(...).statistic == 1.0` on the same input to show the two definitions genuinely differ.
- `test_b07_a_tied_block_cannot_be_cut_in_half` builds a case where row-by-row cumulation would report perfect separation for a score with almost none.
- `test_b10_selection_refuses_to_decline_almost_everybody` shows the unconstrained minimum is "approve nobody" and that the guard rejects it.
- `test_b10_ties_break_toward_the_higher_threshold` pins the tie-break at 0.80 across a zero-cost band running from 0.31.
- `test_b04_bare_arrays_are_refused_by_the_threshold_search` is the partition guard.

Non-audit tests worth knowing about:

- `test_the_cost_table_matches_a_confusion_matrix_at_every_threshold` checks the vectorized table against the 99-call loop it replaced, on scores deliberately rounded to two decimals so ties are present.
- `test_ks_equals_the_scipy_two_sample_statistic_when_the_score_is_ranked_correctly` pins the claim that the signed definition is not a different metric for correctly ordered models.
- `test_a_higher_false_negative_cost_never_relaxes_the_threshold` is a monotonicity property: if it fails, the cost column and the counts have drifted apart.

## Known limits

- **The threshold grid is fixed at two decimal places.** That is a deliberate choice, because a threshold is a published lending policy and 0.14 is a policy while 0.1372549 is a fitting artifact. A caller who wants finer resolution passes their own `thresholds`.
- **`min_approval_rate` is a blunt floor.** A real lender constrains approval rate, expected loss, *and* portfolio yield at once, and yield needs a loan amount and a price this model does not carry.
- **No confidence intervals.** Every metric is a point estimate, so a 0.75 AUC on 350 test rows is presented with the same authority as one on 350,000. Sample sizes travel in the metrics payload as partial mitigation; bootstrap intervals are the honest fix and are not implemented.
- **The cost matrix is symmetric across applicants.** One number for all false negatives assumes a missed default on a 2,000 dollar loan costs the same as one on 35,000.
- **Average precision uses sklearn's step-wise definition,** which is not the trapezoidal interpolation some references use. The two disagree by a small amount on coarse curves.
