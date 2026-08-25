/**
 * Tests for the pure parts of the panel renderers.
 *
 * The renderers themselves need a canvas and are covered by
 * `scripts/probe_dashboard.mjs`. `embargoRows` is here because it is the one piece
 * of panel logic that decides something rather than formatting it: whether a
 * vintage was *removed* or merely absent, which is the difference between drawing
 * no bar and drawing a default rate of zero. `importanceRows` is here for the same
 * reason: it decides which features get no bar at all, and `comparisonRows` for the
 * same reason again: it decides which direction counts as an improvement, and a
 * comparison that colours a regression green is worse than no comparison.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { comparisonRows, embargoRows, importanceRows } from "./panels.js";

/** The live run's own figures, so the fixture cannot drift from reality. */
const EMBARGO = {
  snapshot: "2018-12-01",
  default_rate_by_vintage_before: { 2013: 0.16192, 2014: 0.1749, 2015: 0.18829, 2016: 0.32767 },
  default_rate_by_vintage_after: { 2013: 0.16192, 2014: 0.14457, 2015: 0.12974 },
};

test("a vintage the embargo removed entirely is marked, not treated as zero", () => {
  const rows = embargoRows(EMBARGO);
  const dropped = rows.filter((row) => row.dropped);

  assert.deepEqual(
    dropped.map((row) => row.vintage),
    ["2016"],
  );
  assert.equal(dropped[0].after, null);
  // No change is computable when one side does not exist, and reporting one would
  // claim 2016's rate fell to zero.
  assert.equal(dropped[0].change, null);
});

test("the change is the corrected rate minus the biased one", () => {
  const rows = embargoRows(EMBARGO);
  const year2015 = rows.find((row) => row.vintage === "2015");

  assert.ok(year2015.change < 0, "the embargo lowers the measured rate");
  assert.equal(Number(year2015.change.toFixed(5)), Number((0.12974 - 0.18829).toFixed(5)));
});

test("vintages are ordered by year, not by key insertion", () => {
  const rows = embargoRows({
    default_rate_by_vintage_before: { 2015: 0.2, 2013: 0.1, 2014: 0.15 },
    default_rate_by_vintage_after: {},
  });

  assert.deepEqual(
    rows.map((row) => row.vintage),
    ["2013", "2014", "2015"],
  );
});

test("a vintage present only after the embargo is not marked as dropped", () => {
  // Cannot happen from the pipeline, but the guard is what stops `dropped` from
  // meaning "one of the two sides is missing" instead of "this was removed".
  const rows = embargoRows({
    default_rate_by_vintage_before: {},
    default_rate_by_vintage_after: { 2015: 0.13 },
  });

  assert.equal(rows[0].dropped, false);
  assert.equal(rows[0].before, null);
});

test("no embargo record yields no rows rather than throwing", () => {
  assert.deepEqual(embargoRows(undefined), []);
  assert.deepEqual(embargoRows({}), []);
});

test("a non-numeric rate is dropped rather than plotted as NaN", () => {
  const rows = embargoRows({
    default_rate_by_vintage_before: { 2015: null },
    default_rate_by_vintage_after: { 2015: "n/a" },
  });

  assert.equal(rows[0].before, null);
  assert.equal(rows[0].after, null);
});

/** The live run's top three and its one zero-weight feature. */
const SHAP = {
  features: {
    feature: ["credit_utilization", "fico_midpoint", "loan_to_income_ratio", "term"],
    label: ["Revolving utilization.", "FICO midpoint.", "Principal over income.", "Loan term."],
    mean_abs_log_odds: [0.4594175382323515, 0.2712011441288449, 0.0021961959380983, 0.0],
    mean_log_odds: [0.012209866866249, 0.016771379982513, 0.0003039346708774, 0.0],
    columns: [2, 2, 2, 2],
    rank: [1, 2, 3, 4],
  },
};

test("bar widths are relative to the largest magnitude, not to their sum", () => {
  const rows = importanceRows(SHAP);

  assert.equal(rows[0].width, 100);
  // 0.2712 / 0.4594 = 59.0%. Against the sum of the four it would be 36%, which
  // would make the second-strongest driver of the book look like a third of one.
  assert.equal(Number(rows[1].width.toFixed(1)), 59.0);
});

test("a magnitude too small to see gets the 2% floor", () => {
  const rows = importanceRows(SHAP);

  // 0.0022 / 0.4594 is 0.48% - under half a pixel in a 100px column.
  assert.equal(rows[2].width, 2);
});

test("a feature with exactly zero weight gets no bar at all", () => {
  const rows = importanceRows(SHAP);

  // The one case the floor must not apply to: a visible sliver would claim `term`
  // contributes something, when the split restricts the book to a single term.
  assert.equal(rows[3].magnitude, 0);
  assert.equal(rows[3].width, 0);
});

test("the signed mean is kept separate from the magnitude", () => {
  const rows = importanceRows(SHAP);

  // Same feature, an order of magnitude apart: reading the magnitude as the
  // direction is the usual way an importance chart gets misused.
  assert.equal(rows[0].magnitude, 0.4594175382323515);
  assert.equal(rows[0].direction, 0.012209866866249);
});

test("no SHAP summary yields no rows rather than throwing", () => {
  assert.deepEqual(importanceRows(undefined), []);
  assert.deepEqual(importanceRows({}), []);
});

test("a missing magnitude is null rather than zero, and draws no bar", () => {
  const rows = importanceRows({
    features: { feature: ["x"], mean_abs_log_odds: [null], mean_log_odds: [null] },
  });

  // `Number(null)` is 0, so a null magnitude would otherwise be indistinguishable
  // from a feature the model measured and gave no weight.
  assert.equal(rows[0].magnitude, null);
  assert.equal(rows[0].direction, null);
  assert.equal(rows[0].width, 0);
});

/**
 * The live comparison's own figures: logistic regression under both feature tiers,
 * trimmed to four metrics covering every direction the payload can declare.
 */
const COMPARISON = {
  baseline: "logistic_regression / origination_only",
  metrics: {
    auc_roc: "higher",
    brier_score: "lower",
    selected_threshold_total_cost: "lower",
    approval_rate: "neutral",
  },
  variants: [
    {
      variant: "logistic_regression / origination_only",
      model_type: "logistic_regression",
      feature_tier: "origination_only",
      auc_roc: 0.6829908133365012,
      brier_score: 0.10972252032559372,
      selected_threshold_total_cost: 629.0,
      approval_rate: 0.8554663372452131,
      auc_roc_delta: 0.0,
      brier_score_delta: 0.0,
      selected_threshold_total_cost_delta: 0.0,
    },
    {
      variant: "logistic_regression / with_lender_priced",
      model_type: "logistic_regression",
      feature_tier: "with_lender_priced",
      auc_roc: 0.7254329347166125,
      brier_score: 0.10564044101384307,
      selected_threshold_total_cost: 591.0,
      approval_rate: 0.62816553428042,
      auc_roc_delta: 0.042442121380111275,
      brier_score_delta: -0.004082079311750655,
      selected_threshold_total_cost_delta: -38.0,
    },
  ],
};

/** The cell for one metric of one variant, by name rather than by position. */
const cellFor = (row, metric) => row.cells.find((cell) => cell.metric === metric);

test("a lower score is an improvement when the payload says lower is better", () => {
  const [, variant] = comparisonRows(COMPARISON);

  // The whole point of reading the direction from the payload: both of these are
  // improvements, and their deltas have opposite signs.
  assert.equal(cellFor(variant, "auc_roc").tone, "ok");
  assert.equal(cellFor(variant, "brier_score").tone, "ok");
  assert.equal(cellFor(variant, "auc_roc").delta > 0, true);
  assert.equal(cellFor(variant, "brier_score").delta < 0, true);
});

test("a rise in a lower-is-better metric is a regression", () => {
  const worse = {
    ...COMPARISON,
    variants: [COMPARISON.variants[0], { ...COMPARISON.variants[1], brier_score_delta: 0.004 }],
  };

  assert.equal(cellFor(comparisonRows(worse)[1], "brier_score").tone, "warn");
});

test("a metric the payload marks neutral is never coloured", () => {
  const [, variant] = comparisonRows(COMPARISON);
  const approval = cellFor(variant, "approval_rate");

  // Approval fell 23 points, which is a large change and not a worse one: it is a
  // consequence of the threshold, and colouring it would read as a verdict.
  assert.equal(approval.tone, null);
  assert.equal(approval.delta, null);
  assert.equal(approval.text, "62.8%");
});

test("the baseline row carries no deltas at all", () => {
  const [baseline] = comparisonRows(COMPARISON);

  assert.equal(baseline.isBaseline, true);
  // Zero is what the payload sends, and `+0.0000` against every baseline number
  // reads as a measurement that came out flat rather than as the reference.
  assert.equal(baseline.cells.every((cell) => cell.delta === null), true);
  assert.equal(baseline.cells.every((cell) => cell.deltaText === ""), true);
});

test("the baseline is the one the payload names, not the first row", () => {
  const reordered = {
    ...COMPARISON,
    variants: [COMPARISON.variants[1], COMPARISON.variants[0]],
  };
  const rows = comparisonRows(reordered);

  assert.deepEqual(
    rows.map((row) => row.isBaseline),
    [false, true],
  );
});

test("a delta is written in its metric's own units", () => {
  const [, variant] = comparisonRows(COMPARISON);

  assert.equal(cellFor(variant, "auc_roc").deltaText, "+0.0424");
  // A weighted error count, not a fraction: `-38.0000` would suggest four
  // meaningful decimals in a number that counts loans.
  assert.equal(cellFor(variant, "selected_threshold_total_cost").deltaText, "-38");
  assert.equal(cellFor(variant, "selected_threshold_total_cost").text, "591");
});

test("no comparison yields no rows rather than throwing", () => {
  assert.deepEqual(comparisonRows(null), []);
  assert.deepEqual(comparisonRows({}), []);
  assert.deepEqual(comparisonRows({ variants: [] }), []);
});

test("a metric with no label falls back to a readable one", () => {
  const rows = comparisonRows({
    metrics: { some_new_metric: "higher" },
    variants: [{ variant: "a", some_new_metric: 1 }],
  });

  // A metric added server-side appears without a dashboard change.
  assert.equal(rows[0].cells[0].label, "Some New Metric");
  assert.equal(rows[0].cells[0].text, "1.0000");
});
