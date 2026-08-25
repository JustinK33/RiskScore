/**
 * Tests for the pure parts of the panel renderers.
 *
 * The renderers themselves need a canvas and are covered by
 * `scripts/probe_dashboard.mjs`. `embargoRows` is here because it is the one piece
 * of panel logic that decides something rather than formatting it: whether a
 * vintage was *removed* or merely absent, which is the difference between drawing
 * no bar and drawing a default rate of zero. `importanceRows` is here for the same
 * reason: it decides which features get no bar at all.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { embargoRows, importanceRows } from "./panels.js";

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
