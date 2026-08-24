/**
 * Tests for the pure parts of the panel renderers.
 *
 * The renderers themselves need a canvas and are covered by
 * `scripts/probe_dashboard.mjs`. `embargoRows` is here because it is the one piece
 * of panel logic that decides something rather than formatting it: whether a
 * vintage was *removed* or merely absent, which is the difference between drawing
 * no bar and drawing a default rate of zero.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { embargoRows } from "./panels.js";

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
