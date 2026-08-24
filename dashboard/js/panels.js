/**
 * One function per chart: payload in, canvas drawn and an accessible table out.
 *
 * The split from `charts.js` is deliberate. That file knows about pixels, ticks,
 * and fonts and nothing about credit risk; this one knows what a calibration
 * curve means and nothing about how a tick is placed. So a chart type is added
 * here without touching the geometry, and a rendering fix there applies to every
 * chart at once.
 *
 * Every renderer returns `{label, table}`:
 *
 * - `label` becomes the canvas's `aria-label`. It states what the chart shows and
 *   its headline reading, because a label saying "calibration chart" tells a
 *   screen reader user only that they are missing something.
 * - `table` is the same numbers as a real `<table>`, rendered into a collapsed
 *   `<details>` beside the canvas. This is the actual alternative to the picture.
 *
 * And every renderer handles the empty payload by drawing a message rather than
 * an empty set of axes, because a blank plot with a labelled frame reads as a
 * measurement of zero.
 */

import {
  drawEmpty,
  drawFrame,
  drawLegend,
  drawLine,
  drawMarker,
  drawPoints,
  niceTicks,
  palette,
  prepareCanvas,
} from "./charts.js";
import { renderTable } from "./dom.js";
import { count, number, percent, toRows } from "./format.js";

/** Shared prologue: size the canvas, read the theme, get a context. */
function begin(canvas) {
  const colors = palette();
  const { ctx, width, height } = prepareCanvas(canvas);
  return { ctx, width, height, colors };
}

/** What every renderer returns when there is nothing to draw. */
function nothing(canvas, message) {
  const { ctx, width, height, colors } = begin(canvas);
  drawEmpty(ctx, width, height, message, colors);
  return { label: message, table: null };
}

// --- calibration ---------------------------------------------------------------

/**
 * Predicted probability against observed default rate, per decile, for both the
 * validation and the test partition, with the diagonal.
 *
 * Both partitions on one chart because the pair is the point: the calibrator was
 * *fitted* on validation, so validation sitting on the diagonal proves only that
 * the fit converged. Test is the claim. Showing validation alone was one of the
 * things that made the old artifact look healthier than it was.
 *
 * The two axes share one domain, so the diagonal is at 45 degrees and "above the
 * line" means under-predicting at a glance. Separate nice domains per axis would
 * tilt it and quietly destroy that reading.
 */
export function drawCalibration(canvas, payload) {
  const validation = toRows(payload?.validation || {});
  const test = toRows(payload?.test || {});
  if (validation.length === 0 && test.length === 0) {
    return nothing(canvas, "No calibration data for this run.");
  }
  const { ctx, width, height, colors } = begin(canvas);

  const series = [
    { key: "test", label: "Test", rows: test, color: colors.series[0] },
    { key: "validation", label: "Validation (in-sample)", rows: validation, color: colors.series[1] },
  ].filter((entry) => entry.rows.length > 0);

  const points = (rows) =>
    rows
      .map((row) => [Number(row.mean_predicted_probability), Number(row.observed_default_rate)])
      .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));

  const all = series.flatMap((entry) => points(entry.rows)).flat();
  // One domain for both axes, from the largest value either of them reaches.
  const axis = niceTicks(0, Math.max(...all, 0.05));
  const frame = drawFrame(ctx, {
    width,
    height,
    x: { ...axis, format: (value) => percent(value, 0) },
    y: { ...axis, format: (value) => percent(value, 0) },
    xTitle: "Predicted probability",
    yTitle: "Observed default rate",
    colors,
  });
  if (!frame) return { label: "Calibration chart, too small to draw.", table: null };

  // Perfect calibration, drawn first so the data sits on top of it.
  drawLine(ctx, frame, [[axis.min, axis.min], [axis.max, axis.max]], {
    color: colors.reference,
    width: 1.5,
    dash: [6, 5],
  });

  for (const entry of series) {
    const mapped = points(entry.rows);
    drawLine(ctx, frame, mapped, { color: entry.color, width: 2.5 });
    drawPoints(ctx, frame, mapped, { color: entry.color });
  }

  drawLegend(
    ctx,
    frame,
    [
      ...series.map((entry) => ({ label: entry.label, color: entry.color })),
      { label: "Perfect", color: colors.reference, dash: true },
    ],
    { colors },
  );

  const table = renderTable(
    [
      { key: "bin", label: "Decile", format: (row) => row.bin },
      { key: "partition", label: "Partition" },
      { key: "rows", label: "Loans", align: "right", format: (row) => count(row.rows) },
      {
        key: "mean_predicted_probability",
        label: "Predicted",
        align: "right",
        format: (row) => percent(row.mean_predicted_probability, 2),
      },
      {
        key: "observed_default_rate",
        label: "Observed",
        align: "right",
        format: (row) => percent(row.observed_default_rate, 2),
      },
    ],
    series.flatMap((entry) => entry.rows.map((row) => ({ ...row, partition: entry.label }))),
    { caption: "Calibration by predicted-probability decile" },
  );

  // The headline reading, in the label, because "calibration chart" is not
  // information. The largest gap is what a reader is looking for.
  const worst = test.reduce(
    (most, row) => {
      const gap = Math.abs(Number(row.observed_default_rate) - Number(row.mean_predicted_probability));
      return Number.isFinite(gap) && gap > most.gap ? { gap, row } : most;
    },
    { gap: -1, row: null },
  );
  const label = worst.row
    ? `Calibration by decile. On test, the largest gap between predicted and observed is ` +
      `${percent(worst.gap, 1)} in decile ${worst.row.bin}.`
    : "Calibration by decile, validation only.";

  return { label, table };
}

// --- threshold costs -----------------------------------------------------------

/**
 * Total cost and approval rate across all 99 candidate thresholds, with the one
 * this run selected marked.
 *
 * Two y axes, because cost is a weighted error count and approval rate is a
 * proportion. The old chart divided cost by its own maximum to share one axis,
 * which made the point where the curves cross look meaningful when it was an
 * artifact of the scaling.
 *
 * Validation only, and the chart says so: the threshold was chosen here, and a
 * test cost curve would invite reading the minimum off the partition the headline
 * metrics come from - which is the audit finding (B04) this project exists to
 * have fixed.
 */
export function drawThresholdCosts(canvas, payload, { selectedThreshold = null } = {}) {
  const rows = toRows(payload?.validation || {});
  if (rows.length === 0) return nothing(canvas, "No threshold cost curve for this run.");
  const { ctx, width, height, colors } = begin(canvas);

  const thresholds = rows.map((row) => Number(row.threshold));
  const costs = rows.map((row) => Number(row.total_cost));
  const approvals = rows.map((row) => Number(row.approval_rate));

  const frame = drawFrame(ctx, {
    width,
    height,
    x: { ...niceTicks(Math.min(...thresholds), Math.max(...thresholds)), format: (v) => number(v, 2) },
    y: niceTicks(0, Math.max(...costs)),
    y2: { ...niceTicks(0, 1), format: (v) => percent(v, 0) },
    xTitle: "Decision threshold",
    yTitle: "Total cost",
    y2Title: "Approval rate",
    colors,
  });
  if (!frame) return { label: "Threshold cost chart, too small to draw.", table: null };

  drawLine(
    ctx,
    frame,
    rows.map((row, index) => [thresholds[index], costs[index]]),
    { color: colors.series[0], width: 2.5 },
  );
  drawLine(
    ctx,
    frame,
    rows.map((row, index) => [thresholds[index], approvals[index]]),
    { color: colors.series[1], width: 2, toY: frame.toY2 },
  );

  if (Number.isFinite(Number(selectedThreshold))) {
    drawMarker(ctx, frame, Number(selectedThreshold), {
      color: colors.series[2],
      label: `selected ${number(selectedThreshold, 2)}`,
      colors,
    });
  }

  drawLegend(
    ctx,
    frame,
    [
      { label: "Total cost (left)", color: colors.series[0] },
      { label: "Approval rate (right)", color: colors.series[1] },
    ],
    { colors, align: "right" },
  );

  const cheapest = rows.reduce((best, row) =>
    Number(row.total_cost) < Number(best.total_cost) ? row : best,
  );

  const table = renderTable(
    [
      { key: "threshold", label: "Threshold", format: (row) => number(row.threshold, 2) },
      { key: "total_cost", label: "Total cost", align: "right", format: (row) => count(row.total_cost) },
      {
        key: "approval_rate",
        label: "Approval rate",
        align: "right",
        format: (row) => percent(row.approval_rate),
      },
      { key: "true_positives", label: "TP", align: "right", format: (row) => count(row.true_positives) },
      { key: "false_positives", label: "FP", align: "right", format: (row) => count(row.false_positives) },
      { key: "false_negatives", label: "FN", align: "right", format: (row) => count(row.false_negatives) },
      { key: "true_negatives", label: "TN", align: "right", format: (row) => count(row.true_negatives) },
    ],
    rows,
    { caption: "Cost and confusion counts per candidate threshold, on validation" },
  );

  const label =
    `Total cost and approval rate across ${rows.length} candidate thresholds on validation. ` +
    `Cost is lowest at ${number(cheapest.threshold, 2)}, where the approval rate is ` +
    `${percent(cheapest.approval_rate)}.` +
    (Number.isFinite(Number(selectedThreshold))
      ? ` This run selected ${number(selectedThreshold, 2)}.`
      : "");

  return { label, table };
}
