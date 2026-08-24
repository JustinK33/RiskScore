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
  drawBars,
  drawCategoryLabels,
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
import { count, isMissing, labelize, MISSING, number, percent, signed, toRows } from "./format.js";

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

/**
 * An x axis with no ticks, for a chart whose categories are drawn by
 * `drawCategoryLabels` instead.
 *
 * The domain is 0..1 so a category's centre is `(index + 0.5) / count` in data
 * space, which is what lets a line series be overlaid on bars through the frame's
 * own `toX` rather than by reaching for pixel offsets.
 */
const ORDINAL_X = { min: 0, max: 1, step: 1, ticks: [] };

/** The data-space x of category `index` of `total`, matching `drawBars`' slots. */
const slotCentre = (index, total) => (index + 0.5) / total;

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

// --- the embargo ---------------------------------------------------------------

/**
 * The embargo's before/after rates as one row per origination year.
 *
 * Pure, exported, and tested, because the interesting case is structural rather
 * than numeric: a vintage that appears in `before` and not in `after` was removed
 * *entirely* - not one loan in it had matured by the snapshot. Treating that as a
 * missing number would draw no bar, and no bar next to a 32.8% bar reads as a
 * default rate of zero, which is the opposite of what happened.
 */
export function embargoRows(embargo) {
  const before = embargo?.default_rate_by_vintage_before || {};
  const after = embargo?.default_rate_by_vintage_after || {};
  const vintages = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort(
    // Numeric where both parse - "2013" and "2016" happen to sort correctly as
    // strings, but a quarterly vintage key like "2013Q4" would not.
    (left, right) => Number(left) - Number(right) || left.localeCompare(right),
  );
  // `isMissing` rather than `Number.isFinite` alone, because `Number(null)` is 0 -
  // a null rate would plot as a vintage in which nobody defaulted.
  const rate = (source, vintage) => {
    if (isMissing(source[vintage])) return null;
    const value = Number(source[vintage]);
    return Number.isFinite(value) ? value : null;
  };
  return vintages.map((vintage) => {
    const beforeRate = rate(before, vintage);
    const afterRate = rate(after, vintage);
    return {
      vintage,
      before: beforeRate,
      after: afterRate,
      dropped: afterRate === null && beforeRate !== null,
      change: beforeRate === null || afterRate === null ? null : afterRate - beforeRate,
    };
  });
}

/**
 * Measured default rate by origination year, before and after the embargo.
 *
 * This is the chart the project exists to be able to draw. Filtering to closed
 * loans without an outcome-maturity rule keeps every loan that has already
 * defaulted and throws away the ones still paying, so the measured default rate
 * climbs with every vintage: 16.2%, 17.5%, 18.8%, 32.8%. That reads as a book
 * deteriorating year over year and it is entirely survivorship bias - the recent
 * vintages have simply had less time to finish repaying. Requiring
 * `issue_d + term <= snapshot` flattens it to 16.2%, 14.5%, 13.0%, and deletes
 * 2016 outright.
 *
 * Grouped bars rather than two lines: these are four independent measurements, not
 * a trend through time, and a line implies you can read between the years.
 */
export function drawEmbargo(canvas, manifest) {
  const rows = embargoRows(manifest?.embargo);
  if (rows.length === 0) return nothing(canvas, "This run recorded no embargo comparison.");
  const { ctx, width, height, colors } = begin(canvas);

  const values = rows.flatMap((row) => [row.before, row.after]).filter((value) => value !== null);
  const frame = drawFrame(ctx, {
    width,
    height,
    x: ORDINAL_X,
    y: { ...niceTicks(0, Math.max(...values, 0.05)), format: (value) => percent(value, 0) },
    xTitle: "Origination year",
    yTitle: "Measured default rate",
    colors,
  });
  if (!frame) return { label: "Embargo chart, too small to draw.", table: null };

  // Warm for the biased series and the accent green for the corrected one: these
  // two are not peers, one of them is wrong, and the colours should not suggest a
  // reader may pick either.
  const vintages = rows.map((row) => row.vintage);
  const { slot } = drawBars(ctx, frame, vintages, [
    { values: rows.map((row) => row.before), color: colors.series[2] },
    // `NaN` rather than `null`, because that is what `drawBars` skips - and a
    // dropped vintage must draw nothing rather than a bar of height zero.
    { values: rows.map((row) => row.after ?? Number.NaN), color: colors.series[0] },
  ]);
  drawCategoryLabels(ctx, frame, vintages, { colors, slot });
  drawLegend(
    ctx,
    frame,
    [
      { label: "Closed loans only", color: colors.series[2] },
      { label: "Embargo applied", color: colors.series[0] },
    ],
    { colors },
  );

  const table = renderTable(
    [
      { key: "vintage", label: "Vintage" },
      {
        key: "before",
        label: "Closed only",
        align: "right",
        format: (row) => percent(row.before, 1),
      },
      {
        key: "after",
        label: "Embargoed",
        align: "right",
        format: (row) => (row.dropped ? "removed" : percent(row.after, 1)),
      },
      {
        key: "change",
        label: "Change",
        align: "right",
        // Percentage points, not a percentage of a percentage: the difference
        // between two rates is not itself a rate.
        format: (row) => (row.change === null ? MISSING : `${signed(row.change * 100, 1)} pp`),
      },
    ],
    rows,
    { caption: "Measured default rate by vintage, with and without the outcome-maturity embargo" },
  );

  const worst = rows.reduce((most, row) => ((row.before ?? -1) > (most.before ?? -1) ? row : most), rows[0]);
  const dropped = rows.filter((row) => row.dropped).map((row) => row.vintage);
  const kept = rows.filter((row) => row.after !== null);
  const highestKept = kept.reduce((most, row) => (row.after > most.after ? row : most), kept[0] || null);
  const label =
    `Measured default rate by origination year. Counting closed loans alone, ${worst.vintage} reads ` +
    `${percent(worst.before, 1)}` +
    (highestKept
      ? `; with the embargo applied the highest vintage is ${highestKept.vintage} at ${percent(highestKept.after, 1)}`
      : "") +
    (dropped.length
      ? `. ${dropped.join(", ")} ${dropped.length === 1 ? "drops" : "drop"} out entirely - nothing originated then had matured by the snapshot.`
      : ".");

  return { label, table };
}

// --- per-vintage performance ---------------------------------------------------

/** Partition names short enough for an axis label at 320px. */
const PARTITION_TAGS = { train: "train", validation: "val", test: "test" };

/**
 * Default rate and AUC for each vintage, tagged with the partition it landed in.
 *
 * The category is the vintage *and* the partition, not the vintage alone: the
 * split is chronological, so 2014 is split across train and validation and 2015
 * across validation and test. Merging them would average a fitted partition with a
 * held-out one and report the result as one number for the year.
 *
 * Two axes, because the pair is the diagnostic. A default rate that holds steady
 * while AUC falls means the model is degrading; both moving together usually means
 * the vintage is genuinely different. The AUC axis starts at 0.5 rather than 0
 * because 0.5 is a coin flip, and a bar chart from zero makes 0.62 look like
 * substantial skill.
 */
export function drawVintages(canvas, payload) {
  const rows = toRows(payload?.vintages || {});
  if (rows.length === 0) return nothing(canvas, "No per-vintage breakdown for this run.");
  const { ctx, width, height, colors } = begin(canvas);

  // Abbreviated by lookup, not by truncation: `"validation".slice(0, 5)` is
  // "valid", which reads as an adjective about the vintage rather than the name of
  // a partition. An unknown partition keeps its full name and simply takes more
  // room, which is the failure a reader can act on.
  const categories = rows.map(
    (row) => `${row.vintage} ${PARTITION_TAGS[row.partition] || row.partition}`,
  );
  const rates = rows.map((row) => Number(row.default_rate));
  const aucs = rows.map((row) => Number(row.auc_roc));

  const frame = drawFrame(ctx, {
    width,
    height,
    x: ORDINAL_X,
    y: { ...niceTicks(0, Math.max(...rates.filter(Number.isFinite), 0.05)), format: (v) => percent(v, 0) },
    y2: { ...niceTicks(0.5, Math.max(...aucs.filter(Number.isFinite), 0.75)), format: (v) => number(v, 2) },
    xTitle: "Vintage and partition",
    yTitle: "Default rate",
    y2Title: "AUC ROC",
    colors,
  });
  if (!frame) return { label: "Vintage chart, too small to draw.", table: null };

  const { slot } = drawBars(ctx, frame, categories, [
    { values: rates, color: colors.series[1] },
  ]);
  // The AUC series rides the ordinal axis by asking for each bar slot's centre in
  // data space, so it stays aligned with the bars at any width without this
  // function knowing a single pixel offset.
  const aucPoints = aucs
    .map((value, index) => [slotCentre(index, categories.length), value])
    .filter(([, value]) => Number.isFinite(value));
  drawLine(ctx, frame, aucPoints, { color: colors.series[0], width: 2.5, toY: frame.toY2 });
  drawPoints(ctx, frame, aucPoints, { color: colors.series[0], toY: frame.toY2 });

  drawCategoryLabels(ctx, frame, categories, { colors, slot });
  drawLegend(
    ctx,
    frame,
    [
      { label: "Default rate (left)", color: colors.series[1] },
      { label: "AUC ROC (right)", color: colors.series[0] },
    ],
    { colors, align: "right" },
  );

  const table = renderTable(
    [
      { key: "vintage", label: "Vintage" },
      { key: "partition", label: "Partition", format: (row) => labelize(row.partition) },
      { key: "rows", label: "Loans", align: "right", format: (row) => count(row.rows) },
      { key: "defaults", label: "Defaults", align: "right", format: (row) => count(row.defaults) },
      {
        key: "default_rate",
        label: "Default rate",
        align: "right",
        format: (row) => percent(row.default_rate, 1),
      },
      { key: "auc_roc", label: "AUC", align: "right", format: (row) => number(row.auc_roc) },
      { key: "ks_statistic", label: "KS", align: "right", format: (row) => number(row.ks_statistic) },
      { key: "brier_score", label: "Brier", align: "right", format: (row) => number(row.brier_score) },
      {
        key: "approval_rate",
        label: "Approval",
        align: "right",
        format: (row) => percent(row.approval_rate, 1),
      },
    ],
    rows,
    { caption: "Metrics per origination year, within the partition that year belongs to" },
  );

  const scored = rows.filter((row) => !isMissing(row.auc_roc));
  const weakest = scored.reduce(
    (least, row) => (Number(row.auc_roc) < Number(least.auc_roc) ? row : least),
    scored[0],
  );
  const label = weakest
    ? `Default rate and AUC for ${rows.length} vintage-partition groups. AUC is weakest on ` +
      `${weakest.vintage} (${weakest.partition}) at ${number(weakest.auc_roc)}, against a default rate ` +
      `there of ${percent(weakest.default_rate, 1)}.`
    : `Default rate for ${rows.length} vintage-partition groups; no AUC was recorded.`;

  return { label, table };
}
