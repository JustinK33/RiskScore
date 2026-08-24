/**
 * A canvas chart kit: scales, ticks, axes, legends, lines, points, bars.
 *
 * No charting library, for the same reason there is no bundler - this is six
 * chart types over payloads the service shapes for exactly this purpose, and the
 * smallest credible library is larger than the file you are reading and would
 * still need every one of the decisions below made on top of it.
 *
 * What was wrong with the version this replaces, because each fix is a function
 * here:
 *
 * **Nothing was measured.** Axis titles were placed with hardcoded half-widths -
 * `chartWidth / 2 - 64`, `+ 62`, `- 28`, `- 52` - which are only correct for one
 * string in one font at one size. They were not even correct then: the font was
 * `Inter`, which the page requested in five places and never loaded, so every
 * offset was tuned against a font the browser was not using. Here, every string
 * is positioned with `measureText` and `textAlign`/`textBaseline`.
 *
 * **The left padding was a constant,** so a y axis labelled `0` and one labelled
 * `12,480` got the same 56px and the second one overlapped the plot.
 * `axisPadding` measures the widest label it is actually going to draw.
 *
 * **Ticks were `i / 5`,** which is fine for a probability axis and wrong for
 * everything else: the cost axis printed `Math.round(maxCost * value)`, giving
 * ticks at 2,496 and 4,992. `niceTicks` picks a 1-2-5 step so the labels are
 * numbers a human would have chosen.
 *
 * **Colours were hex literals in the drawing code,** so dark mode was
 * impossible without editing JS. `palette` reads the CSS custom properties from
 * `tokens.css`, so the theme owns them.
 *
 * **`prepareCanvas` reassigned the bitmap on every draw,** which clears it - a
 * visible blank flash per resize frame. It now only resizes when the target size
 * actually changed, and resets the transform every draw instead.
 */

/** Space between the plot and the edge of the canvas, before measurement. */
const BASE_PADDING = { top: 18, right: 16, bottom: 16, left: 12 };

/** Tick mark length, and the gap between a tick and its label. */
const TICK = 5;
const GAP = 6;

/** How many ticks to aim for. A request, not a promise - `niceTicks` rounds. */
const TARGET_TICKS = 5;

/**
 * A 1-2-5 tick step over `[min, max]`, plus the domain rounded out to it.
 *
 * The returned domain is *wider* than the data on purpose: an axis whose last
 * tick is 0.83 because that happened to be the maximum is harder to read than one
 * ending at 1.0, and a line that touches the frame looks clipped.
 *
 * Degenerate inputs are the interesting cases, and all three occur in practice -
 * a single-bin calibration curve, a vintage where every value is identical, a
 * cost column that is all zeros. Each gets a domain with a real width rather than
 * a division by zero that renders every point on one pixel.
 */
export function niceTicks(min, max, target = TARGET_TICKS) {
  let low = Number(min);
  let high = Number(max);
  if (!Number.isFinite(low) || !Number.isFinite(high)) return { min: 0, max: 1, step: 0.25, ticks: [0, 0.25, 0.5, 0.75, 1] };
  if (high < low) [low, high] = [high, low];
  if (high === low) {
    // Expand around the value proportionally, so 1e6 does not get a ±1 window and
    // 0 does not get a zero-width one.
    const pad = Math.abs(low) * 0.1 || 1;
    low -= pad;
    high += pad;
  }

  const rawStep = (high - low) / Math.max(1, target);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  // 1, 2, 5, 10 - the steps people read without doing arithmetic. The thresholds
  // are the geometric midpoints (sqrt(2), sqrt(10), sqrt(50)) rather than the
  // arithmetic ones, so the chosen step is the *nearest* of the four rather than
  // always the next one up: a raw step of 2496 becomes 2000 and yields seven
  // ticks, where rounding up to 5000 would yield three. This is what d3's
  // `tickIncrement` does, and it is the difference between an axis that hits the
  // requested tick count and one that overshoots by half.
  const step =
    (normalized <= Math.SQRT2 ? 1 : normalized <= Math.sqrt(10) ? 2 : normalized <= Math.sqrt(50) ? 5 : 10) *
    magnitude;

  const start = Math.floor(low / step) * step;
  const end = Math.ceil(high / step) * step;
  // Accumulating `value += step` drifts: 0.1 seven times over is 0.7000000000001,
  // which then prints as a tick labelled 0.700 beside one labelled 0.800. Index
  // multiplication plus a decimal rounding keeps the label honest.
  const decimals = Math.max(0, -Math.floor(Math.log10(step)) + 1);
  const count = Math.round((end - start) / step);
  const ticks = Array.from({ length: count + 1 }, (_, index) =>
    Number((start + index * step).toFixed(decimals)),
  );
  return { min: start, max: end, step, ticks };
}

/**
 * A linear map from a data domain to a pixel range.
 *
 * A zero-width domain maps everything to the middle of the range rather than to
 * `NaN`, because one constant series should draw a flat line through the plot and
 * not vanish.
 */
export function linearScale(domainMin, domainMax, rangeMin, rangeMax) {
  const span = domainMax - domainMin;
  if (!Number.isFinite(span) || span === 0) {
    const middle = (rangeMin + rangeMax) / 2;
    return () => middle;
  }
  return (value) => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}

/** Round a number for a tick label without inventing precision. */
export function tickLabel(value, step) {
  const decimals = step >= 1 ? 0 : Math.min(6, Math.ceil(-Math.log10(step)));
  return value.toFixed(decimals);
}

/**
 * Size the canvas bitmap to its CSS box, once per size change.
 *
 * Two things here are not obvious. Assigning `canvas.width` clears the canvas and
 * resets its transform even if the value is unchanged, so the guard is not an
 * optimisation - it is what stops a resize storm from flashing the chart blank
 * every frame. And because the guard means the transform may survive from the
 * previous draw, the transform is set explicitly with `setTransform` rather than
 * accumulated with `scale`, which would compound the device pixel ratio on every
 * redraw and zoom the chart.
 *
 * Returns CSS pixels, so all drawing code works in layout units.
 */
export function prepareCanvas(canvas) {
  const ratio = globalThis.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const bitmapWidth = Math.round(width * ratio);
  const bitmapHeight = Math.round(height * ratio);

  const resized = canvas.width !== bitmapWidth || canvas.height !== bitmapHeight;
  if (resized) {
    canvas.width = bitmapWidth;
    canvas.height = bitmapHeight;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height, resized };
}

/**
 * The chart colours and fonts, read from CSS custom properties.
 *
 * This is the whole reason dark mode reaches the canvas. `getComputedStyle`
 * resolves whichever `:root` block currently applies, so a theme change plus a
 * redraw is sufficient and no colour is written twice.
 *
 * Read once per draw rather than per shape: `getComputedStyle` forces style
 * resolution, and calling it inside a loop over 99 threshold points is a
 * measurable stall.
 */
export function palette(element = globalThis.document?.documentElement) {
  const style = globalThis.getComputedStyle?.(element);
  const read = (name, fallback) => {
    const value = style?.getPropertyValue(name)?.trim();
    return value || fallback;
  };
  return {
    grid: read("--chart-grid", "#e6ece8"),
    axis: read("--chart-axis", "#93a29b"),
    ink: read("--chart-ink", "#3d4a44"),
    inkSoft: read("--chart-ink-soft", "#6f7d77"),
    reference: read("--chart-reference", "#a3b0aa"),
    series: [
      read("--chart-series-1", "#1f7a5a"),
      read("--chart-series-2", "#2f6fbe"),
      read("--chart-series-3", "#b45f1d"),
      read("--chart-series-4", "#7a4fa3"),
    ],
    bandOk: read("--chart-band-ok", "rgba(31,122,90,0.12)"),
    bandWarn: read("--chart-band-warn", "rgba(180,95,29,0.14)"),
    bandBad: read("--chart-band-bad", "rgba(165,42,42,0.14)"),
    font: read("--chart-font", "600 11px system-ui, sans-serif"),
    fontLabel: read("--chart-font-label", "700 12px system-ui, sans-serif"),
  };
}

/**
 * How much room the axis furniture needs, measured rather than assumed.
 *
 * Called before anything is drawn, because the plot rectangle depends on the
 * widest tick label and there is no way to know that without the font loaded and
 * the string in hand.
 */
export function axisPadding(ctx, { yLabels = [], y2Labels = [], xTitle = "", yTitle = "" }) {
  const widest = (labels) => labels.reduce((most, text) => Math.max(most, ctx.measureText(text).width), 0);
  // A line of text is about 1.35x its size; asking the context is more precise but
  // `fontBoundingBoxAscent` is not universally available, so this is the safe form.
  const lineHeight = 14;
  return {
    top: BASE_PADDING.top,
    right: BASE_PADDING.right + (y2Labels.length ? TICK + GAP + widest(y2Labels) : 0),
    bottom: BASE_PADDING.bottom + lineHeight + TICK + GAP + (xTitle ? lineHeight + 2 : 0),
    left: BASE_PADDING.left + TICK + GAP + widest(yLabels) + (yTitle ? lineHeight + 4 : 0),
  };
}

/**
 * Draw the frame: grid, axes, ticks, tick labels, axis titles.
 *
 * Returns the scales and the plot rectangle, so a caller draws data in data
 * units and never touches a pixel offset. Everything downstream of this - lines,
 * points, bars, the legend - goes through the returned `toX`/`toY`.
 *
 * `x` and `y` are `{min, max, step, ticks}` from `niceTicks`, plus an optional
 * `format`. `y2` adds a right-hand axis for a second unit, which the threshold
 * chart needs: cost and approval rate share an x and have nothing else in common,
 * and plotting them on one axis - as the old chart did, by dividing cost by its
 * own maximum - makes the crossing point meaningless.
 */
export function drawFrame(ctx, { width, height, x, y, y2 = null, xTitle = "", yTitle = "", y2Title = "", colors }) {
  ctx.font = colors.font;
  ctx.textBaseline = "alphabetic";

  const format = (axis) => (value) => (axis.format ? axis.format(value) : tickLabel(value, axis.step));
  const yLabels = y.ticks.map(format(y));
  const y2Labels = y2 ? y2.ticks.map(format(y2)) : [];
  const pad = axisPadding(ctx, { yLabels, y2Labels, xTitle, yTitle });

  const plot = {
    left: pad.left,
    top: pad.top,
    right: width - pad.right,
    bottom: height - pad.bottom,
  };
  // A canvas narrower than its own axis furniture is possible at 320px with a long
  // label. Refuse rather than draw a plot with a negative width, which silently
  // mirrors every shape.
  if (plot.right <= plot.left || plot.bottom <= plot.top) return null;

  const toX = linearScale(x.min, x.max, plot.left, plot.right);
  const toY = linearScale(y.min, y.max, plot.bottom, plot.top);
  const toY2 = y2 ? linearScale(y2.min, y2.max, plot.bottom, plot.top) : null;

  // Horizontal grid lines only. Vertical ones as well would double the ink for no
  // extra information on charts whose x axis is already a labelled continuum.
  ctx.lineWidth = 1;
  ctx.strokeStyle = colors.grid;
  ctx.beginPath();
  for (const tick of y.ticks) {
    // The half-pixel offset is what makes a 1px line one crisp pixel instead of
    // two grey ones: canvas coordinates address the boundaries between pixels.
    const at = Math.round(toY(tick)) + 0.5;
    ctx.moveTo(plot.left, at);
    ctx.lineTo(plot.right, at);
  }
  ctx.stroke();

  ctx.strokeStyle = colors.axis;
  ctx.fillStyle = colors.inkSoft;
  ctx.beginPath();
  ctx.moveTo(plot.left + 0.5, plot.top);
  ctx.lineTo(plot.left + 0.5, plot.bottom + 0.5);
  ctx.lineTo(plot.right, plot.bottom + 0.5);
  if (y2) {
    ctx.moveTo(plot.right - 0.5, plot.top);
    ctx.lineTo(plot.right - 0.5, plot.bottom);
  }
  ctx.stroke();

  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  y.ticks.forEach((tick, index) => {
    const at = toY(tick);
    ctx.beginPath();
    ctx.moveTo(plot.left - TICK, Math.round(at) + 0.5);
    ctx.lineTo(plot.left, Math.round(at) + 0.5);
    ctx.stroke();
    ctx.fillText(yLabels[index], plot.left - TICK - GAP, at);
  });

  if (y2) {
    ctx.textAlign = "left";
    y2.ticks.forEach((tick, index) => {
      const at = toY2(tick);
      ctx.fillText(y2Labels[index], plot.right + TICK + GAP, at);
    });
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const tick of x.ticks) {
    const at = toX(tick);
    // Ticks outside the plot happen when the domain was rounded out past the
    // canvas edge; drawing them would put a label under the y axis.
    if (at < plot.left - 1 || at > plot.right + 1) continue;
    ctx.beginPath();
    ctx.strokeStyle = colors.axis;
    ctx.moveTo(Math.round(at) + 0.5, plot.bottom);
    ctx.lineTo(Math.round(at) + 0.5, plot.bottom + TICK);
    ctx.stroke();
    ctx.fillText(format(x)(tick), at, plot.bottom + TICK + GAP);
  }

  ctx.fillStyle = colors.ink;
  ctx.font = colors.fontLabel;
  if (xTitle) {
    // Centred on the plot, not on the canvas: the left padding is larger than the
    // right, so canvas-centred reads as off-centre.
    ctx.fillText(xTitle, (plot.left + plot.right) / 2, height - 14);
  }
  if (yTitle) {
    ctx.save();
    ctx.translate(12, (plot.top + plot.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "top";
    ctx.fillText(yTitle, 0, 0);
    ctx.restore();
  }
  if (y2Title) {
    ctx.save();
    ctx.translate(width - 4, (plot.top + plot.bottom) / 2);
    ctx.rotate(Math.PI / 2);
    ctx.textBaseline = "top";
    ctx.fillText(y2Title, 0, 0);
    ctx.restore();
  }

  return { plot, toX, toY, toY2 };
}

/** A polyline through data-space points. Points outside the domain are clipped. */
export function drawLine(ctx, frame, points, { color, width = 2, dash = null, toY = null } = {}) {
  if (points.length === 0) return;
  const mapY = toY || frame.toY;
  ctx.save();
  // Clipped to the plot so a series that leaves the domain does not draw over the
  // axis labels.
  ctx.beginPath();
  ctx.rect(frame.plot.left, frame.plot.top, frame.plot.right - frame.plot.left, frame.plot.bottom - frame.plot.top);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  points.forEach(([px, py], index) => {
    const at = [frame.toX(px), mapY(py)];
    if (index === 0) ctx.moveTo(at[0], at[1]);
    else ctx.lineTo(at[0], at[1]);
  });
  ctx.stroke();
  ctx.restore();
}

/** Filled circles at data-space points. */
export function drawPoints(ctx, frame, points, { color, radius = 3.5, toY = null } = {}) {
  const mapY = toY || frame.toY;
  ctx.fillStyle = color;
  for (const [px, py] of points) {
    const cx = frame.toX(px);
    const cy = mapY(py);
    if (cx < frame.plot.left - radius || cx > frame.plot.right + radius) continue;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** A vertical marker with a label, for "the threshold this run selected". */
export function drawMarker(ctx, frame, value, { color, label = "", colors }) {
  const at = Math.round(frame.toX(value)) + 0.5;
  if (at < frame.plot.left || at > frame.plot.right) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(at, frame.plot.top);
  ctx.lineTo(at, frame.plot.bottom);
  ctx.stroke();
  ctx.setLineDash([]);
  if (label) {
    ctx.font = colors.font;
    ctx.fillStyle = color;
    // Flip the label to the other side of the line when it would overflow the
    // plot, which is what happens whenever the marker lands near the right edge.
    const width = ctx.measureText(label).width;
    const flip = at + GAP + width > frame.plot.right;
    ctx.textAlign = flip ? "right" : "left";
    ctx.textBaseline = "top";
    ctx.fillText(label, flip ? at - GAP : at + GAP, frame.plot.top + 2);
  }
  ctx.restore();
}

/**
 * Vertical bars over a categorical x, grouped when there is more than one series.
 *
 * Bar geometry is computed rather than fixed: a 30-vintage chart and a 3-vintage
 * chart both fill their plot. `slot` is the width per category, and the group of
 * bars is centred in it with a gap either side.
 */
export function drawBars(ctx, frame, categories, series, { gap = 0.28 } = {}) {
  const slot = (frame.plot.right - frame.plot.left) / Math.max(1, categories.length);
  const groupWidth = slot * (1 - gap);
  const barWidth = Math.max(1, groupWidth / series.length);
  categories.forEach((_category, index) => {
    const groupLeft = frame.plot.left + slot * index + (slot - groupWidth) / 2;
    series.forEach((entry, seriesIndex) => {
      const value = entry.values[index];
      if (!Number.isFinite(value)) return;
      const top = frame.toY(value);
      const base = frame.toY(0);
      ctx.fillStyle = entry.color;
      ctx.fillRect(
        Math.round(groupLeft + barWidth * seriesIndex),
        Math.round(Math.min(top, base)),
        Math.max(1, Math.round(barWidth) - 1),
        Math.max(1, Math.abs(base - top)),
      );
    });
  });
  return { slot };
}

/**
 * Category labels under a bar chart, thinned until they fit.
 *
 * Rotating them would be the other answer; dropping every nth label keeps them
 * horizontal and readable, and a vintage axis reads fine with every other year
 * labelled. The step is derived from the measured widest label, so it adapts to
 * the container instead of to a guess.
 */
export function drawCategoryLabels(ctx, frame, categories, { colors, slot }) {
  ctx.font = colors.font;
  ctx.fillStyle = colors.inkSoft;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const widest = categories.reduce((most, text) => Math.max(most, ctx.measureText(String(text)).width), 0);
  const every = Math.max(1, Math.ceil((widest + GAP * 2) / Math.max(1, slot)));
  categories.forEach((category, index) => {
    if (index % every !== 0) return;
    ctx.fillText(String(category), frame.plot.left + slot * (index + 0.5), frame.plot.bottom + TICK + GAP);
  });
}

/**
 * A legend inside the plot, top-aligned, laid out by measurement and wrapped.
 *
 * Inside rather than below, because a legend below competes with the x axis title
 * for the same strip and the plot loses height to furniture. A translucent
 * backing plate keeps it readable over a line that passes underneath.
 *
 * Wrapped, because "Closed loans only / Embargo applied" measures wider than the
 * plot at 390px. A single row there put the second swatch past the right axis and
 * the backing plate over the tick labels - which no unit test sees, because the
 * text is all still drawn and nothing overflows the *page*. One row per line that
 * fits keeps every entry inside the plot at any width a chart is drawn at.
 */
export function drawLegend(ctx, frame, entries, { colors, align = "left" } = {}) {
  if (entries.length === 0) return;
  ctx.font = colors.font;
  const swatch = 9;
  const itemGap = 14;
  const boxHeight = 18;
  const widths = entries.map((entry) => swatch + GAP + ctx.measureText(entry.label).width);
  // The plate is inset by GAP either side, so that much of the plot is unavailable
  // to the entries themselves.
  const available = frame.plot.right - frame.plot.left - GAP * 4;

  // Greedy wrap: an entry starts a new row when adding it would pass the plot edge.
  // A single entry wider than the whole plot still gets its own row rather than
  // being dropped, because a legend missing a series is worse than one that
  // touches the axis.
  const rows = [];
  let current = { indexes: [], width: 0 };
  entries.forEach((_entry, index) => {
    const extra = widths[index] + (current.indexes.length ? itemGap : 0);
    if (current.indexes.length && current.width + extra > available) {
      rows.push(current);
      current = { indexes: [], width: 0 };
    }
    current.indexes.push(index);
    current.width += current.indexes.length === 1 ? widths[index] : extra;
  });
  rows.push(current);

  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  rows.forEach((row, rowIndex) => {
    const top = frame.plot.top + 4 + rowIndex * boxHeight;
    const start =
      align === "right"
        ? frame.plot.right - row.width - GAP * 2
        : frame.plot.left + GAP + GAP;
    ctx.save();
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = colors.grid;
    ctx.fillRect(start - GAP, top, row.width + GAP * 2, boxHeight);
    ctx.restore();

    const middle = top + boxHeight / 2;
    let cursor = start;
    for (const index of row.indexes) {
      const entry = entries[index];
      ctx.fillStyle = entry.color;
      if (entry.dash) {
        ctx.strokeStyle = entry.color;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(cursor, middle);
        ctx.lineTo(cursor + swatch, middle);
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        ctx.fillRect(cursor, middle - swatch / 2, swatch, swatch);
      }
      ctx.fillStyle = colors.ink;
      ctx.fillText(entry.label, cursor + swatch + GAP, middle);
      cursor += widths[index] + itemGap;
    }
  });
}

/**
 * The message a chart shows when it has nothing to draw.
 *
 * A dedicated function because "no data" and "failed to load" must look
 * different from an empty plot: a blank canvas with axes reads as a measurement
 * of zero, which is the same class of lie as formatting `null` as `0.000`.
 */
export function drawEmpty(ctx, width, height, message, colors) {
  ctx.font = colors.fontLabel;
  ctx.fillStyle = colors.inkSoft;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, width / 2, height / 2);
}

/**
 * Coalesce redraws to one per frame, and only on a real size change.
 *
 * The old dashboard listened for `resize` and refetched two endpoints per event,
 * so dragging a window edge issued hundreds of requests. `ResizeObserver` is the
 * right event - it fires when the *element* changes, which also covers a sidebar
 * opening and a font loading - and `requestAnimationFrame` collapses a burst into
 * one draw. The redraw callback takes no arguments because it reads cached data;
 * nothing here can fetch.
 */
export function observeResize(elements, redraw) {
  let queued = false;
  const observer = new globalThis.ResizeObserver(() => {
    if (queued) return;
    queued = true;
    globalThis.requestAnimationFrame(() => {
      queued = false;
      redraw();
    });
  });
  for (const element of elements) if (element) observer.observe(element);
  return observer;
}
