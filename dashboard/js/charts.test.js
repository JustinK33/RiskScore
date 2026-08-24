/**
 * `node --test dashboard/js/` against a fake 2d context.
 *
 * A canvas cannot be asserted on by looking at it, so the tests take the two
 * routes that are actually available: the geometry functions are pure and are
 * checked against hand-computed values, and the drawing functions are given a
 * recording context so the calls they make can be inspected. That covers the
 * things that were wrong in the previous chart code - constant paddings,
 * `i / 5` ticks, hardcoded text offsets, a bitmap reassigned every frame - which
 * are all decisions made before a single pixel is written.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  axisPadding,
  drawFrame,
  drawLegend,
  linearScale,
  niceTicks,
  palette,
  prepareCanvas,
  tickLabel,
} from "./charts.js";

/**
 * A 2d context that records instead of drawing.
 *
 * `measureText` returns 6px per character - not the real font metric, but the
 * property under test is that the code *asks*, and that a wider label produces a
 * wider padding. A fixed width per character makes that assertable.
 */
function fakeContext() {
  const calls = [];
  const record =
    (name) =>
    (...args) =>
      calls.push({ name, args });
  return {
    calls,
    texts: () => calls.filter((call) => call.name === "fillText").map((call) => call.args[0]),
    font: "",
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineJoin: "",
    textAlign: "",
    textBaseline: "",
    globalAlpha: 1,
    measureText: (text) => ({ width: String(text).length * 6 }),
    save: record("save"),
    restore: record("restore"),
    translate: record("translate"),
    rotate: record("rotate"),
    beginPath: record("beginPath"),
    moveTo: record("moveTo"),
    lineTo: record("lineTo"),
    rect: record("rect"),
    arc: record("arc"),
    clip: record("clip"),
    stroke: record("stroke"),
    fill: record("fill"),
    fillRect: record("fillRect"),
    fillText: record("fillText"),
    clearRect: record("clearRect"),
    setLineDash: record("setLineDash"),
    setTransform: record("setTransform"),
  };
}

/** The palette fallbacks, which is what `palette()` returns with no document. */
const COLORS = palette(undefined);

// --- 1. ticks ---

test("ticks are 1-2-5 steps, not the data range divided by five", () => {
  // The old cost axis printed `Math.round(maxCost * i / 5)`, giving ticks at 2,496
  // and 4,992 for a maximum of 12,480. Nobody reads those.
  const { ticks, step } = niceTicks(0, 12480);
  assert.equal(step, 2000);
  assert.deepEqual(ticks, [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000]);
});

test("the domain is rounded outwards so the data never touches the frame", () => {
  const { min, max } = niceTicks(0.03, 0.83);
  assert.ok(min <= 0.03 && max >= 0.83);
  assert.equal(min, 0);
  assert.equal(max, 1);
});

test("tick labels do not drift with floating point", () => {
  // Accumulating `value += 0.1` gives 0.7000000000000001 on the eighth tick, which
  // prints beside a neighbour labelled 0.800 and looks like a rendering bug.
  const { ticks } = niceTicks(0, 1);
  assert.deepEqual(ticks, [0, 0.2, 0.4, 0.6, 0.8, 1]);
  for (const value of ticks) assert.equal(value, Number(value.toFixed(6)));
});

test("a constant series still gets an axis with width", () => {
  // Every vintage having the same default rate is a real payload, and a
  // zero-width domain would divide by zero.
  const { min, max, ticks } = niceTicks(0.15, 0.15);
  assert.ok(max > min);
  assert.ok(ticks.length >= 2);
});

test("an all-zero column gets a unit domain rather than a zero-width one", () => {
  const { min, max } = niceTicks(0, 0);
  assert.ok(max > min);
});

test("a non-finite domain falls back to 0..1 instead of producing NaN ticks", () => {
  // `Math.max(...[])` is -Infinity, which is exactly what an empty column yields.
  const { ticks } = niceTicks(-Infinity, NaN);
  assert.deepEqual(ticks, [0, 0.25, 0.5, 0.75, 1]);
});

test("a reversed domain is accepted", () => {
  assert.deepEqual(niceTicks(1, 0).ticks, niceTicks(0, 1).ticks);
});

test("tick labels carry the precision of their step and no more", () => {
  assert.equal(tickLabel(2000, 2000), "2000");
  assert.equal(tickLabel(0.2, 0.2), "0.2");
  assert.equal(tickLabel(0.05, 0.05), "0.05");
});

// --- 2. scales ---

test("a scale maps the domain onto the range, inverted for y", () => {
  const toY = linearScale(0, 1, 400, 0);
  assert.equal(toY(0), 400);
  assert.equal(toY(1), 0);
  assert.equal(toY(0.25), 300);
});

test("a zero-width domain maps to the middle of the range, not to NaN", () => {
  const toY = linearScale(5, 5, 400, 0);
  assert.equal(toY(5), 200);
  assert.ok(Number.isFinite(toY(999)));
});

// --- 3. measured padding ---

test("padding grows with the widest label it will actually draw", () => {
  // The constant 56px left padding was correct for "0.0" and overlapped the plot
  // for "12,480".
  const ctx = fakeContext();
  const narrow = axisPadding(ctx, { yLabels: ["0", "1"] });
  const wide = axisPadding(ctx, { yLabels: ["0", "12,480"] });
  assert.ok(wide.left > narrow.left);
  assert.equal(wide.left - narrow.left, 6 * ("12,480".length - 1));
});

test("a right-hand axis is only paid for when there is one", () => {
  const ctx = fakeContext();
  const single = axisPadding(ctx, { yLabels: ["0"] });
  const dual = axisPadding(ctx, { yLabels: ["0"], y2Labels: ["100%"] });
  assert.ok(dual.right > single.right);
});

test("an axis title reserves a line for itself", () => {
  const ctx = fakeContext();
  const bare = axisPadding(ctx, { yLabels: ["0"] });
  const titled = axisPadding(ctx, { yLabels: ["0"], xTitle: "Threshold", yTitle: "Cost" });
  assert.ok(titled.bottom > bare.bottom);
  assert.ok(titled.left > bare.left);
});

// --- 4. the frame ---

const AXIS = niceTicks(0, 1);

test("the frame maps the domain onto the measured plot rectangle", () => {
  const ctx = fakeContext();
  const frame = drawFrame(ctx, {
    width: 800,
    height: 400,
    x: AXIS,
    y: AXIS,
    xTitle: "Predicted probability",
    yTitle: "Observed default rate",
    colors: COLORS,
  });

  assert.ok(frame.plot.left > 0 && frame.plot.right < 800);
  assert.equal(frame.toX(0), frame.plot.left);
  assert.equal(frame.toX(1), frame.plot.right);
  assert.equal(frame.toY(0), frame.plot.bottom);
  assert.equal(frame.toY(1), frame.plot.top);
});

test("axis titles are centred on the plot and drawn with textAlign, not an offset", () => {
  // `chartWidth / 2 - 64` was only ever right for one string in one font. The
  // assertion is that the title is drawn at the plot's centre with the alignment
  // doing the centring.
  const ctx = fakeContext();
  const frame = drawFrame(ctx, {
    width: 800,
    height: 400,
    x: AXIS,
    y: AXIS,
    xTitle: "Predicted probability",
    colors: COLORS,
  });

  const title = ctx.calls.find(
    (call) => call.name === "fillText" && call.args[0] === "Predicted probability",
  );
  assert.ok(title, "the x axis title must be drawn");
  assert.equal(title.args[1], (frame.plot.left + frame.plot.right) / 2);
  assert.equal(ctx.textAlign, "center");
});

test("every y tick gets a label", () => {
  const ctx = fakeContext();
  drawFrame(ctx, { width: 800, height: 400, x: AXIS, y: AXIS, colors: COLORS });
  for (const tick of AXIS.ticks) {
    assert.ok(ctx.texts().includes(tick.toFixed(1)), `missing label for ${tick}`);
  }
});

test("a canvas too small for its own axes refuses rather than drawing inside out", () => {
  // Reachable at 320px with a long y label: the padding exceeds the width, the
  // plot rectangle inverts, and every shape mirrors silently.
  const ctx = fakeContext();
  const frame = drawFrame(ctx, {
    width: 60,
    height: 40,
    x: AXIS,
    y: niceTicks(0, 1234567),
    yTitle: "Cost",
    colors: COLORS,
  });
  assert.equal(frame, null);
});

test("a second axis is drawn on the right with its own scale", () => {
  const ctx = fakeContext();
  const frame = drawFrame(ctx, {
    width: 800,
    height: 400,
    x: AXIS,
    y: niceTicks(0, 12480),
    y2: AXIS,
    colors: COLORS,
  });
  // Cost and approval rate share an x and nothing else. The old chart divided cost
  // by its own maximum to fit one axis, which made the crossing point meaningless.
  assert.notEqual(frame.toY2, null);
  assert.equal(frame.toY2(1), frame.plot.top);
  assert.notEqual(frame.toY(1), frame.plot.top);
});

// --- 5. the legend ---

test("legend items are laid out by measurement, left to right", () => {
  const ctx = fakeContext();
  const frame = drawFrame(ctx, { width: 800, height: 400, x: AXIS, y: AXIS, colors: COLORS });
  const before = ctx.texts().length;
  drawLegend(ctx, frame, [{ label: "Cost", color: "#000" }, { label: "Approval rate", color: "#111" }], {
    colors: COLORS,
  });
  const drawn = ctx.texts().slice(before);
  assert.deepEqual(drawn, ["Cost", "Approval rate"]);

  const positions = ctx.calls
    .filter((call) => call.name === "fillText" && drawn.includes(call.args[0]))
    .map((call) => call.args[1]);
  assert.ok(positions[1] > positions[0], "the second item must start after the first");
});

// --- 6. the bitmap ---

/** A canvas-alike whose CSS box is fixed and whose bitmap is observable. */
function fakeCanvas(cssWidth, cssHeight) {
  const ctx = fakeContext();
  return {
    width: 0,
    height: 0,
    getContext: () => ctx,
    getBoundingClientRect: () => ({ width: cssWidth, height: cssHeight }),
    ctx,
  };
}

test("the bitmap is only reassigned when the target size changed", () => {
  // Assigning `canvas.width` clears the canvas even when the value is unchanged,
  // so the old code's unconditional assignment flashed the chart blank on every
  // frame of a resize drag.
  const canvas = fakeCanvas(800, 400);
  const first = prepareCanvas(canvas);
  assert.equal(first.resized, true);
  assert.equal(canvas.width, 800 * (globalThis.devicePixelRatio || 1));

  const second = prepareCanvas(canvas);
  assert.equal(second.resized, false);
});

test("the transform is set rather than accumulated", () => {
  // `ctx.scale(ratio, ratio)` compounds when the bitmap was not reassigned, which
  // would zoom the chart a little further on every redraw.
  const canvas = fakeCanvas(800, 400);
  prepareCanvas(canvas);
  prepareCanvas(canvas);
  const transforms = canvas.ctx.calls.filter((call) => call.name === "setTransform");
  assert.equal(transforms.length, 2);
  assert.deepEqual(transforms[0].args, transforms[1].args);
});

test("prepareCanvas reports css pixels, so drawing code works in layout units", () => {
  const canvas = fakeCanvas(640, 320);
  const { width, height } = prepareCanvas(canvas);
  assert.equal(width, 640);
  assert.equal(height, 320);
});

test("a zero-size canvas does not produce a zero-size bitmap", () => {
  // A canvas inside a `display: none` panel measures 0x0, and a 0-width bitmap
  // throws in some browsers when drawn to.
  const canvas = fakeCanvas(0, 0);
  prepareCanvas(canvas);
  assert.ok(canvas.width >= 1);
  assert.ok(canvas.height >= 1);
});

// --- 7. the palette ---

test("the palette falls back to readable colours with no document", () => {
  // Server-side or in a test there is no `getComputedStyle`; a palette of empty
  // strings would draw nothing at all and look like a data problem.
  assert.match(COLORS.grid, /^#|rgb/);
  assert.equal(COLORS.series.length, 4);
  for (const color of COLORS.series) assert.match(color, /^#|rgb/);
});
