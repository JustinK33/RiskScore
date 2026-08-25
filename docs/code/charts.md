# `dashboard/js/charts.js`

## Purpose

Draw on a canvas correctly. Pixels, ticks, scales and fonts - nothing about credit risk.

The old dashboard's charts were wrong in five specific ways, and each one is a rule in this file.

**Nothing was measured.** Text was positioned with hardcoded half-widths - `chartWidth / 2 - 64`, `+62`, `-28`, `-52` - tuned by eye against `Inter`, a font the page requested in five places and never actually loaded. So every label was offset by the difference between Inter's metrics and the system stack's, on every machine, permanently. Here every piece of text is placed with `measureText` and `textAlign`, and the font is the system stack.

**Left padding was constant.** A y axis labelled `0.0` and one labelled `-12,500` got the same 12px gutter, so the second drew its labels off the left edge of the canvas. `axisPadding` measures the widest label the axis will actually draw and reserves exactly that.

**Ticks were `i / 5` over the data range.** That produces axes labelled `0.137, 0.274, 0.411`. `niceTicks` produces 1-2-5 steps and rounds the domain outward, so the data never touches the frame and the labels are numbers a reader can subtract.

**Colours were hex literals in drawing code.** Which meant dark mode changed the page and not the plots. `palette()` reads the `--chart-*` custom properties from the document element on every draw, so the theme reaches the canvas.

**`prepareCanvas` reassigned the bitmap on every draw.** Assigning `canvas.width` clears the canvas even when the value is unchanged, so a resize storm - or any redraw - produced a visible blank flash. The assignment is now guarded on the target size having actually changed.

## Public API

| Name | What it is |
| --- | --- |
| `niceTicks(min, max, target = 5)` | `{min, max, step, ticks, decimals}`. A 1-2-5 step, the domain rounded outward, and the tick values. |
| `linearScale(dMin, dMax, rMin, rMax)` | A function mapping the domain onto the range. |
| `tickLabel(value, step)` | A tick's text at the precision its step implies. |
| `prepareCanvas(canvas)` | `{ctx, width, height}` in **CSS pixels**, with the transform set for the device ratio. |
| `palette(element)` | The `--chart-*` custom properties, resolved, with readable fallbacks. |
| `axisPadding(ctx, {yLabels, y2Labels, xTitle, yTitle})` | The padding the measured furniture needs. |
| `drawFrame(ctx, spec)` | Axes, grid, ticks and titles. Returns `{plot, toX, toY, toY2}` or `null`. |
| `drawLine`, `drawPoints` | A series, clipped to the plot. |
| `drawMarker(ctx, frame, value, opts)` | A labelled vertical rule. |
| `drawBars(ctx, frame, categories, series, opts)` | Grouped bars, geometry derived from the category count. |
| `drawCategoryLabels(ctx, frame, categories, opts)` | Ordinal x labels, thinned by measurement. |
| `drawLegend(ctx, frame, entries, opts)` | Greedy-wrapped legend rows. |
| `drawEmpty(ctx, width, height, message, colors)` | The centred message a panel with no payload draws. |
| `observeResize(elements, redraw)` | `ResizeObserver` plus `requestAnimationFrame` coalescing. |
| `BASE_PADDING`, `TICK`, `GAP`, `TARGET_TICKS` | `{top: 18, right: 16, bottom: 16, left: 12}`, `5`, `6`, `5`. |

## Inputs and outputs

Numbers and a `CanvasRenderingContext2D` in; drawing, and geometry objects, out.

Imports nothing. No fetching, no DOM construction, no domain vocabulary.
`palette` is the one function that reads the document, and it degrades to hardcoded readable colours when there is no `document` at all - which is what lets it be tested under `node --test`.

**Everything is in CSS pixels.** `prepareCanvas` sets the bitmap to `size * devicePixelRatio` and then applies `setTransform(ratio, 0, 0, ratio, 0, 0)`, so every subsequent coordinate, font size, and line width in this file and in `panels.js` is a layout unit. Drawing code never sees the device ratio.

`drawFrame` returns the contract every panel draws against:
- `plot` - the rectangle inside the axes, as `{x, y, width, height, right, bottom}`.
- `toX`, `toY`, `toY2` - the scale functions for the left and optional right axes.

## Invariants and failure modes

**`niceTicks` steps at 1, 2, 5 and powers of ten, chosen at geometric midpoints.**
The multiplier is picked by comparing the raw step against `Math.SQRT2`, `Math.sqrt(10)`, and `Math.sqrt(50)` times the power of ten - which is d3's `tickIncrement`, transcribed. Arithmetic midpoints (1.5, 3.5, 7.5) look equivalent and put the boundary in the wrong place for a logarithmic quantity, giving a step of 2 where 1 is the better fit for a sixth of the cases.

**The domain is rounded outward, so the data never touches the frame.**
`min` floors to a multiple of the step and `max` ceils. A point drawn exactly on the axis is a point a reader cannot see and cannot read the value of.

**Tick labels are built by index multiplication and then `toFixed(decimals)`.**
Accumulating `value += step` drifts: five additions of 0.1 give `0.7000000000000001`, and that is what the axis would say. `min + i * step` then a fixed precision derived from the step's own magnitude gives `0.7`. `decimals` comes from the step, so a step of 0.05 labels two places and a step of 5 labels none - a tick carries the precision of its step and no more.

**A degenerate domain is still an axis with width.**
A constant series, an all-zero column, a `NaN` bound, and a reversed pair are four separate cases and each has its own test. A constant series gets a unit domain around its value; an all-zero column gets `0..1`; a non-finite bound falls back to `0..1` rather than producing `NaN` ticks that then draw nothing and log nothing; a reversed pair is accepted and ordered. The reason all four are handled rather than guarded against at the call site is that all four occur - a run with one vintage, a PSI column with no drift, a metric absent from a partial run.

**`linearScale` maps a zero-width domain to the middle of the range.**
`(v - min) / (max - min)` is `NaN` when the domain is a point, and `NaN` coordinates draw nothing at all - silently. The midpoint is the only defensible answer, and it is visible.

**`prepareCanvas` only assigns the bitmap when the target size changed, and uses `setTransform` not `scale`.**
The guard is what removes the blank flash on resize. `setTransform` rather than `scale` because `scale` *accumulates*: two draws without an intervening reset give a 2x-scaled chart, then 4x. There is no `save`/`restore` pairing to get wrong if the transform is set absolutely each time.

A zero-size canvas - one inside a `hidden` section - does not get a zero-size bitmap, because a zero-width bitmap throws on some operations and produces an unrecoverable context on others.

**`palette` reads custom properties once per draw, and never caches across draws.**
Caching would be the obvious optimization and would break the theme toggle, which changes `data-theme` and calls `redraw()` with no other signal. One `getComputedStyle` per chart per draw is five calls per frame in the worst case, which is not measurable against the drawing itself.

**`axisPadding` measures, and only pays for what is drawn.**
A right-hand axis costs nothing when there is no `y2`. An axis title reserves exactly one line. The widest y label decides the left gutter. This is the difference between a `-12,500` label sitting inside the canvas and sitting outside it.

**`drawFrame` returns `null` when the canvas is narrower than its own furniture.**
At 320px with a wide y axis, the plot rectangle's width goes negative, and drawing into a negative rectangle produces axes that cross and a series drawn inside out - which looks like data. Refusing and letting the caller draw the empty message is the only honest outcome. Every `panels.js` renderer checks for `null`.

**Grid lines are horizontal only, and 1px lines are offset by half a pixel.**
Vertical grid lines on charts whose x axis is ordinal (vintages, PSI buckets) would imply a continuum that is not there. The half-pixel offset is because a 1px line drawn on an integer coordinate straddles two device rows and renders as a 2px blur at ratio 1.

**`drawLine` clips to the plot rectangle.**
A series whose domain was computed from a different partition, or a point outside the rounded domain, would otherwise draw over the axis labels.

**`drawMarker` flips its label at the right edge.**
The selected-threshold marker sits wherever the threshold is, including at 0.98. A label anchored left at that position is drawn off the canvas.

**`drawBars` derives geometry from the category count, and `drawCategoryLabels` thins by measurement.**
Neither takes a hardcoded bar width or a hardcoded "every nth label". Twelve vintages at 1440px label all twelve; the same twelve at 320px label every third, decided by `measureText` against the available slot.

**`drawLegend` greedy-wraps rather than overflowing.**
A legend too wide for the plot wraps to a second row instead of spilling past the axis, and the wrap is computed from measured entry widths.

**`observeResize` coalesces with `requestAnimationFrame`.**
A drag-resize fires `ResizeObserver` dozens of times a second. Without coalescing that is dozens of full redraws per second; with it, one per frame. Combined with `api.js`'s cache, a resize storm issues zero network requests and draws once per frame.

## What must NOT live here

- **Anything about credit risk.** No metric names, no PSI bands, no partition names, no tier vocabulary. `panels.js` owns all of it. The test for whether a change belongs here is whether it would make sense in a different project.
- **Colour literals.** Every colour arrives through `palette()`. A hex in drawing code is invisible to the theme.
- **Fetching or caching payloads.** `api.js` owns the network; `main.js` owns state.
- **`document.createElement`.** The `<details>` fallback table beside each chart is built by `dom.js`. This file draws into a context it is handed.
- **The device pixel ratio, past `prepareCanvas`.** If a second function in this file multiplies by `devicePixelRatio`, the transform contract has been broken.

## Related tests

`dashboard/js/charts.test.js`, 25 tests - the largest suite in the dashboard, because this is where the arithmetic is.

The tests assert *exact* values, not shapes. `assert(ticks.length > 0)` is why the `i / 5` axes survived as long as they did.

- `ticks are 1-2-5 steps, not the data range divided by five` and `the domain is rounded outwards so the data never touches the frame` pin the two properties the old implementation lacked.
- `tick labels do not drift with floating point` is the `0.7000000000000001` case.
- `tick labels carry the precision of their step and no more`.
- Four degenerate-domain tests - `a constant series still gets an axis with width`, `an all-zero column gets a unit domain rather than a zero-width one`, `a non-finite domain falls back to 0..1 instead of producing NaN ticks`, `a reversed domain is accepted` - one per case that occurs in real payloads.
- `a scale maps the domain onto the range, inverted for y` and `a zero-width domain maps to the middle of the range, not to NaN`.
- `padding grows with the widest label it will actually draw`, `a right-hand axis is only paid for when there is one`, and `an axis title reserves a line for itself` cover `axisPadding` against a stub context whose `measureText` returns declared widths, so the assertions are on arithmetic rather than on a font.
- `the frame maps the domain onto the measured plot rectangle`, `axis titles are centred on the plot and drawn with textAlign, not an offset`, and `every y tick gets a label`.
- `a canvas too small for its own axes refuses rather than drawing inside out` is the `null` return.
- `a second axis is drawn on the right with its own scale`.
- `legend items are laid out by measurement, left to right` and `a legend too wide for the plot wraps instead of spilling past the axis`.
- `the bitmap is only reassigned when the target size changed` is the blank-flash fix, asserted by counting assignments through a proxy. `the transform is set rather than accumulated` is the `setTransform` rule. `prepareCanvas reports css pixels, so drawing code works in layout units` is the contract every other file depends on. `a zero-size canvas does not produce a zero-size bitmap`.
- `the palette falls back to readable colours with no document` is what makes the rest of the suite runnable outside a browser.

The end-to-end proof is `scripts/probe_dashboard.mjs`, which at six widths in two themes asserts every canvas has a non-zero bitmap, a CSS width of at least 100px, and an `aria-label` that is not still the loading placeholder.

## Known limits

- **No animation and no interaction.** No tooltips, no hover, no zoom, no pan. A canvas has no hit testing, so any of those would need a spatial index and a `mousemove` handler. The `<details>` table beside each chart is the exact-value affordance instead.
- **`niceTicks` is linear only.** No log scale. Nothing on this dashboard spans orders of magnitude; a log axis would be a new function, not a flag.
- **`drawBars` assumes bars fit.** Beyond roughly twenty categories the bars are sub-pixel. The vintage chart is bounded by the split windows, so this has not bitten, and the mitigation would be to draw the table instead - which already exists.
- **Text is measured, but the font is not waited for.** `measureText` reflects whatever font is resolved at draw time. This is correct here because the font is the system stack and needs no loading; adding a webfont would reintroduce the original bug in a subtler form, and would need a `document.fonts.ready` before the first draw.
- **`observeResize` coalesces to a frame, not to an idle callback.** A continuous drag redraws at 60fps. Measured at roughly 2ms per chart, so five charts fit in a frame; a sixth heavier chart would want a trailing-edge debounce.
