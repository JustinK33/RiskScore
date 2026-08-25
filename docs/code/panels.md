# `dashboard/js/panels.js`

## Purpose

Everything that knows what a credit-risk report *means*.

`charts.js` draws pixels and `format.js` formats numbers; neither has heard of a PSI band, a lender-priced tier, or an outcome-maturity embargo.
This file is where the vocabulary lives, and it is the only file where a decision about how to read a metric can be made.

It has one shape, repeated eleven times: **every renderer returns `{label, table}`.**

- `label` becomes the canvas's `aria-label`, and it states the *headline reading* - "Validation calibration: 10 bins, mean predicted 0.148 against observed 0.151" - not a description of a picture. A screen reader user gets the conclusion.
- `table` is the accessible alternative, a real `<table>` from `dom.js`, mounted in a `<details>` beside the chart.

Both come from the same payload in the same function, so a chart and its table cannot disagree.
That is not a convenience: the old dashboard's two charts had `aria-label="Calibration curve"` and no table at all, which is a picture with a caption saying it is a picture.

The other rule: **direction is never decided here.** Whether a lower value of a metric is better comes from the payload's own `metrics` map, produced by `reporting.py`.
A client-side list of "lower is better" metrics would be a second source of truth that silently disagrees with the model card.

## Public API

| Name | What it is |
| --- | --- |
| `drawCalibration(canvas, payload)` | Reliability curves for both partitions against a 45-degree diagonal. |
| `drawThresholdCosts(canvas, payload, {selectedThreshold})` | Cost and approval rate against threshold, two y axes. Validation only. |
| `embargoRows(embargo)` | **Pure.** Before/after default rate per vintage, with `dropped` marked. |
| `drawEmbargo(canvas, manifest)` | Grouped bars from `embargoRows`. |
| `drawVintages(canvas, payload)` | Per-vintage metrics, tagged by partition. |
| `drawScorePsi(canvas, payload, {reference, comparison})` | Score-distribution PSI by bucket. |
| `featurePsiTable(payload, opts)` | Per-feature PSI with band colouring. |
| `importanceRows(payload)` | **Pure.** Bar widths relative to the largest magnitude. |
| `featureImportanceTable(payload)` | The importance table, bars as DOM nodes. |
| `variantName(variant)` | `"Logistic Regression, origination only"`. |
| `comparisonRows(payload)` | **Pure.** Variant rows with per-metric deltas against the named baseline. |
| `comparisonTable(payload)` | The comparison table. |

The five exports marked pure take a payload and return plain data.
They are pure because they are where the reasoning is, and reasoning that needs a canvas to test does not get tested.

## Inputs and outputs

Takes a `<canvas>` (or nothing, for the table-only exports) and a report payload as the API sends it.
Returns `{label, table}` from every renderer, plain arrays from the pure ones.
Imports `charts.js`, `dom.js`, and `format.js`.
Does not fetch, does not own state, and does not decide which run is displayed.

Internal helpers worth knowing:
- `begin(canvas)` - `prepareCanvas` plus `palette`, the two calls every renderer opens with.
- `nothing(canvas, message)` - `drawEmpty` plus a `{label, table}` for a panel with no payload, so an absent report still produces an accessible caption.
- `ORDINAL_X = {min: 0, max: 1, step: 1, ticks: []}` - the x spec for a categorical axis: a domain with no ticks, because `drawCategoryLabels` places the labels instead.
- `slotCentre(index, total) = (index + 0.5) / total` - the fractional x of a category's centre, which is what makes bars and their labels agree without either knowing the plot width.
- `PARTITION_TAGS = {train: "train", validation: "val", test: "test"}` - the abbreviations the vintage axis has room for.
- `MISSING_BUCKET = "__missing__"` - the score-PSI bucket for rows with no score.
- `METRIC_LABELS` and `METRIC_FORMATS` - display name and formatter per metric, with `labelize` as the fallback.

## Invariants and failure modes

**Both calibration partitions share one domain, so the diagonal is at 45 degrees.**
The reliability diagonal is only meaningful if the x and y scales are equal.
Computing a domain per axis - or per partition - tilts it, and a tilted diagonal makes an over-confident model look calibrated.
Validation and test are drawn on the same axes for the same reason: the comparison between them *is* the reading.

**`drawThresholdCosts` plots validation only (audit B04).**
The threshold was selected on validation.
A cost curve drawn from test scores would show a minimum at a different threshold than the one the bundle carries, and a reader would reasonably conclude the threshold was chosen badly.
The selected threshold is drawn as a marker so the choice is visible on the curve it was made from.

**Two y axes, because cost and approval rate are not the same quantity.**
Cost is unitless and unbounded; approval rate is a fraction.
On one axis the approval-rate line is flat against the bottom.

**A vintage the embargo removed entirely is marked `dropped`, not treated as zero.**
`embargoRows` compares `before` against `after`.
A vintage present in `before` and absent from `after` is `{dropped: true}` with a null rate, and `drawEmbargo` plots `NaN` for it rather than `null` - because `NaN` breaks the line and leaves a gap, while a chart library reading `null` as 0 would draw a vintage with a 0% default rate.
A vintage with a 0% default rate is a claim; a vintage with no mature loans is an absence, and the embargo panel exists precisely to show the difference.

**A vintage present only *after* the embargo is not marked dropped.**
Not a real case today, but the guard is one line and its absence would make the flag mean "differs" rather than "removed".

**The change is the corrected rate minus the biased one, in that order.**
The whole point of the panel is that the uncorrected rate is *too high* for immature vintages, so a negative change is the correction working.
Subtracting the other way would show the honest number as the regression.

**Vintages are ordered by year, not by object key insertion order.**
`Object.keys` on a JSON object preserves insertion order for string keys, which is the server's serialization order, which is not guaranteed to be sorted.
A vintage chart with 2015 before 2014 is a chart that says nothing.

**A non-numeric rate is dropped rather than plotted.**
Via `numericColumn`, which returns the count of what it dropped so the caption can say so.

**The AUC axis starts at 0.5, not 0.**
0.5 is the no-skill floor.
An axis from 0 spends half its height on a region no model occupies and compresses the range where the differences are.

**Score PSI is summed from the payload's own per-bucket contributions.**
Not recomputed from the two distributions.
`drift.py` decides the bucket edges, the missingness handling, and the epsilon on an empty bucket; a client-side re-derivation would disagree in the third decimal place and there would be no way to tell which was right.
`MISSING_BUCKET` is a bucket, not a filter - a shift in *how often the model can score at all* is drift, and dropping it would hide the most severe kind.

**Importance bars are relative to the largest magnitude, not to their sum.**
A share-of-total bar makes the top feature's bar shrink as unrelated features are added.
Relative-to-max keeps the top bar full width and every other bar a readable fraction of it.

**An exact zero magnitude gets no bar at all**, and a missing magnitude is `null` rather than 0. A feature the model gave no weight and a feature whose importance was not computed are different facts, and a 1px bar for the first would read as a small weight.

**The signed mean is kept separate from the magnitude.**
Mean absolute SHAP is how much a feature moves the score; mean signed SHAP is which way.
Collapsing them loses the direction, and reporting only the signed mean makes a feature that pushes hard both ways look irrelevant.

**Feature importance is a DOM table with CSS bars, not a canvas.**
Roughly 30 rows of labelled horizontal bars is a table, and a table gets text selection, `overflow-wrap` on long feature names, and a screen-reader reading for free.
Drawing it on a canvas would have needed a scroll region inside a bitmap.

**Comparison direction comes from the payload, always.**
`comparisonRows` reads each metric's declared direction out of the payload's `metrics` map.
A rise in a lower-is-better metric is a regression, a fall is an improvement, and a metric the payload marks neutral is never coloured at all - because colouring a neutral metric asserts a preference the model does not have.

**The baseline is the one the payload names.**
Matched by name, with position as the fallback.
Assuming the first row is the baseline breaks the moment the server changes its ordering, and the failure is silent - every delta computed against the wrong variant.

**The baseline row carries no deltas.**
A row of `+0.000` against itself is noise that reads as a measurement.

**A delta is written in its metric's own units.**
`METRIC_FORMATS` per metric, so an AUC delta is three decimals and a cost delta is a grouped integer.
One shared formatter would print `+1,234.000` for a cost and `+0` for an AUC change of 0.0004.

**Every renderer handles an absent payload and a refused frame.**
`nothing(canvas, message)` for no payload; `drawFrame` returning `null` at a width too small for its own axes for the second.
Both paths still return a `{label, table}`, so the caption and the `<details>` are never left saying "loading".

## What must NOT live here

- **Pixel arithmetic.** Ticks, scales, padding, legend wrapping, text measurement all belong to `charts.js`. If a function here computes a coordinate from a width, the boundary has moved.
- **Fetching, or any knowledge of which run is active.** `main.js` passes payloads in.
- **Recomputing anything the server computed.** PSI totals, bucket edges, metric directions, embargo counts, SHAP magnitudes. Every one of those has an authoritative implementation in `src/risk_score/` with its own tests, and a second implementation here would be a second answer.
- **Colour literals.** Tones are token names; the palette arrives via `charts.js`'s `palette()`.
- **`innerHTML`.** Feature names come from a fitted preprocessor and metric labels from a payload.

## Related tests

`dashboard/js/panels.test.js`, 20 tests, all against the pure exports.
No canvas, no browser.

That split is deliberate: the drawing is verified in a real browser by `scripts/probe_dashboard.mjs`, and the *reasoning* is verified here, where it can be asserted exactly.

- Embargo, five tests: `a vintage the embargo removed entirely is marked, not treated as zero`, `the change is the corrected rate minus the biased one`, `vintages are ordered by year, not by key insertion`, `a vintage present only after the embargo is not marked as dropped`, `no embargo record yields no rows rather than throwing`, and `a non-numeric rate is dropped rather than plotted as NaN`.
- Importance, five tests: `bar widths are relative to the largest magnitude, not to their sum`, `a magnitude too small to see gets the 2% floor`, `a feature with exactly zero weight gets no bar at all`, `the signed mean is kept separate from the magnitude`, `a missing magnitude is null rather than zero, and draws no bar`, and `no SHAP summary yields no rows rather than throwing`.
- Comparison, six tests, and these are the ones that matter most because a wrongly-coloured delta is a wrong conclusion: `a lower score is an improvement when the payload says lower is better`, `a rise in a lower-is-better metric is a regression`, `a metric the payload marks neutral is never coloured`, `the baseline row carries no deltas at all`, `the baseline is the one the payload names, not the first row`, `a delta is written in its metric's own units`, `no comparison yields no rows rather than throwing`, and `a metric with no label falls back to a readable one`.

Every one of the "no payload yields no rows rather than throwing" tests exists because `main.js` renders from `Promise.allSettled` and a partly-available run must still paint.
A renderer that throws on an absent report takes the ten panels after it down with it.

The browser-side proof, per width and theme: `featurePsiRows >= 2` and `importanceRows >= 2` (one row is the empty placeholder, so anything less is a headed table with no body), `comparisonRows >= 1` with `comparisonDeltas >= 1` whenever there is more than one variant, `importanceBars >= 2` measured at a real width of at least 1px above 640px and exactly 0 below it, every `.data-fallback table` non-empty, and every canvas carrying an `aria-label` that is no longer the loading placeholder.

## Known limits

- **The drawing halves are not unit tested.** `drawCalibration` and the other five canvas renderers are covered only by the probe, which needs Chrome and a running server and is not in CI. A CSS or drawing regression that does not change a pure function's output would pass CI. Accepted; the alternative is a screenshot baseline.
- **`METRIC_LABELS` and `METRIC_FORMATS` are hand-maintained.** A new metric renders with `labelize` and three decimals. Visible on the page, harmless, and the fallback is the reason it is not enforced.
- **One baseline per comparison.** `comparisonRows` computes deltas against a single named variant. A three-way comparison where each variant should be read against a different reference would need a different shape.
- **No drill-down anywhere.** A PSI row cannot be expanded to its bucket histogram, and a vintage bar cannot be clicked through to its rows. The artifact CSVs are linked from the run panel for that.
- **`slotCentre` assumes categories are evenly spaced.** Vintages are quarters, which they are - but a payload with an irregular time axis would be drawn as though it were regular. The x labels would still be correct, so the misreading would be about spacing, not values.
