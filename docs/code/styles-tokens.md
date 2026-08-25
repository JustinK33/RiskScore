# `dashboard/styles/tokens.css`

## Purpose

Every value the dashboard uses, declared once.

Three rules govern this file, and each one exists because the old stylesheet broke it.

**1. No component styles here.** `app.css` may only *reference* these names. So a colour change is a one-line edit, not a grep across 1200 lines.

**2. Chart colours are custom properties like every other colour, because canvas has no cascade.** `charts.js` reads them with `getComputedStyle` and therefore follows the theme automatically. Hardcoded `#2f6fbe` inside drawing code is how the previous dashboard ended up with dark-mode charts that were unreadable on a dark page, and there was nowhere to fix it but the JS - which meant a second palette, maintained by hand, that could and did disagree with the first.

**3. Sizes the canvas has to agree with live here too.** The type scale, the plot padding, and the two font strings. `measureText` needs a font string, and it gets it from here, so an axis label measured in JS uses the same number the CSS laid out. Two scales that can disagree produce labels that are correctly measured against the wrong font - which is exactly the bug the old dashboard had, and it had it invisibly, because both numbers looked right in isolation.

## Public API

The file declares no selectors of its own beyond `:root` and the two dark-theme gates.
Its API is the custom-property names.

| Group | Names |
| --- | --- |
| Surfaces and ink | `--bg`, `--surface`, `--surface-soft`, `--surface-sunken`, `--text`, `--text-inverse`, `--muted`, `--line`, `--line-strong` |
| Semantic colour | `--accent`, `--accent-strong`, `--accent-soft`, `--blue`, `--ok`, `--warn`, `--warn-soft`, `--danger`, `--danger-soft`, `--focus` |
| Elevation | `--shadow-1`, `--shadow-2` |
| Charts | `--chart-plot-bg`, `--chart-grid`, `--chart-axis`, `--chart-ink`, `--chart-ink-soft`, `--chart-series-1`..`4`, `--chart-reference`, `--chart-band-ok`, `--chart-band-warn`, `--chart-band-bad` |
| Spacing | `--space-1`..`--space-8` = 4, 8, 12, 16, 20, 24, 32, 48px |
| Type | `--font-sans`, `--font-mono`, `--text-xs`..`--text-3xl` = 11, 12, 13, 15, 18, 22, 30, 34px, `--leading-tight`, `--leading` |
| Canvas fonts | `--chart-font`, `--chart-font-label` |
| Shape | `--radius-sm`, `--radius`, `--radius-lg`, `--border` |
| Layout | `--content-width` = 1240px |

The `--chart-*` names are the ones with a second consumer: `charts.js`'s `palette()` resolves them on every draw.
Renaming one silently falls back to that function's hardcoded readable default rather than failing, so a rename must be made in both files.

## Inputs and outputs

No imports.
Loaded by `index.html` as a plain `<link>` **before** `app.css` - not via `@import`, which would serialize the two requests behind each other.

Read by `app.css` through `var()`, and by `charts.js` through `getComputedStyle(document.documentElement)`.

`color-scheme` is declared per theme, which is what makes form controls, scrollbars, and the `<select>` popup follow - none of which a custom property can reach.

## Invariants and failure modes

**Light is the default palette.**
Declared on `:root` unconditionally, with dark applied on top.
So a stylesheet that fails to load halfway, or a `prefers-color-scheme` the browser does not report, degrades to legible dark-on-white rather than to black-on-black.

**Dark is declared twice, and the duplication is deliberate.**
Once as `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) }`, once as `:root[data-theme="dark"]`.
The media query is the OS preference; the attribute is an explicit choice.
The `:not([data-theme="light"])` gate is the load-bearing part: without it, a reader who picks light on a dark-mode machine gets dark anyway and the toggle looks broken.
The alternative - one block with a `:where()` selector list - was rejected because the two blocks answer different questions and a reader has to be able to see which is which.

**Dark is not the light palette inverted.**
Three specific differences.
Surfaces get *lighter* as they come forward, the opposite of light mode, because a raised surface on a dark page reads as closer when it is brighter.
Shadows are nearly invisible, so borders do the separating and `--line` is proportionally stronger.
The accent is lightened, because `#1f7a5a` on `#161a18` fails contrast for text - an inverted palette would have shipped an accent nobody could read.

**Chart series are ordered by role, not by preference.**
Series 1 is the subject of the chart, 2 the comparison, 3 the reference.
So `--chart-series-1` is the same *meaning* across five charts, and a reader who learns the calibration chart can read the vintage chart.
They are also chosen to be distinguishable in greyscale as well as in hue, because a colour-blind reader and a printed page are the same problem.

**`--chart-reference` is separate from the series colours.**
The calibration diagonal and the PSI action lines are not data; giving them a series colour would make them read as a fourth measured series.

**The spacing scale is 4px and has exactly eight steps.**
Every gap and pad in `app.css` is one of these eight numbers.
That is what stops the 13px/14px/18px drift the old stylesheet had - it had eleven distinct paddings, none of them intentional.

**The type scale is in px, not rem.**
Two reasons, and the first is decisive: the canvas font string has to be an absolute size, and a `rem`-based scale means JS would have to resolve the root font size and multiply, giving two computations that can disagree.
The cost is that a browser font-size preference is not honoured in this fixed-layout dashboard, which is a real accessibility trade-off - mitigated by page zoom working correctly, since every length including the breakpoints is in px and zoom scales the px.

**`--chart-font` and `--chart-font-label` are complete font shorthand strings, not sizes.**
`600 11px system-ui, ...`.
`canvas.font` takes a shorthand and nothing else, so a size alone would need JS to assemble a weight and a family - which is where a second font stack would creep in.

**The font stack is the system stack, with no webfont.**
The old dashboard requested `Inter` in five places and never loaded it, so every hardcoded text offset in the drawing code was tuned against metrics the page never had.
A system stack needs no loading, so `measureText` at first draw is correct.
Adding a webfont would reintroduce the original bug in a subtler form and would need a `document.fonts.ready` gate before the first draw.

**`--border` is a whole shorthand, not a width.**
`1px solid var(--line)`.
Nested `var()` resolves per theme, so one token carries both the geometry and the themed colour, and `app.css` never spells `solid` again.

**Every semantic colour has a soft variant where it is used as a background.**
`--warn`/`--warn-soft`, `--danger`/`--danger-soft`, `--accent`/`--accent-soft`.
A banner is soft-background with full-strength text; using the full strength as a background would fail contrast against the text on it in one theme or the other.

**`--focus` is its own token, not `--accent`.**
The focus ring has to be visible against every surface including the accent-coloured button, so it is blue in both themes rather than following the accent.

## What must NOT live here

- **Selectors for components.** No `.panel`, no `.metric`, no `.data-table`. This file has `:root` and two theme gates.
- **`@media` queries other than `prefers-color-scheme`.** The responsive breakpoints are in `app.css`, where the content that stops fitting is.
- **One-off values.** A colour used in exactly one rule still belongs here if it is a colour; a number that is genuinely geometry local to one component (a `320px` sidebar track) belongs in `app.css`.
- **A third palette.** If the charts need a colour, it is a `--chart-*` token, not a literal in `charts.js`.
- **`rem` or `em` in the type scale.** See above.

## Related tests

Nothing in `node --test` reads CSS, and a unit test of a custom property's value would restate the file.

What does verify it:

- `charts.test.js::the palette falls back to readable colours with no document` covers the other side of the contract - that a missing or renamed `--chart-*` token degrades to something legible rather than to `""`, which the canvas would treat as transparent and draw nothing.
- `scripts/probe_dashboard.mjs` runs all six widths in **both themes**, emulating `prefers-color-scheme` rather than writing `localStorage` - so it measures the OS-preference path, which is the one with no JS involved. It records `bg=` (the computed `body` background) in every output line, so a theme that failed to apply is visible as the light background appearing in a dark run. It also asserts every canvas has an `aria-label` and a real bitmap in both themes, which is what would fail if `palette()` returned nothing.
- The probe disables the HTTP cache per session, because the service sends an `ETag` with no `Cache-Control` and Chrome heuristically reuses a stylesheet it already has. That made an earlier run report a fixed CSS bug as still broken.

## Known limits

- **The dark block is duplicated verbatim, about 30 declarations twice.** A colour changed in one and not the other produces a theme that differs depending on how it was selected, and nothing catches it. A `@custom-selector` or a preprocessor would fix it; both mean a build step, which [ADR 0009](../decisions/0009-vanilla-dashboard.md) rejects. The mitigation is that the two blocks are adjacent and identical in order, so a diff shows an asymmetric edit.
- **Contrast ratios are not automatically checked.** They were checked by hand against WCAG AA for text and the values chosen accordingly, but a future colour edit has nothing enforcing it. An axe or Lighthouse pass in CI would need a headless browser.
- **No high-contrast or `forced-colors` support.** A `forced-colors: active` block would need `system-color()` keywords throughout and the canvas would need a separate palette, since forced colours do not reach a bitmap.
- **The px type scale ignores a browser font-size preference.** Deliberate, argued above, and the honest cost of having the canvas and the CSS agree. Page zoom is the workaround and it works.
- **`--content-width` is a single max width.** No wide-screen layout beyond 1240px; the page centres. Fine for a report page, and a reader with a 4K monitor gets whitespace rather than an eleven-column table.
- **Only four series colours.** A comparison of more than four variants would repeat one. `riskscore compare` publishes four, which is why there are four.
