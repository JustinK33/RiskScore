# `dashboard/styles/app.css`

## Purpose

Every component style on the page, and every responsive decision.

It reads `tokens.css` and declares no raw values of its own - if a colour or a size is not a `var(--...)`, it belongs in the token file.
That rule is what makes a theme change one edit and a spacing change one edit, and it is why this file is long without being hard to change.

The layout is **grid only**, and one detail in that is load-bearing: `minmax(0, 1fr)` wherever a track holds a canvas or a table.
Bare `1fr` resolves to `auto` for min-width purposes, so a single wide child - a nine-column comparison table, a canvas with a long axis label - pushes the whole document into a horizontal scrollbar.
That is exactly how the old stylesheet overflowed at 320px *despite* having a `min-width` on the body: the body was fine and a grandchild was not.

## Public API

Class names, consumed by `dom.js`, `panels.js`, `score.js`, `main.js`, and `index.html`.

| Group | Selectors |
| --- | --- |
| Page frame | `.topbar`, `.skip-link`, `.wrap`, `.panel`, `.panel-header`, `.panel-grid`, `.panel-grid.one-row`, `.panel-grid.stack-late`, `.side-panel` |
| Status | `.status[data-state]` (`loading`/`ok`/`warn`/`error`), `.banner[data-tone]` (`info`/`warn`/`danger`/`ok`) |
| Identity | `.identity-grid`, `.def`, `.chip`, `.chip-label`, `.chip-split`, `.chip-arrow`, `.mono` |
| Metrics | `.metrics-grid`, `.metric` |
| Tables | `.data-table`, `.table-scroll`, `.table-empty`, `.num`, `.data-fallback` |
| Charts | `.chart`, `figcaption` |
| Forms | `.field`, `.field.inline`, `.field-group`, `.hint`, `.required` |
| Importance | `.importance-table`, `.magnitude-cell`, `.bar`, `.feature-rows` |
| Comparison | `.comparison-table`, `.delta`, `.columns-cell` |
| Score | `.verdict[data-decision]`, `.verdict-decision`, `.verdict-figure`, `.reason-table`, `.tornado`, `.tornado-cell`, `.tornado-head`, `.tornado-left`, `.tornado-right`, `.bar-up`, `.bar-down` |

Breakpoints: `640px`, `900px`/`901px`, `1200px`, `520px`.
Plus `prefers-reduced-motion` and one `@supports (grid-template-rows: subgrid)`.

## Inputs and outputs

Reads `tokens.css` via `var()`.
Loaded by `index.html` as a plain `<link>` after it - two `<link>`s rather than an `@import`, because `@import` serializes the second request behind the first.

Nothing reads this file back.
The one coupling in the other direction is that `panels.js` and `score.js` name specific classes for cells they want hidden at narrow widths (`.tornado-cell`, `.tornado-head`, `.magnitude-cell`, `.columns-cell`), so those names are a contract between the two files.

`body { min-width: 320px }` sets the floor.
`html { scroll-padding-top: var(--space-8) }` so an anchored panel is not hidden under the sticky header.

## Invariants and failure modes

**`minmax(0, 1fr)` on every track that can hold a canvas or a table.**
Stated above; it is the single most repeated decision in the file and the one that a "simplification" to `1fr` would silently undo.
The failure is not a broken layout - it is a 24px horizontal scrollbar that macOS does not even draw until something scrolls.

**Breakpoints are chosen where the *content* stops fitting, not at device names.**
640px is where the two bar columns stop having room for a bar; 900px is where the side panel can no longer sit beside the main content; 1200px is where the widest tables start hiding columns inside their scroll container; 520px is where the metric grid drops to one column.
None of them correspond to a phone.

**`.panel-header > .field.inline { width: 360px; max-width: 100% }`.**
Both `min-width` and a flex basis were tried first and both are wrong, and the file says so: `min-width` lets the field grow past the header on a long option label, and a flex basis does not constrain a grid child.
A fixed width with a percentage cap is the only form that both reserves space and yields at 320px.

**`.field select, .field input { width: 100%; min-width: 0 }`.**
The `min-width: 0` is the necessary half.
Form controls have an intrinsic minimum width that ignores `width: 100%`, so without it a `<select>` holding a long run-id label is wider than its grid track and overflows the panel.

**`.identity-grid > .def:first-child { grid-column: span 2 }`, gated on `@media (min-width: 640px)`.**
The run id is the longest value in the strip and spanning two columns keeps it on one line.
Gated because at 320px there is only one column, and `span 2` on a one-column grid creates a second implicit column - which is an overflow, not a wider cell.

**`.metrics-grid` goes 4 → 2 → 1.**
Eight metric cards.
Three columns leaves an orphan; the sequence 4/2/1 always fills its rows.

**`.panel-grid { grid-template-columns: minmax(0, 1fr) 320px }` with `.side-panel { grid-column: 2; grid-row: 1 / span 2 }`.**
The side panel spans both rows because the main column holds a chart and a caption.
`@media (min-width: 901px) { .panel-grid.one-row .side-panel { grid-row: 1 } }` for the panels whose main column is a single row - the comparison and the importance table.
The `901px` rather than `900px` is so it cannot overlap the `max-width: 900px` block by one pixel.

**`@media (max-width: 1200px)` spells `.panel-grid.stack-late .side-panel` *and* `.panel-grid.one-row.stack-late .side-panel`.**
Because the `one-row` rule above has higher specificity and would otherwise keep the side panel in column 2 after the grid has collapsed to one column - putting it in an implicit second column, which is an overflow.
Two selectors rather than `!important`.

**`.table-scroll { max-height: 420px; overflow: auto }`, with sticky headers.**
`.data-table thead th` is `position: sticky` with `white-space: normal` while body cells are `nowrap` - headers wrap so the column can be narrow, values do not wrap so a number is never split across lines.

This container is also a hazard, and the probe exists partly because of it: a scroll box **contains its own overflow**, so a table whose last columns sit outside it looks identical to one that fits, since macOS draws no scrollbar until something scrolls.
That is how a nine-column comparison table shipped with its approval-rate column unreachable at 1440px.
`probe_dashboard.mjs` measures every `.table-scroll`'s `scrollWidth` against its `clientWidth` at every width and reports the one that wanted more.

**`.field-group { grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)) }` - `auto-fill`, not `auto-fit`.**
`auto-fit` collapses empty tracks, so a fieldset with two fields stretches them across the full width while the fieldset above it has five at 200px.
`auto-fill` keeps the column rhythm consistent down the form, which is what makes the generated form read as a form rather than as four unrelated blocks.

**Subgrid is behind `@supports`, and the reason is in the file.**
`@supports (grid-template-rows: subgrid) { .field-group .field { grid-row: span 3; grid-template-rows: subgrid; align-content: start } }`.
It aligns every label, control, and hint across the row.
`span 3` *without* subgrid support would make each field occupy three rows of the parent grid with nothing filling them, so the fallback has to be no rule at all rather than a partial one.

**`.reason-table td, .reason-table th { vertical-align: middle }` but `.comparison-table th, .comparison-table td { vertical-align: top }`.**
A reason row is a bar that has to line up with its label; a comparison row has a wrapping variant name beside single-line numbers, and centring those against a two-line name puts the numbers in the gap.

**`.feature-rows th[scope="row"]` gets `overflow-wrap: anywhere`.**
Feature names come from a fitted preprocessor and can be long compound identifiers with no break opportunity.
Without this, one name sets the column width for the whole table.

**`.importance-table .magnitude-cell { min-width: 140px; width: 22% }` and `.tornado-cell { min-width: 140px; width: 30% }`.**
A bar column needs a floor to be readable and a cap so it does not take the row.
Both are dropped entirely at `@media (max-width: 640px)`, along with `.tornado-head` and `.columns-cell` - below that width the bar is narrower than its own border and the numeric column is the reading.
The probe asserts both directions: at least two measurable bars above 640px, and exactly zero below it.

**`:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px }`, and no `outline: none` anywhere.**
`:focus-visible` rather than `:focus` so a mouse click on a button does not leave a ring.
Every interactive control on the page is reachable by keyboard with a visible ring, which is checked by hand at each width.

**`.skip-link` is animated by `top`, not by `display`.**
A `display: none` skip link is not in the accessibility tree and cannot be focused, which defeats the purpose.
It sits off-screen above the viewport and slides in on focus.

**`@media (prefers-reduced-motion: reduce)` removes every transition and animation.**
One block, at the top, so it cannot be forgotten by a rule added later - a rule added below it would win on order, so new transitions have to be checked against it.
That is a real maintenance hazard and the reason it is worth naming here.

**`.status[data-state]` has four states and the text always carries the meaning.**
`loading`, `ok`, `warn`, `error`.
The colour is redundant, not load-bearing, which is the requirement for a reader who cannot distinguish the four.
Same for `.banner[data-tone]` and for `.delta`, where the sign is present in the text.

**`.chip-arrow` is styled but decorative**, and `main.js` marks it `aria-hidden`. A screen reader announcing "right arrow" between two counts adds nothing to a reading order that already conveys the sequence.

**No `!important` anywhere, and no ID selectors.**
Specificity is managed by writing the more specific selector, as in the `stack-late` case above.
Every rule is a class or an element.

## What must NOT live here

- **Raw colours, sizes, radii, shadows, or font stacks.** Every one is a `var(--...)` from `tokens.css`. A hex literal here is invisible to the theme.
- **`prefers-color-scheme`.** Theming is entirely in `tokens.css`; this file styles components and never asks what theme it is in.
- **Chart colours.** The canvas gets its palette from `charts.js` reading the tokens. A `.chart` rule cannot reach inside a bitmap.
- **`display: none` for anything an assistive technology should still find.** Hiding is `hidden` on the element, set from JS, or `top`-based off-screening for the skip link.
- **`outline: none`.** For any reason.
- **Flexbox for page layout.** Grid only, so a track can be constrained with `minmax(0, 1fr)`. Flex is used inside a few components where the content genuinely flows.

## Related tests

No unit tests.
Nothing in `node --test` parses CSS, and an assertion that a declaration says what it says is a restatement.

`scripts/probe_dashboard.mjs` is the test for this file, and most of its assertions exist because of a bug in it:

- **Overflow, attributed.** It walks every element in `body`, compares each right edge to the document's `clientWidth` with 1px of slack for fractional layout, sorts the offenders by how far out they are, and reports the widest five by selector. "The page overflows by 24px" is not actionable; "`.panel-header > .field.inline` ends at 344px in a 296px viewport" is - and that is the report that produced the `width: 360px; max-width: 100%` rule.
- **Hidden table columns.** Every `.table-scroll` whose `scrollWidth` exceeds its `clientWidth`, reported with the width it wanted, checked at 900px and above. This is invisible to the document-level check and invisible in a screenshot.
- **Bar columns, both directions.** At least two measurably-wide `.bar` nodes above 640px; exactly zero below it. So the deliberate narrow-layout drop is asserted as a decision rather than tolerated as an absence.
- **Zero-width reason bars.** Counts `.bar` nodes inside `#scoreResult` that are laid out and measure under 1px - the 2% floor's proof, and it excludes bars in columns the narrow layout does not render at all.
- **Canvas sizing.** Every canvas at least 100px wide with a non-zero bitmap, which catches a grid track that collapsed.
- **Both themes.** Twelve configurations - six widths in light and dark - with the computed `body` background reported in each line.

All twelve pass, against both a default server and one with the mutating routes enabled.

The remaining check is by hand: keyboard traversal reaching every control with a visible ring, and a look at each width for the things a script cannot judge - a caption that wraps badly, a chip that sits a pixel off its neighbour.

## Known limits

- **The probe is not in CI.** It needs Chrome and a running server with a published run, so a CSS regression that breaks the 390px reflow would pass CI. This is the largest gap in the dashboard's verification, and it is accepted rather than solved: the alternative is a headless-browser job plus a screenshot baseline for a portfolio dashboard. Named as a consequence in [ADR 0009](../decisions/0009-vanilla-dashboard.md).
- **1198 lines in one file.** Splitting it per component would mean either more `<link>`s or an `@import` chain that serializes. Grouped by section with banner comments instead, and the grouping order matches the document order in `index.html` so a rule is findable.
- **Breakpoint values are repeated across blocks** - `640px` appears as both a `min-width` and a `max-width`, and `900`/`901` as a pair. Custom properties cannot be used in media queries, and `@custom-media` is not shipped, so the numbers are literal. An off-by-one between a pair is the failure mode, which is why the 901 exists.
- **No `forced-colors` or high-contrast support.** Would need `system-color()` keywords throughout, and the canvas would need a separate palette since forced colours do not reach a bitmap.
- **No print stylesheet.** The charts would print as drawn - the palette is greyscale-distinguishable, so a colour printer is not required - but the sticky headers, the scroll containers, and the collapsed `<details>` would all print wrong. A `@media print` block that opens every `<details>` and unsets the scroll caps is the obvious addition.
- **`.table-scroll`'s 420px cap is a fixed pixel height.** On a short viewport it is most of the screen; on a tall one it wastes room. A `min(420px, 60vh)` would be better and has not been needed.
- **The reduced-motion block is at the top, so a transition added below it wins.** A rule added later has to be checked against it by hand. Moving it to the bottom of the file would fix the ordering hazard at the cost of putting an accessibility guarantee somewhere nobody reads first.
