# `dashboard/js/dom.js`

## Purpose

The node-building primitives, so that no other file needs `innerHTML` and no other file hand-rolls a `<table>`.

Two problems it closes.

**Untrusted text reaches this page.**
A run id, a feature name, a dataset id, a validation message, and - worst - a failed job's `detail`, which is the verbatim exception text from a child training process.
None of those originate in this repository's markup.
`el()` and `setText()` set `textContent` and never parse HTML, so there is no place in the dashboard where a string becomes markup.
That is not a mitigation applied to a risky pattern; the risky pattern is absent.

**A table is where the interesting question is per-column.**
Eight tables on this page each need: a caption, `<th scope="col">` headers, a `<th scope="row">` first cell, an alignment per column, a conditional colour on some cells, a bar node in one cell of two of them, and an empty state that is visible rather than an empty `<tbody>`.
Built as markup strings that is unreadable; built as eight hand-written `<table>` builders it drifts, and it did - the old dashboard had two tables with different empty states and only one with `scope`.
`renderTable` makes the column the unit, which is where the variation actually is.

The tables are also the *accessibility contract*, not a convenience.
Every canvas on this page has a `<details>` beside it holding the same numbers, and `dataTableDetails` builds it.
A screen reader gets the values; an `aria-label` only gets the headline.

## Public API

| Name | What it is |
| --- | --- |
| `$(selector, root = document)` | `querySelector`. |
| `$$(selector, root = document)` | `querySelectorAll` as a real array. |
| `el(tag, props, children)` | One element. `dataset`, `attrs`, and `style` are merged; everything else is assigned as a property. |
| `replaceChildren(node, children)` | Swap an element's children in one operation, dropping nullish entries. |
| `setText(selectorOrNode, value)` | Set text, or `MISSING` for an absent value. Tolerates a missing node. |
| `renderTable(columns, rows, opts)` | A `<table>` from a column spec. `opts` is `{caption, empty, classes}`. |
| `dataTableDetails(summaryText, table)` | A collapsed `<details>` wrapping a table: a chart's accessible alternative. |
| `setBanner(node, message, tone)` | Show or clear a banner. `message` falsy clears and hides. |
| `setStatus(node, message, state)` | The header status pill. Sets `textContent` and `data-state`. |
| `definition(term, value, {mono})` | One `<div class="def">` holding a `<dt>`/`<dd>` pair. |

A column entry is `{key, label, format, align, tone, render, className}`.

## Inputs and outputs

Takes primitives and plain objects; returns DOM nodes, or nothing.
Imports only `MISSING` from `format.js`.
No fetching, no state, no knowledge of credit risk - the word "PSI" does not appear in this file.

`format(row)` receives the **whole row**, not just the cell value, because a cell's rendering frequently depends on a sibling column: the PSI value's colour comes from the band beside it, and a comparison delta's units come from the metric named in its own row.

`tone(row)` returns a token *name*.
The cell gets `style="color: var(--warn)"`, so the palette stays in `tokens.css` and a theme switch needs no re-render of the table's colours.

`render(row)` returns child nodes instead of text, for the one kind of cell whose content is a bar rather than a number.

## Invariants and failure modes

**`el` assigns properties, not attributes, by default.**
`textContent`, `className`, `disabled`, `value` are all properties.
`attrs` exists for the handful of things that are genuinely attributes with no property equivalent worth using - `scope`, `colspan`, `aria-label`, `role`.
Going through properties means a value is never stringified into markup on its way in.

**`el` skips `null` and `undefined` props and children, and `false` children.**
That is what lets a caller write `caption ? el("caption", ...) : null` inline rather than building an array conditionally.
`false` is skipped as well so `condition && el(...)` works; `0` is *not* skipped, because a cell of zero is a legitimate child.

**`className` lands on the header cell, not just the body cells.**
Under `table-layout: fixed` the *first row* decides the column widths.
A width rule applied only to `<td>`s is ignored, which is precisely how a nine-column comparison table shipped with its last column unreachable.
`cellClass(column)` is called for both the `<th>` and the `<td>`.

**The first column is a `<th scope="row">`, every header a `<th scope="col">`.**
Without `scope`, a screen reader reads a grid of unattached numbers.
With it, a cell is announced as "AUC ROC: 0.683".
Since these tables are the alternative to the canvases, an unscoped table would leave the charts with no accessible representation at all.

**An empty `rows` renders the empty message *inside* the table.**
A headed table with an empty `<tbody>` reads as a load that never finished, and there are legitimate empty states here - no comparison published, no SHAP summary in the run.
The message goes in a `<td colspan>` so the table is still a valid table and the caption is still attached to it.

**`setText` maps `null`, `undefined`, and `""` to `MISSING`, and tolerates a missing node.**
The missing-node tolerance matters because `main.js` renders eleven panels from a `Promise.allSettled` and a partly-available run must still paint what it has.
A `setText` that threw on a node the current markup does not have would turn one absent report into a blank page.

**`setBanner` hides with the `hidden` property, not `display: none`.**
`hidden` removes the element from the accessibility tree as well as from the layout.
A banner hidden with CSS is still announced, so a screen reader user would hear a stale error that a sighted user cannot see.
Clearing also empties `textContent`, so the next `hidden = false` cannot flash the previous message.

**`setStatus` writes text and a `data-state`, and never a colour.**
`app.css` styles `.status[data-state="loading" | "ok" | "warn" | "error"]`.
The text always carries the meaning on its own, so the state is redundant colour rather than load-bearing colour - which is the requirement for anyone who cannot distinguish the four.
The node is an `aria-live="polite"` region in the markup, so a failed load is announced without stealing focus.

**`definition`'s `mono` option is a class, not a font.**
Run ids, dataset hashes, and commit shas go in `.mono`; prose does not.
The font stack itself is in `tokens.css`.

## What must NOT live here

- **`innerHTML`, `insertAdjacentHTML`, `document.write`, or any string-to-markup path.** The whole security argument of this file is that these do not appear anywhere in `dashboard/js/`. One use would end it.
- **Fetching.** `api.js` owns the network. A `renderTable` that could fetch would make every table render non-deterministic.
- **Domain knowledge.** No metric names, no PSI bands, no tier names. `panels.js` supplies those as column specs.
- **Colours.** `tone` names a token; a hex literal here would be invisible to the theme switch.
- **Chart drawing.** `charts.js` owns the canvas. This file's contribution to a chart is the `<details>` table beside it.

## Related tests

There is no `dom.test.js`, deliberately.

Every function here is a thin composition over `document.createElement` and `Node.append`.
A unit test of `el("div", {className: "x"})` asserts that `className` assignment works, which is the platform's guarantee, not this file's.
The behaviour that *is* this file's - column specs, scopes, the empty state, tone-to-token - is exercised on real nodes by every `panels.js` and `score.js` test that builds a table, and there are 31 of those.

The parts a unit test could not have caught are covered by `scripts/probe_dashboard.mjs`, in real Chrome:

- `tables=` in its output line asserts every `.data-fallback table` has a non-zero row count, so a chart whose accessible alternative silently emptied is a failure.
- `importanceBars` counts `.bar` nodes with a measured width of at least 1px, which is the only way to verify that the `render(row)` path produced a *visible* node rather than an empty `<td>`.
- `wideTables` measures every `.table-scroll` container's `scrollWidth` against its `clientWidth`, which is what catches a `className` that failed to reach the header cell and therefore never set a column width.
- The `offenders` walk names the widest overflowing element by selector, so a table that breaks the layout is attributed to a class rather than to "the page".

`renderTable`'s handling of an absent cell shares its guarantee with `format.js`'s `MISSING`, tested exhaustively in `format.test.js`.

## Known limits

- **`renderTable` renders every row.** No virtualization, no pagination. The largest table on this page is the feature PSI table at roughly 30 rows; `.table-scroll` caps the visible height at 420px and scrolls. A feature set in the hundreds would want a different approach, and the upgrade path is a slice plus a row count in the caption, not a windowing library.
- **No sorting and no filtering.** Row order is decided by the panel that builds the rows - importance by magnitude, vintages by year, PSI by the payload's order. Column-click sorting would need this file to own state, which it currently does not.
- **`el`'s property-first assignment is silent about typos.** `el("div", {classname: "x"})` sets an expando named `classname` and renders an unstyled div. A whitelist was rejected as more code than the mistake costs, and the probe catches the visible consequence.
- **`setStatus` has no queue.** Two status changes in the same frame mean only the second is announced. In practice the states are `loading` then one terminal state, so there is nothing to queue.
- **`dataTableDetails` is collapsed by default.** A reader who wants the numbers has to open it. Expanding by default would double the page height for readers who only want the charts; the `<summary>` text states what is inside.
