# 0009 - A dashboard with no build step

Status: accepted.
Affects `dashboard/index.html`, `dashboard/styles/*.css`, `dashboard/js/*.js`, `dashboard/package.json`.

## Context

The dashboard is the only thing most readers of this repository will actually run.
It renders eleven panels: run identity, eight metric cards, calibration, threshold costs, the embargo before/after comparison, per-vintage metrics, score and feature PSI, feature importance, a model comparison table, a score-an-applicant form, and a retrain panel.
That is enough surface that "just use a framework" is a defensible instinct, so the decision to keep it as plain files needs to be written down rather than assumed.

Four facts about this particular dashboard pushed against a toolchain.

**It renders one document, from one origin, with no routing.**
Every panel is driven by a `GET` against the same FastAPI app that serves the HTML.
There is no client-side navigation, no shared mutable store beyond a single `state` object in `main.js`, no optimistic updates, and no list a user reorders.
The problems a framework is good at - reconciling a large tree against frequent fine-grained state changes - do not occur here.
A panel either has its payload or it does not, and when it does it is rendered once.

**The charts have to read CSS.**
`charts.js` calls `getComputedStyle(element)` to pull `--chart-*` custom properties, because a `<canvas>` has no cascade and dark mode has to reach the plotted lines.
A charting library would want its colours passed in as literals, which means a second source of truth for the palette and a theme toggle that updates it by hand.
The one thing a chart library would definitely have saved - axis ticks and label measurement - is about 120 lines in `charts.js` (`niceTicks`, `axisPadding`, `drawFrame`), and those 120 lines are covered by 25 tests that assert exact tick values.

**A build step would break the clone-and-run story.**
`riskscore serve` mounts `dashboard/` as static files.
With a bundler, the mount would have to point at a `dist/` that either gets committed (a tracked build output, and a diff nobody can review) or gets built (a `npm ci && npm run build` between `pip install` and a working dashboard, on a project whose install instructions are otherwise one line).
The Dockerfile would grow a Node stage for a directory that ships eleven text files.

**The interesting logic is pure, and pure logic does not need a browser to test.**
Ticks, formatting, embargo row diffing, comparison deltas, importance bar widths, PSI banding, reason-code scaling, job status lines - all of it is data in, data out.
`node --test` runs 104 tests over those functions in under a second with zero dependencies installed.
The parts that genuinely need a browser (does the grid reflow at 390px, does the file picker look right) are checked by `scripts/probe_dashboard.mjs` driving real Chrome over CDP, which a jsdom-based test runner could not have answered anyway.

## Decision

**1. No npm dependencies, no bundler, no framework, no charting library.**
`dashboard/package.json` has no `dependencies` and no `devDependencies` fields at all.
It exists for two lines: `"type": "module"`, so `node --test` parses `js/*.js` as ES modules, and `"scripts": {"test": "node --test"}`.
There is nothing to `npm install`; `node --test` from `dashboard/` works on a fresh clone with only Node present.

**2. ES modules loaded with `<script type="module">`.**
`index.html` ends with `<script type="module" src="/js/main.js">` and the module graph does the rest - `main.js` imports `api`, `dom`, `format`, `panels`, `score`, and `retrain`; `panels` imports `charts`, `dom`, and `format`.
`type="module"` is deferred by specification, so this also removes the blocking `<script>` the old dashboard had at the end of `<body>`.
Imports are relative with explicit `.js` extensions, which is what both the browser and Node require, so the same files run in both without a resolver.

**3. Layering enforced by import direction, not by tooling.**
`format.js` imports nothing and touches no DOM.
`dom.js` imports nothing and knows nothing about credit risk.
`charts.js` imports nothing and deals only in pixels, ticks, and fonts.
`panels.js` imports `charts`, `dom`, and `format`, and is the only place that knows what a PSI band or a lender-priced tier is.
`main.js` imports everything and is the only place that fetches.
That ordering is the whole architecture, and it is legible from the import lines at the top of each file - which is roughly what a bundler's dependency graph would have told us, for free.

**4. `node --test` on the pure helpers; real Chrome for the rest.**
Every module except `dom.js` and `main.js` has a sibling `*.test.js`.
`dom.js` is untested directly because it is a thin wrapper over `document.createElement` whose behaviour is the platform's, and it is exercised indirectly by every `panels.js` test that builds a table.
`main.js` is untested directly because it is wiring, and a unit test of wiring asserts that the wiring is what it is.
Both are covered instead by `scripts/probe_dashboard.mjs`, which loads the page in headless Chrome at twelve viewport-and-theme configurations and asserts no console errors, no horizontal overflow, and that specific nodes carry real content.

**5. One inline blocking script, deliberately.**
`index.html` has exactly one inline `<script>`, in `<head>`, which reads `localStorage.getItem("riskscore.theme")` and sets `data-theme` on `<html>`.
It has to be inline and blocking: `main.js` is deferred, so a dark-mode user would get a white flash before it ran.
That is the only exception to rule 2, and it is why the theme key is duplicated between `index.html` and `main.js`'s `THEME_STORAGE`.

**6. JavaScript is formatted by hand.**
There is no prettier and no eslint.
The style is 2-space indent, double quotes, semicolons, trailing commas in multi-line literals - matched to what is already in the file being edited.
Python has `ruff format` in CI; JS does not, and a formatter is the one dependency that would have to run in CI to be worth anything.

## Consequences

**A fresh clone runs the dashboard with no Node step.** `pip install -e ".[serve]"` then `riskscore serve` is the whole path. Node is needed only to run the JS tests, and that is a developer concern, not an install step.

**The JS is not minified and not tree-shaken.** Eleven files, about 4,300 lines, served uncompressed by Starlette's `StaticFiles`. Over localhost that is irrelevant; behind the gzip middleware over a network it is roughly 30 KB. For a dashboard served next to its own API, this is not a number worth a toolchain.

**Every module is one HTTP request.** Ten module fetches on first load, all from the same origin over HTTP/1.1 or HTTP/2 depending on what fronts it. Cached thereafter. The waterfall is one level deep because the import graph is shallow, so this is not the pathological many-round-trips case that motivated bundling in the first place.

**No JSX, no templates, no `innerHTML`.** `dom.js`'s `el()` is what builds every node. That is more verbose than markup - a table column is an object with a `render` function rather than a template - but it is also why there is no `innerHTML` anywhere in the codebase, and run ids, feature labels, and a child process's exception `detail` all arrive from off-page. `textContent` cannot execute them.

**No type checking on the JS.** `src/` is `mypy --strict`; `dashboard/js/` has nothing equivalent. The mitigation is that every module boundary is tested with the wrong types on purpose - `format.test.js` asserts what `null`, `""`, `[]`, `true`, and `"12.3"` each do, because `Number([]) === 0` is exactly the bug that turns an absent value into a confident zero.

**A future framework migration is not blocked, but it is also not free.** `format.js`, `charts.js`, and the pure functions in `panels.js` would port unchanged; `dom.js`, `main.js`, and the DOM-building halves of `panels.js` and `score.js` would be rewritten. That is the honest cost, and it is the reason the pure/impure boundary is where it is.

**Dark mode works in the charts.** Because the palette lives in `tokens.css` as custom properties and `charts.js` re-reads them on every draw, toggling the theme and calling `redraw()` is the entire implementation. No library needed to be told about a second colour scheme.

**Resize is free.** `api.js` caches parsed payloads in memory, so `observeResize` → `redraw()` re-renders from cache and issues zero requests. This is a property of owning the fetch layer; it is achievable with a library, but not by default.

**`node --test` is fast enough to run on every commit.** 104 tests, no install, no transpile, no jsdom boot. The reason those tests exist at all is that they cost nothing to run.

**The probe is the only thing that catches layout regressions**, and it needs Chrome on the machine. CI runs the `node --test` suite; the probe is a local pre-commit tool. That is a real gap: a CSS change that breaks the 390px reflow would pass CI. Accepted, because the alternative is a headless-browser job and a screenshot baseline for a portfolio dashboard.

## Alternatives considered

**React, Vue, or Svelte.** All three solve fine-grained reactive updates over a large component tree. This dashboard renders each panel once per `load()` and re-renders only canvases on resize. The state is one object with twelve fields, mutated in one function. Buying a reconciler for that means buying a build step, a dependency tree, and a `dist/` for a problem the page does not have.

**Chart.js, Plotly, or D3.** Rejected for three reasons. The palette would become a second source of truth outside `tokens.css`, breaking the theme toggle. Two of the charts are unusual enough that the library would be fought rather than used - `drawThresholdCosts` has two independent y axes, and `drawEmbargo` draws grouped bars where one series is deliberately `NaN` for a dropped vintage. And the accessibility contract here is that every canvas is `role="img"` with an `aria-label` stating the headline reading plus a `<details>` data table generated from the same payload, which is a thing this code does uniformly and a library would do differently per chart type. D3 specifically would have replaced `niceTicks` with `d3.ticks` - and `niceTicks` is a 30-line transcription of `d3.tickIncrement`, tested against exact expected outputs.

**Vite with no framework.** Would give HMR and a bundle without React. But it still means `npm install`, a `dist/`, a Docker Node stage, and a `serve` command that has to know whether it is pointing at source or output. The thing it buys - HMR - is replaced adequately by a browser reload against a static mount, because there is no client state to preserve across an edit.

**Web components.** Would give encapsulation without a dependency. Rejected because shadow DOM would cut the panels off from `tokens.css`, which is the mechanism the whole theme system runs on, and because the encapsulation problem it solves does not exist in an eleven-file page with one stylesheet.

**A `<template>`-based renderer instead of `el()`.** Cloning `<template>` nodes and filling slots is idiomatic and needs no library. Rejected because the tables are data-shaped - `renderTable(columns, rows)` where a column carries `format`, `tone`, and `render` functions - and expressing "this cell's colour token depends on its row's PSI value" in markup means putting logic in attributes. `el()` keeps it in JavaScript where it is testable.

**TypeScript with `tsc` emitting nothing (`--checkJs` on JSDoc).** Tempting: type checking with no runtime output and no bundler. Rejected for now because it would mean JSDoc annotations on every function in eleven files and a `tsconfig.json`, to catch a class of bug that the "absent is not zero" tests already target directly. This is the alternative most likely to be revisited.

**Prettier for the JS.** Rejected because a formatter is only worth having if CI enforces it, and enforcing it means the Node dependency this ADR exists to avoid. Hand-formatting eleven files to a consistent style is achievable; hand-formatting a hundred would not be, and that is the threshold at which this gets revisited.
