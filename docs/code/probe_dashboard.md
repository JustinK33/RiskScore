# `scripts/probe_dashboard.mjs`

## Purpose

Measure the dashboard in a real browser at every breakpoint, and name the element that broke.

It exists because the failures this dashboard actually had are invisible to both of the cheaper options. A unit test on a stubbed DOM cannot see a layout - `getBoundingClientRect` returns zeros - and a screenshot shows you *that* something is wrong without telling you which of four hundred elements caused it. The specific bug that motivated the script was a 24px horizontal scrollbar at 320px that macOS never drew, on a page whose `body` already had a `min-width`: the body was fine and a grandchild was not.

So the assertions are measurements, and the report is attributed. Not "the page overflows by 24px" but ``.panel-header > .field.inline` ends at 344px in a 296px viewport`` - which is the line that produced the `width: 360px; max-width: 100%` rule in `app.css`.

It also drives the score panel end to end. That form is generated from `/api/schema` at runtime and read back out of live DOM nodes, so a control whose `value` does not round-trip - a `type="month"` handed `Jun-2015`, a required `<select>` with no matching option - produces a browser validation block or a 422, and neither is reachable from a unit test. Nothing short of real Chrome talking to a real server proves the generated form is submittable at all.

It is **deliberately not in CI**. It needs an installed Chrome and a running server with a published run. That is the largest gap in the dashboard's verification and it is named as such in [ADR 0009](../decisions/0009-vanilla-dashboard.md).

## Public API

A script, not a module. No exports.

```
node scripts/probe_dashboard.mjs [base-url]
```

The URL is a **positional** argument, defaulting to `http://127.0.0.1:8125/`. There is no env var for it - passing one silently probes the default port instead. `CHROME` *is* read from the environment, defaulting to the macOS bundle path.

Exits `0` on a clean run, `1` on the first failing configuration, after printing every configuration.

| Constant | Value | Why |
| --- | --- | --- |
| `PORT` | `9333` | The CDP port, not the server's. |
| `WIDTHS` | 320, 520, 768, 900, 1024, 1440 | The plan's list: below the smallest phone, the metrics-grid break, a tablet, the side-panel break, a laptop, the content cap. |
| `THEMES` | `light`, `dark` | Both, because the palette is not an inversion. |
| `NARROW_BAR_WIDTH` | 640 | Where `app.css` drops the bar columns. |
| `FULL_TABLE_WIDTH` | 900 | Above this, a table hiding columns is a bug rather than the scroll container doing its job. |

Twelve configurations: six widths in two themes.

## Inputs and outputs

Speaks the Chrome DevTools Protocol over a WebSocket, using Node's built-in `fetch` and `WebSocket` and nothing else - no puppeteer, no playwright, no npm dependency of any kind. `spawn` from `node:child_process` starts Chrome with `--headless=new` and a persistent profile at `/tmp/riskscore-probe-profile`.

Reads the target server's `/readyz` once, before Chrome, to learn whether it will accept a retrain.

Writes one line per configuration to stdout, plus an indented line per problem. The line carries measurements even on success - `scroll=`, `bg=`, `canvas=`, `tables=`, `psi=`, `imp=`, `cmp=`, `wide=[]`, `score=`, `verdict=`, `retrain=`, `banners=` - so a clean run is still a reading, and a value drifting toward zero is visible before it becomes a failure.

Touches no file in the repository.

## Invariants and failure modes

**The readiness signal is the dashboard's own `#status` region, not a timer and not `load`.**
`data-state` leaves `loading` only once every fetch has settled and every panel has rendered, so it is exactly the condition the assertions need. A `render()` that threw halfway leaves it at `loading` and the run fails on the 15s deadline with the width that failed. Polling a real signal rather than sleeping a guessed interval is also why this script has no flake.

**The cache is disabled for every session, and that is a fix, not hygiene.**
The profile directory persists between runs and the service sends an `ETag` with no `Cache-Control`, so Chrome heuristically reuses a stylesheet it already has. That made an earlier run report a CSS fix as still broken. A probe that reads stale bytes is worse than no probe.

**The theme is emulated, not written to `localStorage`.**
`Emulation.setEmulatedMedia` with `prefers-color-scheme`. So it measures the OS-preference path - the one with no JS involved, and therefore the one where a `:not([data-theme="light"])` gate in `tokens.css` is load-bearing. Writing storage would test the toggle instead and would leave the media-query branch unexercised.

**Overflow is attributed, with 1px of slack.**
Every element in `body` gets its right edge compared to the document's `clientWidth`; the slack is there because a fractional layout width rounds up and is not a scrollbar. Offenders are sorted by how far out they are and the widest five are reported by selector.

**Every `.table-scroll` is measured separately, and only at 900px and above.**
A scroll box **contains** its own overflow, so the document-level check above is blind to a table whose last columns sit outside it - and macOS draws no scrollbar until something scrolls, so it is blind in a screenshot too. That is how a nine-column comparison table shipped with its approval-rate column unreachable at 1440px. Below 900px a scrolling table is the intended design, which is why the check has a floor.

**The bar columns are asserted in both directions.**
At least two measurable bars above 640px; **exactly zero** below it. So the deliberate narrow-layout drop is checked as a decision rather than tolerated as an absence - and a bar surviving below 640px is the 250px of overflow the tornado column taught us to avoid.

**Zero-width reason bars are counted, and only the laid-out ones.**
The renderer has a 2% floor precisely so a reason code is never an invisible bar. Bars in a column the narrow layout does not render at all are excluded rather than counted, because at those widths the log-odds text is the reading.

**`retrainVisible` is compared against the target server's own `/readyz`, not a constant.**
"The panel is missing" and "the panel is correctly switched off" are the same pixels. Reading `mutating_routes` from the server under test means one script checks both branches, and the run is only meaningful against both servers - a default one and one started with `RISKSCORE_ALLOW_UPLOAD=1 RISKSCORE_ALLOW_RETRAIN=1`.

**Placeholder rows are distinguished from empty tables.**
`featurePsiRows < 2` and `importanceRows < 2` fail, because one row is the "No rows." placeholder - an empty table wearing a header. `comparisonRows < 1` fails, because there the one-row placeholder is a legitimate state and the assertion is that the panel rendered *something*; a real multi-variant comparison additionally has to have drawn deltas. `comparisonNote < 40` characters fails, because a panel that explains nothing is a panel a reader cannot act on.

**A canvas has to be wide, bitmapped, and labelled.**
Under 100px CSS width catches a collapsed grid track; a zero bitmap catches a `prepareCanvas` that never ran; an `aria-label` still ending in `loading` catches a chart that failed to draw. That last one only works because `index.html` ships the placeholder label ending in that word.

**A metric card reading `-` is a failure.**
`-` is `format.js`'s MISSING and also the initial markup, so the assertion covers "the run has no value for it" and "the lookup never happened" with one check.

**The three page-side probes run in a fixed order, and the order is the point.**
`PROBE` first, to measure the run the page opened with. Then `SCORE`, then `PROBE` **again** - because the verdict block and the reason table are the tallest things this page can add and their overflow has to be measured with them present. `SWITCH` last, because it replaces every panel's data, so anything measured after it would describe a different run.

**`SWITCH` asserts the option labels are distinct before switching.**
Two options spelled the same way are two options a reader cannot choose between, which is what happened when `riskscore compare` published variants that differ only by tier within the same minute. A one-run tree is the ordinary state and passes.

**`SWITCH` waits for the identity strip to show the run it asked for**, not merely for the status pill to settle. A run in the registry whose reports are missing produces a page of dashes and an amber pill, which is indistinguishable from a fresh load in a screenshot.

**Backticks cannot appear inside `PROBE`, `SCORE`, or `SWITCH`.**
They are template literals evaluated in the page, so a comment mentioning `` `.table-scroll` `` is parsed as interpolation and throws `ReferenceError: scroll is not defined` - inside Chrome, where the message surfaces as a probe that "did not return". The three blocks use plain prose in their comments for this reason.

**Console errors fail the configuration.**
`Log.enable` plus a filter on `level === "error"`, and the first one is reported. This is what catches a module that failed to load, a lookup that threw, and `main.js`'s `boot` rethrowing a render bug.

**Chrome is killed in a `finally`.**
An assertion that throws must not leave a headless browser and a 9333 listener behind, because the next run would attach to the old one.

**It reports every configuration, then exits.**
`failures` counts configurations, not problems, and the loop does not break early. A CSS change that breaks three widths should show all three in one run rather than one per invocation.

## What must NOT live here

- **A dependency.** Node's `fetch`, `WebSocket`, `spawn`, and `once` are enough for CDP. Adding puppeteer would put a browser download and a version matrix into a repository whose dashboard has no `node_modules` at all.
- **Any assertion about numbers being *correct*.** This script checks that a value is present, measurable, and laid out. Whether the AUC is right is `tests/test_evaluation.py`.
- **A screenshot baseline.** Byte-comparing renders across Chrome versions and font stacks is a maintenance job with no owner here. The measurements are the assertion.
- **Writes to `reports/`, or any mutating request.** It reads, it scores, and it switches runs. It never posts a retrain, even against a flags-on server.
- **A sleep as a synchronization primitive.** Every wait polls a real condition with a deadline.
- **Backticks in the injected sources.** See above.

## Related tests

It *is* the test - the top of the file says it is not a pytest test and says why.

What it cannot cover, and what covers that instead:

- The pure helpers it exercises through the page have their own suites: `format.test.js` (19), `charts.test.js` (25), `api.test.js` (17), `panels.test.js` (20), `score.test.js` (11), `retrain.test.js` (9). 101 tests under `node --test`, run from `dashboard/`.
- Every payload it reads is asserted server-side in `tests/test_api.py` and `tests/test_routes_admin.py`.
- Keyboard traversal and the things a script cannot judge - a caption that wraps badly, a chip a pixel off its neighbour - are checked by hand at each width. The `prefers-reduced-motion` and `forced-colors` paths are not checked at all.

Last full run: **all 12 configurations clean against both servers** - default flags on 8125 and both mutating routes enabled on 8126, so `expectRetrain` was exercised in both directions.

## Known limits

- **Not in CI, which is the whole limitation.** A CSS regression that breaks the 390px reflow, or a lifecycle bug in `main.js` that does not throw, passes CI. Fixing it means a headless-Chrome job plus a server with a published run in the workflow - reachable now that the synthetic end-to-end job exists, and not yet done.
- **One browser.** Chrome only. Safari and Firefox differ on `<select>` intrinsic width, on `type="month"`, and on sticky table headers inside a scroll container - all three of which this page depends on.
- **`--hide-scrollbars`.** Overflow is measured, not seen, so a real scrollbar's 15px of gutter is not in the layout. That is what makes the widths comparable across platforms and it means a layout that only breaks *because* of a scrollbar is missed.
- **Six widths, not a sweep.** A break at 641px between two sampled widths is invisible. A continuous sweep would be slow and would report the same failure many times.
- **`deviceScaleFactor: 1`, `mobile: false`.** No retina bitmap check, no touch emulation, no `hover: none` media query. `prepareCanvas` handles a DPR of 2 and that path is unit-tested rather than probed.
- **The profile at `/tmp/riskscore-probe-profile` persists.** Deliberate, so Chrome starts fast, and the reason `Network.setCacheDisabled` is not optional. It is also litter that nothing cleans up.
- **The score probe needs a server with a loaded bundle.** Against a bundle-less service, `/readyz` fails, every panel is a dash, and the run fails at the first width with a status error - correct, but the diagnosis is "start the server properly" and the script does not say so.
- **No timing assertions.** It never checks that a resize storm issues zero requests, which is `api.js`'s central claim and is verified by unit test and by hand in DevTools instead. `Network.requestWillBeSent` is already enabled and counting it would be a small addition.
