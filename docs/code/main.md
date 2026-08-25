# `dashboard/js/main.js`

## Purpose

Boot, state, and wiring. The only module that touches both the network and the DOM - and it does neither directly: `api.js` fetches, `dom.js` builds, `panels.js` draws, `charts.js` paints.

Two properties define this file, and both are corrections of the dashboard it replaces.

**`redraw` never fetches.**
Everything a chart needs is already in `state`, so a window resize, a theme change, and a font finally settling all cost one draw and zero requests. The old dashboard re-requested two endpoints on every `resize` event and discarded the outcome with `.catch(() => {})`. Dragging a window edge was therefore a request storm whose failures were invisible and whose responses could arrive out of order and paint stale data over fresh.

**A partly-available run still renders.**
Reports are fetched with `Promise.allSettled`, so a run trained before a given report existed shows every panel it can and *names* the ones it cannot. The old version treated any non-OK response from any of three endpoints as one fatal "Report artifacts are missing. Run the baseline pipeline first." - which was also what it said when the service was still starting up, and when the API key was wrong.

## Public API

| Name | What it is |
| --- | --- |
| `reload(runId = state.runId)` | Drop every cached payload and re-read from the server. |

That is the whole export. `boot()` runs at module scope, and `<script type="module">` is deferred, so the page is wired as soon as the document is parsed with no `DOMContentLoaded` listener.

`reload` is exported for `retrain.js`'s `onDone`, which is the one thing outside this file that needs to invalidate the page.

## Inputs and outputs

Reads: `/readyz`, then ten report endpoints, then `/api/schema` (via `score.js`).
Writes: the DOM, `localStorage["riskscore.theme"]`, and `history.replaceState`.

`state` is a plain object, not a store:

```
runId, activeRunId, model, manifest, metrics, calibration, thresholdCosts,
vintages, drift, shap, comparison, runs, problems, mutatingRoutes
```

A plain object because there is one page, one run selected at a time, and no component that owns a slice of it. `load` **replaces the whole object** rather than mutating fields, so a half-applied run cannot be rendered.

Two constants carry judgements: `MIN_TEST_ROWS = 1000` and `MIN_TEST_POSITIVES = 50`, below which a test-set metric is noise being reported to three decimals. `ARTIFACTS` is the twelve run-directory files worth linking, in reading order - card, manifest, metrics, then the CSVs, then a figure, then the log.

## Invariants and failure modes

**`/readyz` is fetched first and alone.**
When the answer is "no bundle loaded", every report below it is a 503, and reporting six 503s is worse than reporting the one cause. On that path `load` sets the banner to the service's own reason plus the command that fixes it, and returns without fetching anything else.

**A `runId` of `null` is passed straight through, not resolved first.**
`null` means the active run, which is exactly what the report endpoints mean by an absent `run_id`. Resolving it to a concrete id would cost a round trip before the first paint.

**Each settled promise is unwrapped exactly once.**
`value(settled, name)` records a failure into `problems` as a side effect. Calling it twice on the same rejection would report the same missing report twice in the banner. This is why `identity` and `history` are pulled out into locals before the `state` literal rather than being read inline in two places.

**A 404 from `/api/comparison` is explicitly carved out of `problems`.**
It is not a missing report; it is the ordinary state of a tree whose runs were trained rather than compared. Left in, it would turn the status pill amber and claim the run was incomplete on every default install. Any *other* failure from that endpoint is a real one and is reported like the rest.

**`/api/comparison` is fetched without a `run_id`.**
A comparison describes several runs at once and lives at the report root, so it does not change when the picker does. The panel says so in its own note.

**`paint` installs the `aria-label` and the `<details>` from the same call that draws the pixels.**
Not inside the renderer, and not from a separate lookup - so a panel *cannot* forget its accessible alternative. The caption text and the `aria-label` are the same string, which is also why every renderer's label states the headline reading rather than describing a picture.

**`driftPartitions` reads the compared partition names from `/api/metrics`, and defaults to `"reference"`/`"comparison"`.**
The chart is told, not guessing. A chart captioned "train against test" when nothing in the payload said so would be an invention, and the honest default keeps the axis truthful when metrics is the report that failed to load.

**The run picker labels every option with the tier as well as the model.**
`riskscore compare` publishes every variant within the same second, and `shortRunId` keeps the stamp only to the minute - so two runs differing solely by tier produced two options spelled identically and choosing between them was a coin flip. The active run is marked `· active`, because "which one is serving `/predict`" is a different question from "which one am I looking at". The probe asserts the option labels are distinct.

**The picker is disabled with fewer than two runs**, rather than being hidden - a control that vanishes is a feature a reader cannot find.

**Switching runs uses `replaceState`, not `pushState`.**
Switching runs is filtering a view. Filling the back button with report views is not what a reader means by back. Selecting the active run drops the query string entirely, so the URL is clean by default and only carries `?run_id=` when it is pinned to a past run.

**The row-flow chips carry no colour.**
Every stage in the funnel removes rows, so four orange numbers in a row would read as four warnings. The sign carries the direction on its own. The arrow between the funnel and the split chips is `aria-hidden`, because the reading order already conveys the sequence and "right arrow" between two counts adds nothing.

**`rows_unknown_maturity` is shown even when it is zero, and especially then.**
It counts loans whose term or issue date could not be read, which the embargo has to drop because it cannot prove they matured. A silent zero is the difference between "no rows were unreadable" and "nobody checked".

**The embargo summary is the manifest's own sentence, verbatim.**
It is what `run.log` records and what the model card quotes. Rewording it here would give a reader two versions of one fact to reconcile.

**The comparison panel is present even when nothing has been compared.**
It says so and names the command. Hiding it would leave a reader with no reason to believe the choice between two models was measured at all.

**"Same split" is an integrity check, not decoration.**
Every delta in the comparison table is a difference between two numbers computed on the same rows. If two variants disagree on their row counts, the deltas are comparing two different questions, and the panel says `no - the variants disagree on row counts` instead of printing them as a result.

**The best variant is named the way the table names it.**
`variantName(bestRow)` rather than the raw `logistic_regression / with_lender_priced`. The same variant spelled two ways in two panels reads as two variants.

**Which features are categorical comes from the manifest, never from the column count.**
Every numeric feature happens to expand to exactly two columns under the current preprocessor, so inferring from the count would work today. It is a property of the preprocessor, not a fact about the data, and the report would start lying the first time a numeric feature gained a binned encoding.

**A zero-magnitude feature is reported as zero, not as "small".**
A feature scores exactly zero when it takes one value across the whole training window, which is a fact about the split rather than about the feature - and the note says exactly that.

**Artifact links are run-scoped.**
`/artifacts/{run_id}/{name}`. The old links pointed at `/artifacts/metrics/logistic_regression_metrics.json`, a layout that stopped existing when runs became directories, so every one of them was a 404 on the page. No `target="_blank"`: a CSV downloads and a PNG renders, and opening either in a new tab is the reader's decision to make with a modifier key.

**The sanity banner exists because of a specific committed artifact.**
The version in this repository before the rewrite reported an AUC to four decimals on a test set of 116 rows containing one default, with no caveat. A metric on 116 rows is not wrong, it is noise, and the dashboard is the last place that can say so before somebody quotes it. The positive count is derived from the test row count and the default rate rather than being read directly, because a run predating the field would otherwise get no warning.

**`syncThemeButton` reads state and never writes it, and that separation is load-bearing.**
An `applyTheme(effectiveTheme())` at boot writes the OS preference into `localStorage`, which pins the theme to whatever the machine happened to be on the first visit and stops the page following the OS ever again. So boot calls the read-only sync; only a click calls `applyTheme`.

**The button label names the action, not the state.**
It reads "Dark theme" while the page is light. `aria-pressed` carries the state.

**A theme change ends in `redraw()`.**
Canvas has no cascade: the colours were baked into pixels at draw time. The CSS follows a theme change; the charts cannot, until they are redrawn.

**The OS-preference listener only acts while no explicit choice exists.**
`if (document.documentElement.dataset.theme) return`. Without the listener, a machine switching to dark at sunset leaves the charts drawn in the light palette on a dark page. Without the guard, the OS would override a reader's explicit choice.

**One `observeResize` over all five canvases**, not one per chart. It fires on a window resize, on a panel reflowing, and once when the layout first settles - which is the moment a canvas finally has a width to be sized against, and therefore the first real draw.

**`mountScorePanel` is not awaited.**
It needs only `/api/schema`. A reader who came to try a prediction should not wait on ten report fetches for the form to appear.

**`mountRetrainPanel` is mounted after the first load, and exactly once.**
After, because whether the panel exists at all is a fact the service reports in `/readyz`, which `load` reads. Once, because `load` runs again on every run switch and a listener added per switch is a retrain per click.

**A retrain's `onDone` clears the URL before reloading.**
A reader who arrived on `?run_id=<old>` and then retrained would otherwise be shown the new active run under a link to the previous one - and would share that link. It then reloads with `null` rather than the id the job reported: they are the same run, and asking for the active one keeps the page pointed at whatever is being served rather than pinned to an id.

**`boot`'s `catch` reports a render bug, not a failed request.**
`load` handles request failures itself, so anything reaching that handler is a bug in this file. It says so, sets the error state, and rethrows - so the console keeps the stack and the probe's console-error assertion fails.

**`reload` invalidates the whole cache first.**
A retrain changes what every panel on the page describes, so a partial invalidation would need to know which reports a retrain can change. The answer is all of them.

## What must NOT live here

- **`fetch`.** Every request goes through `api.js`, which is what makes the caching, de-duplication, timeouts, and error shape universal.
- **`document.createElement`.** `dom.js` builds nodes. A node built here would miss the scopes and the tone-to-token convention.
- **Canvas drawing or tick arithmetic.** `charts.js` and `panels.js`.
- **Any recomputation of a reported figure.** The panels read what the server computed. The one derived value in this file is the sanity banner's positive count, and it is derived because the field it would prefer does not exist in older manifests.
- **A second definition of the theme storage key.** It is spelled here and in `index.html`'s inline script, and that duplication is documented in [ADR 0009](../decisions/0009-vanilla-dashboard.md) as the deliberate exception - the inline script must run before first paint and cannot import.
- **Fetching inside `redraw`.** This is the property the file exists to guarantee.

## Related tests

There is no `main.test.js`, deliberately, and for the same reason as `dom.js`: this file is wiring. A unit test of wiring asserts that the wiring is what it is, and the interesting failures here are ordering and lifecycle - the theme not persisting at boot, a listener added twice, a redraw that fetches - none of which a stubbed-DOM test observes honestly.

The pure logic it would otherwise contain has been pushed into modules that *are* tested: `embargoRows`, `importanceRows`, `comparisonRows`, and `variantName` in `panels.test.js`; every formatter in `format.test.js`; `jobLine` and the mount guard in `retrain.test.js`; caching and polling in `api.test.js`.

What covers this file is `scripts/probe_dashboard.mjs`, in real Chrome at six widths in two themes, twelve configurations. Per configuration it asserts: the status region reaches a non-loading state within 15s, `identityRows === 6`, `flowChips` present, `artifacts === 12`, `embargoFacts === 4`, every metric card non-empty (`"-"` in any card is a failure), every canvas sized with a real bitmap and an `aria-label` that is no longer the loading placeholder, every `.data-fallback table` non-empty, no console error, and no horizontal overflow - attributed to the widest offending element by selector rather than reported as "the page scrolls".

Three of its probes exercise this file's lifecycle specifically:

- The load wait reads `#status`'s `data-state`, which is only set once every fetch has settled and every panel has rendered - so it is the readiness signal, and a `render()` that threw halfway leaves it at `loading` and fails the run.
- `SWITCH` picks another option, dispatches `change`, and waits for the identity strip to show the run it asked for - which is the whole run-history feature end to end, including `replaceState` and the second `load`. A run present in the registry whose reports are missing produces a page of dashes and an amber pill, and that is indistinguishable from a fresh load in a screenshot. It also asserts the option labels are distinct, which is the fix for two compare runs sharing a minute.
- `retrainVisible` is compared against the target server's own `/readyz`, so the mount decision is checked in both directions.

The server side of every payload this file reads is tested in `tests/test_api.py` and `tests/test_routes_admin.py`.

## Known limits

- **No unit tests, by the argument above.** The consequence is real: a lifecycle regression that the probe's assertions do not happen to cover, and that does not throw, would pass CI - because the probe is not in CI. It needs Chrome and a running server with a published run.
- **`state` is module-global and reassigned.** Two loads racing - a fast run switch during a slow load - would have the later assignment win, which is correct, but the earlier `render()` may already have run against the earlier state. In practice the picker is the only trigger and a second change during a load is rare. An abort signal per load, cancelling the previous one, is the fix if it ever matters.
- **A load fetches ten reports every time, including on a run switch back to one already seen.** `api.js` caches by path including the query string, so the second visit is served from cache, but the ten promises are still created.
- **`ARTIFACTS` is a hardcoded list of twelve names.** A run directory that gains a file gets no link until this list is edited, and one that lacks a file gets a link that 404s. The allowlist on the server is the authority; this is a second copy of part of it. A `GET /api/runs/{id}` field listing the actual files would remove the duplication.
- **The sanity thresholds are hardcoded here, not in the manifest.** 1000 rows and 50 positives are judgements this page makes about the server's output. Reasonable, but a reader cannot see where they came from without reading this file - which is what this page is for.
- **No polling of `/readyz` after boot.** A service that loses its bundle while the page is open shows a stale header badge until a reload or a run switch.
