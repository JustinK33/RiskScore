# `dashboard/index.html`

## Purpose

The document.
Every element the JS writes into already exists here, named, in reading order.

That is the one property the file is built around: **the structure is static and the content is not.** No panel is created by JS, no section is appended, nothing is inserted after load.
`main.js` fills nodes that are already in the document, which is what makes the reading order, the heading hierarchy, and the label-to-control associations reviewable by reading one file instead of tracing string concatenation across five.

It is also the page's accessibility contract, and it is written here rather than applied later.
Every canvas is a `role="img"` with a `<figcaption>` and a `<details>` fallback beside it before any data exists.
Every form control has a real `<label for>`.
The status region is a `role="status"`.
A renderer cannot forget any of it, because none of it is the renderer's job.

The prose is part of the file, not decoration.
Each section carries a `.section-note` stating what the numbers below it mean and which partition they came from - "Every number here comes from the test partition, scored once with the frozen model, calibrator, and threshold.
The threshold and the calibrator were fitted on validation." A dashboard that reports a metric without saying which rows produced it is how the previous version came to present an AUC on 116 rows as a result.

## Public API

The ids `main.js`, `score.js`, and `retrain.js` look up.
Renaming one here breaks a lookup there and nothing else.

| Group | Ids |
| --- | --- |
| Chrome | `content`, `themeToggle`, `themeToggleLabel`, `status`, `healthBanner`, `sanityBanner` |
| Identity | `runSelect`, `runSelectHint`, `identityGrid`, `rowFlow` |
| Metrics | `aucRoc`, `averagePrecision`, `ksStatistic`, `brierScore`, `brierScoreNote`, `ece`, `eceNote`, `defaultRate`, `approvalRate`, `selectedThreshold`, `selectedThresholdNote` |
| Charts | `calibrationChart`/`Caption`/`Fallback`, `thresholdChart`/…, `embargoChart`/…, `vintageChart`/…, `scorePsiChart`/… |
| Tables | `comparisonTable`, `featurePsi`, `importanceTable` |
| Side panels | `runDetails`, `artifactLinks`, `comparisonFacts`, `comparisonNote`, `embargoFacts`, `embargoSummary`, `driftFacts`, `driftNote`, `importanceFacts`, `importanceNote` |
| Score | `scoreForm`, `scoreBanner`, `scoreFields`, `scoreSubmit`, `scoreExample`, `scoreResult` |
| Retrain | `retrainSection`, `retrainForm`, `retrainBanner`, `retrainDataset`, `retrainModel`, `retrainTier`, `retrainKey`, `retrainSubmit`, `retrainProgress` |

Section order: identity, test-set metrics, calibration + threshold + run details, comparison, embargo + vintage, drift, importance, score, retrain.

## Inputs and outputs

Loads `/styles/tokens.css`, then `/styles/app.css`, then `/js/main.js` as a module.
Served by the API's static mount, so those paths are absolute from the service root.

Reads `localStorage["riskscore.theme"]` in one inline script.
Nothing else here touches the network or storage.

## Invariants and failure modes

**One inline script, and it has to be inline and blocking.**
`main.js` is a module and therefore deferred, so a stored dark preference applied from there lands *after* first paint - a white flash on every load, on every visit, for every reader who chose dark.
The script reads one string and sets one attribute.
It is wrapped in `try`/`catch` because storage can be blocked outright, in which case the OS preference still applies through `prefers-color-scheme`.

This is the one place the theme storage key is spelled twice.
[ADR 0009](../decisions/0009-vanilla-dashboard.md) records it as the deliberate exception: an inline script cannot import.

**Two `<link>`s, not one `@import`.**
`tokens.css` must be parsed before `app.css` resolves its `var()`s, but an `@import` inside the first sheet is only *discovered* after that sheet has parsed - two serialized round trips instead of two parallel ones.
The file says so at the point of the decision.

**The favicon is an inline SVG data URI.**
No second file, no 404 in the console, no request.
The accent green is a literal because `currentColor` is not available to a favicon - the one hardcoded colour in the project, and it is outside the cascade by definition.

**`.skip-link` is the first focusable element in the document.**
Followed by the theme toggle and the run picker.
Tab order is document order, and the document order is the reading order, because nothing is reordered by CSS `order` or by `grid-row`/`grid-column` in a way that moves an interactive control.

**`#status` is `role="status"`.**
That is an implicit `aria-live="polite"` region, so a load failure is announced without stealing focus.
The old dashboard wrote every failure into a plain `<div>`, which a screen reader never saw.
`#retrainProgress` is the same, which is what makes a fit that takes minutes audible.

**Both banners ship `hidden` and are un-hidden by JS.**
The `hidden` property, not `display: none` - see [dom.md](dom.md).
An empty banner that is merely styled invisible is still in the accessibility tree.

**Every canvas has `role="img"` and a placeholder `aria-label` ending in `, loading`.**
The real label is installed by `main.js`'s `paint`, from the data.
The placeholder matters twice: it means a canvas is never an unlabelled image even before the first draw, and the probe asserts no `aria-label` still says `loading` after the page settles - so a chart that failed to render is caught by the label it never replaced.

**Each canvas is inside a `<figure>` with a `<figcaption>` and an empty fallback `<div>` as siblings.**
The `<details>` table goes in the `<div>`.
Present in the markup before any data exists, so the alternative to a picture is structural rather than something a renderer remembered to add.

**Every metric card starts at `-`, never at `0` or blank.**
`-` is `format.js`'s MISSING, so the initial state and the absent state are the same glyph and neither can be read as a measurement.
The probe fails on any card still showing `-` after load, which makes the placeholder its own test.

**Three metric cards have ids on their `<small>` notes.**
`brierScoreNote`, `eceNote`, `selectedThresholdNote`.
The static text is the general case; `main.js` replaces it when the run has something more specific to say - which partition the threshold came from, how many bins the ECE used.
The default text is correct on its own, so a run that reports nothing extra is not left with a gap.

**`#retrainSection` ships `hidden`, and the markup says why.**
Not the panel inside it - the whole section, including its `<h2>` and its note, so a switched-off feature leaves no orphan heading in the document outline.

**The retrain form reuses the score form's generated structure by hand.**
`fieldset.field-group` wrapping `div.field > (label, control, small.hint)`.
So it inherits the subgrid alignment and needs no CSS of its own, and a spacing change to the generated form moves both.
Its labels are the request's own field names - `dataset`, `model_type`, `feature_tier`, `api_key` - which is what the score form does, so a reader sees the API's vocabulary in both places.

**`#retrainModel` is empty in the markup and filled by `retrain.js`; `#retrainTier` is not.**
The models are the service's own `MODEL_TYPES`, so writing them here would let the markup drift from what the service accepts.
The two tiers are a closed set the page already names in four other places, and both are spelled in the option values the API takes.

**`required` and `accept=".csv,text/csv"` are on the file input, and the API re-checks both.**
Browser validation is a convenience for the reader.
The size cap, the CSV sniff, and the key check are the server's, because the form is not the only client.

**The API key field is `type="password"` with `autocomplete="off"`.**
So it is not shoulder-readable and not offered back by the browser on a machine somebody else uses.
Where it goes afterwards - `sessionStorage`, this tab only - is stated in the hint under the field rather than left to the runbook.

**Every `.section-note` names its partition or its source command.**
Test-set metrics say test and say the threshold came from validation.
The comparison note says `riskscore compare` and says that measuring activates nothing.
The importance note says the training sample and says the units are log-odds.
The score note says `POST /predict` and says blank optional fields are imputed rather than zeroed.

**`<code>` for every command and field name.**
`riskscore activate`, `POST /predict`, `RISKSCORE_API_KEY`, `int_rate`.
So a reader can tell a thing they can type from a thing being described.

**The footer links `/docs` and `/readyz`.**
Both served by the same process, which is the sentence above them.
It is the shortest available proof that the dashboard is not a separate server reading the same directory - which is what it was before `scripts/serve_dashboard.py` was deleted.

**`lang="en"`, `<meta name="viewport">` with no `maximum-scale`, and one `<h1>`.**
Pinch-zoom is never disabled.
Headings go `h1` → `.section-heading` `h2` → panel `h2` → `h3` for the artifact list, and every `<section>` is `aria-labelledby` its own heading, so the document outline is the page structure.

## What must NOT live here

- **Any second `<script>`.** The theme script is the exception and the ADR records why. Everything else goes through the module graph, where it can be imported and tested.
- **Inline `style` attributes and inline `<style>`.** The one runtime exception is `score.js` setting a tornado bar's width, which is a measured value and cannot be a class.
- **Inline event handlers.** No `onclick`. Listeners are added in the module that owns the behaviour.
- **Data.** No hardcoded metric, no example number, no run id. Everything numeric arrives from the API. The one committed applicant example lives in `score.js`, where it is documented as being in the control's dialect rather than the API's.
- **Presentational markup.** No `<br>` for spacing, no `<b>`, no empty `<div>`s for gaps. Layout is grid in `app.css`.
- **`target="_blank"`.** Opening in a new tab is the reader's decision with a modifier key.
- **A `defer` attribute.** `type="module"` is deferred by spec, so there is nothing to forget.

## Related tests

No unit tests.
Nothing in `node --test` parses HTML, and asserting that an element with a given id exists is a restatement of the file - it is also the assertion that would pass while the element sat inside a collapsed grid track, unreachable and unreadable.

What verifies it is `scripts/probe_dashboard.mjs`, in real Chrome, and most of its assertions are really assertions about this file:

- **The id contract, by count.** `identityRows === 6`, `artifacts === 12`, `embargoFacts === 4`, `retrainModels >= 2`, every metric card non-empty. A lookup that returned `null` leaves the node it would have filled empty, and each of those counts is the node that would stay empty.
- **Every canvas** sized with a real bitmap and an `aria-label` that is no longer the loading placeholder - the placeholder text is what makes that assertion possible.
- **Every `.data-fallback table`** non-empty, so the `<details>` alternatives are checked as content and not just as markup.
- **No console error**, which catches a module that failed to load or a lookup that threw.
- **No horizontal overflow** at six widths in two themes, attributed to the widest offending element by selector.

The keyboard path is checked by hand: tab from the top reaches the skip link, the theme toggle, the run picker, every score field, and both buttons, each with a visible ring, in document order.

The server side of the static mount - that `/`, `/styles/*`, and `/js/*` are served and that nothing outside `dashboard/` is reachable - is `tests/test_api.py`.

## Known limits

- **440 lines with the section notes in them.** The prose is the largest part of the file and it duplicates parts of `docs/` by necessity: a reader of the page will not open the docs. The mitigation is that each note states a fact the API also reports, so a contradiction is visible on the page itself.
- **No `<noscript>`.** The page renders its chrome and its prose with JS off and every value stays `-`. A `<noscript>` block saying so would be two lines and has not been added; the honest version is that the entire dashboard is a JS client and a server-rendered fallback is a different project.
- **The theme flash is fixed for `localStorage`, not for a slow stylesheet.** The inline script runs before CSS is applied, so the attribute is set in time - but a cold cache still paints unstyled text briefly. Inlining critical CSS would fix it and would reintroduce the duplication the token file exists to remove.
- **The ids are a flat namespace shared with three JS files.** No prefixing convention, so `#status` could plausibly be claimed by two panels. Each is looked up in exactly one place, and `rg '"scoreBanner"'` finds both ends - which is the whole mitigation. Nothing enforces it.
- **The retrain form's structure is duplicated from the generated one.** Written by hand to match what `score.js` builds. A change to the generated field shape has to be mirrored here, and nothing catches a divergence except that it looks wrong.
- **`docs/data-dictionary.md` is referenced in prose but not linked.** The file hint says "in the columns the data dictionary lists" without an anchor, because the docs are not served by the API. Serving `docs/` as static files would fix it and would put the whole documentation tree behind the same port as the model.
