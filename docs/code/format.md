# `dashboard/js/format.js`

## Purpose

Make **absent** look different from **zero**.

Every number on the dashboard arrives as JSON from an API whose payloads are built from pandas frames, and pandas has more ways to say "no value" than JavaScript has to notice: `null` from a `NaN`, `""` from an empty CSV cell, and a numeric string from a column that was read as text.
The old dashboard formatted with `Number(value).toFixed(3)`, so a missing Brier score rendered as `0.000` - a confident, wrong, perfectly plausible reading of a model that was never measured.
On a credit-risk page that is the worst possible failure: a metric of zero is the *best* achievable Brier score, so the display said "flawless" where the truth was "unknown".

This module is the single funnel every displayed value goes through.
It is pure - no DOM, no imports - so the "absent is not zero" rule is testable directly, and it is tested exhaustively against the values that actually break naive coercion.

## Public API

| Name | What it is |
| --- | --- |
| `MISSING` | `"-"`. The one spelling of absent on the whole page. |
| `isMissing(value)` | True for `null`, `undefined`, `""`, any non-number-non-string type, and any non-finite number. |
| `number(value, digits = 3)` | Fixed-digit decimal. |
| `percent(value, digits = 1)` | Scales by 100 and appends `%`. |
| `count(value)` | Grouped integer, no decimals. |
| `signed(value, digits = 3)` | Explicit `+` or `-`, and no signed zero. |
| `cost(value)` | Grouped integer, unitless. |
| `timestamp(value)` | An ISO instant to `YYYY-MM-DD HH:MM UTC`. |
| `shortRunId(runId)` | `20260824T1432Z-lr-oo-abc1234` to its date, minute and commit. |
| `PSI_MODERATE`, `PSI_SIGNIFICANT` | `0.1` and `0.25`. |
| `psiBand(value)` | `{label, tone}` for a PSI figure. `tone` is a token name, not a colour. |
| `labelize(name)` | `snake_case` to Title Case, initialisms left upper. |
| `toRows(columns)` | A columnar payload to an array of row objects. |
| `numericColumn(values)` | `{values, dropped}` - the finite entries, and how many were not. |

## Inputs and outputs

Strings and numbers in, strings out.
`toRows` and `numericColumn` take and return plain arrays and objects.
Nothing here reads the DOM, touches `window`, fetches, or throws.

Every formatter returns `MISSING` rather than throwing, because a formatter that throws takes the whole panel down over one bad cell, and a panel of dashes with the rest of the page intact is strictly more useful than an empty page.

`Intl.NumberFormat` instances are cached in a `Map` keyed on `JSON.stringify(options)`.
Constructing one is expensive relative to formatting with it, and a full render formats a few hundred values.
The locale is deliberately left `undefined` so grouping and the decimal separator follow the reader's browser.

## Invariants and failure modes

**`isMissing` checks the type before it checks the value, and that order is load-bearing.**
`Number([])` is `0`, `Number(true)` is `1`, `Number(null)` is `0`, and `Number(" ")` is `0`.
A guard written as `!Number.isFinite(Number(value))` therefore passes an empty array through as a hard zero.
So the gate is: nullish or empty string, then `typeof` is neither `"number"` nor `"string"`, then finiteness.
A numeric *string* is accepted on purpose - a CSV-derived payload sends `"0.7123"` - but an array, an object, and a boolean are not.

**A real zero still formats as zero.**
This is the other half of the same invariant and the reason it needs its own test.
An approval rate of exactly 0, a contribution of exactly 0, a cost of exactly 0 are all legitimate readings and must not become dashes.

**Digits are pinned, not trimmed.**
`number(0.5)` is `0.500`, not `0.5`.
A column of metrics where the digit count varies per row is unreadable in a table, and `minimumFractionDigits` is set equal to `maximumFractionDigits` for exactly that reason.

**`percent` multiplies rather than using `style: "percent"`.**
`Intl`'s percent style would be the obvious choice, but it applies locale-specific spacing before the `%` and would put the sign in a locale-specific place.
Every percentage on this page appears in a table column of its own width, so a stable `12.3%` beats a locally idiomatic one.

**`signed` never emits `+0.000`.**
An exactly-zero contribution is not a positive contribution.
A tornado plot where a zero-weight feature is labelled `+0.000` reads as a small push in the approve direction, which is a claim the model did not make.

**`cost` is unitless.**
The cost matrix in this project is a *ratio* - the relative price of a false negative against a false positive - not currency.
Formatting it with a currency symbol would invent a unit the model never had, and would then have to invent a currency.

**`timestamp` renders in UTC, always.**
The run id embeds a UTC instant.
If the generated-at line rendered in local time, the two identifiers for the same run would disagree by hours, and somebody would eventually conclude they were different runs.
The formatter takes the ISO string's own fields via `Date.prototype.toISOString`, so there is no timezone database involved and no DST edge.

**An unparseable instant is a dash, not `Invalid Date`.**
`new Date("nonsense").toISOString()` throws a `RangeError`; `String(new Date("nonsense"))` is the literal text `Invalid Date`.
Both are worse than a dash.

**`shortRunId` returns anything that is not run-id-shaped untouched.**
It matches the `<instant>-<model>-<tier>-<sha>` shape and, failing that, hands the string back.
A future run-id format change therefore degrades to a long label rather than to a truncated one that drops the part that distinguishes two runs.

**`psiBand` returns a token *name*, not a colour.**
`{tone: "warn"}` becomes `color: var(--warn)` at the point of use, so the palette stays in `tokens.css` and dark mode follows without this module knowing a theme exists.
The bands are inclusive at their lower bound - exactly 0.1 is moderate, exactly 0.25 is significant - matching how the thresholds are stated in the drift literature and in `drift.py`.

**`labelize` leaves initialisms upper.**
`auc_roc` becomes `AUC ROC` and not `Auc Roc`, via an `INITIALISMS` set.
Otherwise every metric label on the page reads as a typo.

**`toRows` truncates to the shortest column.**
A columnar payload with columns of unequal length is a server bug, but the client's job is to render what it can.
Zipping to the longest would produce `undefined` cells that then have to be guarded everywhere downstream; zipping to the shortest produces fewer rows, which is visible and safe.
A payload that is not columnar at all yields no rows rather than throwing.

**`numericColumn` counts what it drops.**
A chart that silently skips non-finite points is a chart that lies about its sample size, so the count comes back with the values and the panel puts it in the caption.

## What must NOT live here

- **The DOM.** This module is imported by `panels.js`, `score.js`, and `main.js`, all of which build nodes. If a formatter returned a node, none of these tests could run under plain `node --test`.
- **Domain thresholds other than PSI's.** The PSI bands are here because they are a *display* decision - which of three words to print. The approval-rate floor, the cost matrix, and the decision threshold are model decisions and live in `src/risk_score/`.
- **Colours.** `psiBand` names a token. A hex literal here would be invisible to the theme.
- **Fetching, caching, or anything asynchronous.** Every function here is synchronous and total.

## Related tests

`dashboard/js/format.test.js`, 19 tests.
Run with `node --test` from `dashboard/`.

- `every kind of absent value formats as a dash, never as zero` is the reason the module exists. It runs `null`, `undefined`, `""`, `NaN`, `Infinity`, `[]`, `{}`, and `true` through every formatter. `[]` and `true` are the two that a `Number()`-based guard passes.
- `a real zero still formats as zero` is its inseparable counterpart. Without it, "return a dash when in doubt" passes the first test and destroys legitimate data.
- `a numeric string is accepted, because CSV-derived payloads send them` pins the deliberate exception to the type check.
- `digits are pinned rather than trimmed` and `percent scales by 100 and keeps one decimal by default` fix the output shapes tables depend on.
- `a positive contribution carries an explicit plus and a negative a minus` and `an exact zero contribution is not signed` are the two halves of `signed`.
- `an instant renders in UTC to the minute, matching the run id` and `an unparseable instant is a dash rather than Invalid Date`.
- `a run id shortens to its instant and its commit` and `anything that is not a run id is returned untouched`.
- `psi bands are inclusive at their lower bound` tests exactly `0.1` and exactly `0.25`, which is where an off-by-one comparison operator hides. `a psi band names a token rather than a colour, so dark mode follows` asserts the return value is not a hex string.
- `a snake_case name titles, with initialisms left upper`.
- `columns become rows`, `columns of unequal length truncate to the shortest`, `a non-columnar payload yields no rows instead of throwing`, and `non-finite entries in a column are counted, not silently dropped`.

## Known limits

- **No locale is chosen, so grouping varies by reader.** `1,234` and `1.234` are the same number to two different browsers. Correct behaviour, but it means a screenshot from one machine does not match another byte for byte.
- **`timestamp` is minute precision only.** Two runs in the same minute render identically here; `shortRunId` disambiguates them by commit, and the run picker uses that. Two runs in the same minute *at the same commit* - which `riskscore compare` produces - are distinguished only by the tier, which is why the picker label appends `variantName` rather than relying on this.
- **`labelize`'s initialism list is hand-maintained.** A new metric named with an unlisted initialism renders as Title Case. The failure is cosmetic and visible on the page.
- **No unit awareness at all.** `number` does not know that a Brier score and an AUC have different sensible digit counts; callers pass `digits`. A per-metric format map exists, but in `panels.js` (`METRIC_FORMATS`) where the metrics are known, not here.
