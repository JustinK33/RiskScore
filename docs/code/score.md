# `dashboard/js/score.js`

## Purpose

Score one applicant.
This is the panel the whole project exists to make possible.

Before the FastAPI service there was no inference path at all: `joblib.dump` wrote a pickle that nothing loaded, and the dashboard's only POST retrained the model rather than scoring anybody.
A credit-risk project that cannot score a single applicant is a report generator.
This panel is the demonstration that it can.

**The form is generated from `GET /api/schema`, never written by hand.** That is the entire point of the endpoint: the fields, which are required, their bounds, and the categories the *fitted* encoder actually saw all come from the bundle being served. Retrain on a different feature tier and the form changes, rather than silently accepting inputs the model ignores. A hand-written form would drift from the model on the first retrain and there would be no signal that it had.

Two more decisions.

**Categoricals render as `<select>`, not text.** `OneHotEncoder` is configured `handle_unknown="infrequent_if_exist"`, so `"rent"` for `"RENT"` is accepted, pooled into the infrequent bucket, and scored - a plausible-looking default probability produced by a typo, with no error anywhere in the stack. A closed choice list makes that unreachable from this page. It is the single most consequential UI decision in the dashboard, because the failure it prevents is invisible.

**Reason codes are a CSS tornado plot, not a canvas.** Five rows of a signed number with a long label is a table. Drawing it would mean rotated text, a fallback table beside it, and two representations to keep in step. As `<div>`s with percentage widths the bars reflow, are readable at 320px, and every number is present as text.

## Public API

| Name | What it is |
| --- | --- |
| `EXAMPLE_APPLICANT` | A plausible applicant for the demo button, in the *control's* dialect. |
| `coerce(field, raw)` | One raw form value to something `/predict` accepts. |
| `hint(field)` | The sentence under one control. |
| `buildForm(schema)` | `{node, read, fill}`. The caller never touches an input directly. |
| `reasonBars(reasons)` | Each reason plus `{increases, width}`. Widths relative to the largest contribution. |
| `renderPrediction(prediction)` | The verdict block and the reason-code table, as nodes. |
| `mountScorePanel({form, body, banner, result, exampleButton, submitButton})` | Fetch the schema, build, wire submit. |

## Inputs and outputs

`GET /api/schema` in, a form out; a form's values in, `POST /predict` out, a verdict block in.

Imports `getSchema` and `predict` from `api.js`, three node helpers from `dom.js`, and three formatters from `format.js`.

`buildForm` returns three things and no inputs:
- `node` - the `.form-body`, one `<fieldset class="field-group">` per feature tier.
- `read()` - everything the user actually supplied, coerced. Blank fields are **absent from the object**, not present as `null`.
- `fill(values)` - prefill from a plain object, silently ignoring names the schema does not have.

Tiers are ordered by `TIER_ORDER = ["timeline", "loan_request", "borrower", "lender_priced"]` so the form reads chronologically - when the loan was issued, what was asked for, who asked, and then anything the lender priced.
A tier not in the list sorts last rather than being dropped.

## Invariants and failure modes

**A blank field is `undefined`, never `0` and never `null`.**
`coerce` returns `undefined` for `""`, and `read()` omits those keys entirely.
Sending `0` for a blank `delinq_2yrs` asserts a fact the user never entered; for `annual_inc` it asserts a false one and would produce a confidently wrong decline.
The imputer on the server handles a genuinely absent optional field, and it can only do that if the field is absent.

**A zero the user actually typed survives.**
The other half of the same invariant, and the reason both have tests.
"Return undefined when in doubt" passes the first and destroys real data.

**An unparseable numeric is forwarded, not dropped.**
`coerce` passes a non-numeric string through so the server's 422 names the field.
Dropping it would score the applicant as if the field were blank and then report a decision built on less than the user supplied - which is the worst of the three options, because it looks like success.

**A categorical keeps its exact case.**
No normalization here.
The `<select>`'s options are the training vocabulary verbatim, so the value submitted is a value the encoder saw.
Any case folding in this file would be a second opinion about what the encoder accepts.

**A required `<select>` has no blank option; an optional one does.**
`control()` prepends `<option value="">not supplied</option>` only when `field.required` is false, so a required select cannot be submitted empty and the browser's own validation catches it before a request is made.

**`type="month"` for dates, because both date fields are month-precision in the source.**
`issue_d` and `earliest_cr_line` are `Jun-2015`-shaped in the extract.
A day picker would invent precision the data does not have.
`type="month"` yields `2015-06`, which is `%Y-%m`, one of the formats `DEFAULT_DATE_FORMATS` accepts.

**`step="any"` on every numeric, not a per-field step.**
`dti` is fractional and `open_acc` is an integer.
A wrong step makes the browser reject a valid value with a message about the nearest allowed one, which reads as a bug in the form.

**Every control has an `aria-describedby` pointing at its hint**, and every label a real `for`. The required marker is an `<abbr title="required">`, so it is announced rather than being a bare asterisk.

**A categorical's hint states where its list came from.**
"17 values appeared in training; anything else was never learned." A reader who wants Wyoming and finds seventeen states needs to know the list is the *fitted vocabulary*, not a validation rule somebody chose - and specifically needs to know that `/predict` would accept `"WY"`, pool it, and score it.
That is exactly why the page does not offer it.

**`EXAMPLE_APPLICANT` is in the control's dialect, deliberately.**
`36` not `" 36 months"`, `62.5` not `"62.5%"`, `"2015-06"` not `"Jun-2015"`.
`CanonicalizeFrame` accepts either, but an `<input type="month">` renders *blank* for anything that is not `%Y-%m` and a number input rejects a percent sign - so a prefill in the raw dialect would half-load and look broken.
It also carries `fico_range_low`/`high`, which the served bundle's schema offers but neither real extract in `data/raw/` contains; `fill` ignores names the schema does not have, so a bundle trained on real data simply leaves them out.

**`fill` writes `""` for a name it has no value for**, so a second click of the example button after a manual edit produces the example and not a mixture.

**Reason-code widths are relative to the largest absolute contribution, not to the total.**
The total log-odds is dominated by the baseline, so scaling to it renders all five reasons as slivers.
The chart's job is to rank and compare drivers.

**A contribution too small to see still gets a visible mark: a 2% floor.**
A real contribution rendered at zero width is a reason the reader cannot see.
The probe asserts this directly - it counts `.bar` nodes that are laid out and measure under 1px, and requires zero of them.

**An all-zero set of reasons draws no bars rather than dividing by zero.**
`largest === 0` yields width 0 for every row, and the log-odds column still carries the numbers.

**A missing `log_odds` is treated as no contribution, not as `NaN`.**
`Number(reason.log_odds) || 0`.
A `NaN` width becomes `width: NaN%`, which the browser drops, leaving a row with no bar and no explanation for why.

**The tornado grid is a `<div>` inside the `<td>`, not the `<td>` itself.**
`display: grid` on a `<td>` stops it being a table-cell, so the browser wraps it in an anonymous cell and the row's bottom borders get drawn at two different heights.
Found by looking at it; the fix is one wrapper.

**The bar column is paired with its header by class, not by `nth-child`.**
`.tornado-cell` and `.tornado-head` are both hidden below 640px.
An `nth-child` selector would target the wrong column the moment a column is added, and the failure would be a hidden log-odds value rather than a hidden plot.

**The verdict shows both log-odds figures, because the difference between them is the explanation.**
The baseline is what the model says about nobody in particular; the reasons below sum to the gap.
Showing only the total makes the reason codes look like an unrelated list.

**Submit is the only submit path, and validation is not disabled.**
The form has no `action` and no `method`, and the handler calls `preventDefault` - letting the browser navigate would lose the page.
`novalidate` is deliberately *not* set, so required fields and numeric bounds are enforced by the browser before a request is made.

**The submit button is disabled for the duration and re-enabled in a `finally`.**
A double submit is two scores and two `latency_ms` values racing into one result block.

**Errors go to the panel's own banner, not the page-level one.**
A rejected applicant is a local problem.
Reporting it in the header would look like the dashboard breaking.
A 422's `detail` names the offending field and is shown verbatim, because it is the most useful string on the page at that moment.

**A schema that will not load aborts the mount with a banner and no form.**
Half a form built from a partial schema would submit a partial applicant.

## What must NOT live here

- **A hand-written field list.** The moment a field name is spelled in this file rather than read from the schema, the form can disagree with the model. `EXAMPLE_APPLICANT` is the one exception and it is guarded by a test asserting it covers every required field.
- **Any judgement about what the score means.** Approve/decline comes from the response's `decision`, computed against the threshold the bundle carries. This file does not compare a probability to a number.
- **The threshold, the cost matrix, or the tier definitions.** All server-side.
- **Canvas drawing.** The tornado plot is CSS. If it needs a canvas it needs `charts.js`.
- **Case folding, trimming beyond whitespace, or any input normalization.** The encoder's vocabulary is exact.

## Related tests

`dashboard/js/score.test.js`, 11 tests, over `coerce`, `reasonBars`, and `EXAMPLE_APPLICANT`.

- The four `coerce` tests are the "absent is not zero" rule at the input boundary, mirroring `format.js`'s at the output boundary: `a blank field is absent, not zero`, `a numeric field becomes a number`, `a zero the user actually typed survives`, `an unparseable numeric is forwarded so the server can name the field`, and `a categorical keeps its exact case`.
- The four `reasonBars` tests: `bars scale to the largest absolute contribution, not to their sum`, `a contribution too small to see still gets a visible mark`, `an all-zero set of reasons draws no bars rather than dividing by zero`, and `a missing log_odds is treated as no contribution, not as NaN`.
- `the example applicant supplies every field the model requires` asserts against the real schema's required set. Without it, a retrain on a tier with a new required field leaves the demo button producing a form the browser refuses to submit - and the failure appears in the demo, in front of an audience.
- `the example uses month-precision dates, which is a format the parser accepts` pins the dialect decision so a "helpful" change to `Jun-2015` fails here rather than rendering a blank date input.

The end-to-end proof is `scripts/probe_dashboard.mjs`'s `SCORE` probe, which is the only automated check that the *generated* form is submittable at all.
It clicks the example button, runs `checkValidity()` over every element and names any that fail, calls `requestSubmit()`, and waits for a verdict - so a control whose `value` does not round-trip produces either a browser validation block or a 422, both of which a unit test cannot see.
Per width and theme it also asserts `scoreFields >= 4`, `scoreSelects >= 1` (a text box where a select belongs is the typo-pooling hole, and it looks identical in a screenshot), a non-empty `verdict` decision, `reasonRows > 0`, and zero zero-width bars.

The server side is `tests/test_scoring.py` and `tests/test_api.py`.

## Known limits

- **One hardcoded example applicant.** Marked `ponytail:` in the source. Several archetypes - a thin file, a high-DTI case, a clear approve - would make a better demo, and the upgrade is a list plus a `<select>` in place of the button.
- **No batch scoring in the UI.** `/predict/batch` exists and takes up to 1000 rows; this panel scores one applicant. A CSV drop zone would need a preview table, per-row error reporting, and a results download.
- **No client-side range feedback.** The browser's own `min`/`max` validation is the only bound check, so out-of-range input is reported on submit rather than as it is typed.
- **The reason count is fixed at the API default of 5.** `predict` accepts `topK`; nothing on the page sets it.
- **`fill` stringifies everything.** Fine for the current field kinds, but a future boolean or multi-select field would need its own branch.
- **The tornado bars are hidden below 640px.** The log-odds column is the reading at those widths, which is correct but less immediate. Rotating or stacking the plot was rejected as more CSS than the narrow case is worth.
