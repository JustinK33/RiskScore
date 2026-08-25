/**
 * Tests for the pure parts of the score panel: value coercion and bar scaling.
 *
 * `buildForm` and `renderPrediction` need a DOM and are covered by
 * `scripts/probe_dashboard.mjs`, which drives the real page in Chrome. The two
 * functions here are the ones where a wrong answer is silent - a blank field
 * becoming `0` scores an applicant on a fact nobody entered.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { coerce, EXAMPLE_APPLICANT, reasonBars, reasonValue } from "./score.js";

const numeric = { name: "annual_inc", kind: "numeric" };
const categorical = { name: "home_ownership", kind: "categorical" };

test("a blank field is absent, not zero", () => {
  assert.equal(coerce(numeric, ""), undefined);
  assert.equal(coerce(numeric, "   "), undefined);
  assert.equal(coerce(categorical, ""), undefined);
});

test("a numeric field becomes a number", () => {
  assert.equal(coerce(numeric, "62000"), 62000);
  assert.equal(coerce({ ...numeric, name: "dti" }, "18.2"), 18.2);
});

test("a zero the user actually typed survives", () => {
  // The distinction the blank rule exists for: `delinq_2yrs: 0` is a claim about
  // the applicant, and dropping it would let the imputer substitute the median.
  assert.equal(coerce({ name: "delinq_2yrs", kind: "numeric" }, "0"), 0);
});

test("an unparseable numeric is forwarded so the server can name the field", () => {
  assert.equal(coerce(numeric, "sixty thousand"), "sixty thousand");
});

test("a categorical keeps its exact case", () => {
  // "rent" is not "RENT": the encoder pools an unknown category rather than
  // rejecting it, so a lower-cased value would score silently.
  assert.equal(coerce(categorical, "RENT"), "RENT");
  assert.equal(coerce(categorical, " CA "), "CA");
});

test("bars scale to the largest absolute contribution, not to their sum", () => {
  const bars = reasonBars([
    { feature: "a", log_odds: -0.4 },
    { feature: "b", log_odds: 0.2 },
    { feature: "c", log_odds: -0.1 },
  ]);

  assert.deepEqual(
    bars.map((bar) => Math.round(bar.width)),
    [100, 50, 25],
  );
  assert.deepEqual(
    bars.map((bar) => bar.increases),
    [false, true, false],
  );
});

test("a contribution too small to see still gets a visible mark", () => {
  const bars = reasonBars([{ log_odds: 1 }, { log_odds: 0.0001 }]);

  assert.equal(bars[0].width, 100);
  assert.ok(bars[1].width >= 2, "a real contribution must not render as nothing");
});

test("an all-zero set of reasons draws no bars rather than dividing by zero", () => {
  const bars = reasonBars([{ log_odds: 0 }, { log_odds: 0 }]);

  assert.deepEqual(
    bars.map((bar) => bar.width),
    [0, 0],
  );
});

test("a missing log_odds is treated as no contribution, not as NaN", () => {
  const bars = reasonBars([{ log_odds: 0.5 }, { log_odds: null }]);

  assert.equal(bars[1].width, 2);
  assert.equal(bars[1].increases, false);
});

test("the example applicant supplies every field the model requires", () => {
  // The required set comes from `/api/schema` at runtime; this asserts the four
  // that the origination-only tier marks required, so the demo button cannot
  // produce a form that fails its own validation.
  for (const name of ["issue_d", "loan_amnt", "term", "annual_inc"]) {
    assert.ok(name in EXAMPLE_APPLICANT, `${name} is required but absent from the example`);
  }
});

test("the example uses month-precision dates, which is a format the parser accepts", () => {
  // `<input type="month">` round-trips these; a "Jun-2015" here would not load
  // into the control and the field would silently render blank.
  assert.match(EXAMPLE_APPLICANT.issue_d, /^\d{4}-\d{2}$/);
  assert.match(EXAMPLE_APPLICANT.earliest_cr_line, /^\d{4}-\d{2}$/);
});

test("an absent field reads as words, not as the encoder's sentinel", () => {
  // Missingness is predictive, so an unfilled field is often the top-ranked
  // reason. Showing `__missing__` there presents a modelling artefact as if the
  // applicant had typed it.
  assert.equal(reasonValue("__missing__"), "not provided");
  assert.equal(reasonValue(null), "not provided");
  assert.equal(reasonValue(undefined), "not provided");
  assert.equal(reasonValue(""), "not provided");
});

test("a real value is shown as itself, including a falsy zero", () => {
  assert.equal(reasonValue("RENT"), "RENT");
  assert.equal(reasonValue(" 36 months"), " 36 months");
  // `0` is a measurement, not an absence: `inq_last_6mths` of 0 is meaningful and
  // a `||` fallback would have printed "not provided" for it.
  assert.equal(reasonValue(0), "0");
});

test("a raw ratio is cut to readable precision, and an integer keeps none", () => {
  // The engineered features are divisions, so this is what the API actually sends.
  assert.equal(reasonValue(0.24193548387096775), "0.2419");
  assert.equal(reasonValue(9), "9");
  assert.equal(reasonValue(0.625), "0.625");
  // A numeric string is still a number: the form posts strings and some fields
  // round-trip that way.
  assert.equal(reasonValue("0.24193548387096775"), "0.2419");
});
