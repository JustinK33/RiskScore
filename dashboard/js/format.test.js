/**
 * `node --test dashboard/js/` - no framework, no jsdom, no dependencies.
 *
 * These run in CI beside the python suite. They are worth having because the
 * absent-is-not-zero rule is the kind of thing that regresses the moment
 * somebody adds a formatter and copies the guard from the wrong neighbour, and
 * because rounding is easy to get wrong in a way that looks right.
 *
 * Separator note: `format.js` formats with the *reader's* locale, so a literal
 * `"0.683"` would fail on a machine whose decimal separator is a comma. The
 * tests derive the two separators from `Intl` and assert against those, which
 * still pins every digit, the digit count, and the rounding - the actual logic.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  MISSING,
  PSI_MODERATE,
  PSI_SIGNIFICANT,
  cost,
  count,
  isMissing,
  labelize,
  number,
  numericColumn,
  percent,
  psiBand,
  shortRunId,
  signed,
  timestamp,
  toRows,
} from "./format.js";

/** The locale's decimal separator, e.g. "." or ",". */
const DECIMAL = new Intl.NumberFormat(undefined, { minimumFractionDigits: 1 })
  .format(1.5)
  .charAt(1);

/** The locale's grouping separator, e.g. "," or "." or a narrow no-break space. */
const GROUP = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 })
  .format(1000)
  .charAt(1);

/** "0.683" written in whatever separators this machine uses. */
function decimal(whole, fraction) {
  return `${whole}${DECIMAL}${fraction}`;
}

// --- 1. absent is not zero ---

test("every kind of absent value formats as a dash, never as zero", () => {
  // `""` is the one that matters most: `Number("")` is 0, so an empty CSV cell
  // would otherwise render as a measured 0.000.
  for (const absent of [null, undefined, "", "n/a", NaN, Infinity, -Infinity, {}, [], true]) {
    assert.equal(number(absent), MISSING, `number(${JSON.stringify(absent)})`);
    assert.equal(percent(absent), MISSING, `percent(${JSON.stringify(absent)})`);
    assert.equal(count(absent), MISSING, `count(${JSON.stringify(absent)})`);
    assert.equal(signed(absent), MISSING, `signed(${JSON.stringify(absent)})`);
    assert.equal(cost(absent), MISSING, `cost(${JSON.stringify(absent)})`);
    assert.equal(isMissing(absent), true, `isMissing(${JSON.stringify(absent)})`);
  }
});

test("a real zero still formats as zero", () => {
  // The other half of the rule. A default rate of exactly 0 in a vintage with no
  // defaults is a fact, and it must not render as "-".
  assert.equal(number(0), decimal(0, "000"));
  assert.equal(percent(0), `${decimal(0, 0)}%`);
  assert.equal(count(0), "0");
  assert.equal(isMissing(0), false);
});

test("a numeric string is accepted, because CSV-derived payloads send them", () => {
  assert.equal(number("0.6829908"), decimal(0, 683));
  assert.equal(count("12000"), `12${GROUP}000`);
});

// --- 2. rounding and digits ---

test("digits are pinned rather than trimmed", () => {
  // The trailing zeros are the point: a metric card whose value is sometimes
  // "0.7" and sometimes "0.683" changes width as the data changes.
  assert.equal(number(0.7), decimal(0, 700));
  assert.equal(number(0.7, 2), decimal(0, 70));
  assert.equal(number(1), decimal(1, "000"));
});

test("percent scales by 100 and keeps one decimal by default", () => {
  assert.equal(percent(0.13345), `${decimal(13, 3)}%`);
  assert.equal(percent(0.13345, 2), `${decimal(13, 35)}%`);
  assert.equal(percent(1), `${decimal(100, 0)}%`);
});

test("counts and costs are grouped and never show decimals", () => {
  assert.equal(count(1619), `1${GROUP}619`);
  assert.equal(cost(1234.7), `1${GROUP}235`);
});

// --- 3. signs ---

test("a positive contribution carries an explicit plus and a negative a minus", () => {
  // The sign is the reason code: "+0.412 increases risk" and "-0.412 reduces
  // risk" are opposite sentences in an adverse action notice.
  assert.equal(signed(0.412), `+${decimal(0, 412)}`);
  assert.equal(signed(-0.1177), `-${decimal(0, 118)}`);
});

test("an exact zero contribution is not signed", () => {
  assert.equal(signed(0), decimal(0, "000"));
  assert.equal(signed(-0), decimal(0, "000"));
});

// --- 4. timestamps ---

test("an instant renders in UTC to the minute, matching the run id", () => {
  // The run id for this instant begins 20260824T215753249Z. Rendering local time
  // here would make the header and the id look like different runs.
  assert.equal(timestamp("2026-08-24T21:57:53.249Z"), "2026-08-24 21:57 UTC");
});

test("an unparseable instant is a dash rather than Invalid Date", () => {
  assert.equal(timestamp("not a date"), MISSING);
  assert.equal(timestamp(null), MISSING);
  assert.equal(timestamp(""), MISSING);
});

// --- 5. run ids ---

test("a run id shortens to its instant and its commit", () => {
  assert.equal(
    shortRunId("20260824T215753249Z-logistic_regression-origination_only-53f540e"),
    "20260824·2157 53f540e",
  );
});

test("anything that is not a run id is returned untouched", () => {
  // Slicing an unknown string at fixed offsets is how a label starts lying.
  assert.equal(shortRunId("dev"), "dev");
  assert.equal(shortRunId(""), MISSING);
  assert.equal(shortRunId(undefined), MISSING);
});

// --- 6. PSI bands ---

test("psi bands are inclusive at their lower bound", () => {
  // Inclusive matches `risk_score.drift`, which uses >=. A value exactly on 0.25
  // reported as "moderate" here and "significant" in the CSV is the kind of
  // disagreement that costs an hour.
  assert.equal(psiBand(0.05).band, "stable");
  assert.equal(psiBand(PSI_MODERATE).band, "moderate");
  assert.equal(psiBand(0.24).band, "moderate");
  assert.equal(psiBand(PSI_SIGNIFICANT).band, "significant");
  assert.equal(psiBand(3).band, "significant");
});

test("a psi band names a token rather than a colour, so dark mode follows", () => {
  assert.equal(psiBand(0.3).tone, "danger");
  assert.equal(psiBand(0.15).tone, "warn");
  assert.equal(psiBand(0).tone, "ok");
  assert.equal(psiBand(null).tone, "muted");
});

// --- 7. label fallback ---

test("a snake_case name titles, with initialisms left upper", () => {
  assert.equal(labelize("auc_roc"), "AUC ROC");
  assert.equal(labelize("expected_calibration_error"), "Expected Calibration Error");
  assert.equal(labelize("dti"), "DTI");
});

// --- 8. columnar payloads ---

test("columns become rows", () => {
  assert.deepEqual(toRows({ bin: [1, 2], rows: [10, 20] }), [
    { bin: 1, rows: 10 },
    { bin: 2, rows: 20 },
  ]);
});

test("columns of unequal length truncate to the shortest", () => {
  // A row with a hole in it renders as a real value beside a dash, which reads
  // as a measurement that came back empty rather than as a payload bug.
  assert.deepEqual(toRows({ a: [1, 2, 3], b: [10] }), [{ a: 1, b: 10 }]);
});

test("a non-columnar payload yields no rows instead of throwing", () => {
  // The dashboard must render a message, not a blank page, when an endpoint
  // returns something unexpected.
  assert.deepEqual(toRows(null), []);
  assert.deepEqual(toRows({}), []);
  assert.deepEqual(toRows({ note: "not an array" }), []);
});

test("non-finite entries in a column are counted, not silently dropped", () => {
  // A chart of 8 of 10 bins that says nothing looks like a chart of 8 bins.
  assert.deepEqual(numericColumn([1, null, "2", NaN, ""]), { values: [1, 2], dropped: 3 });
  assert.deepEqual(numericColumn(undefined), { values: [], dropped: 0 });
});
