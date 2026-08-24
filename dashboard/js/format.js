/**
 * Value -> string. No DOM, no fetch, no state: every function here is pure, which
 * is what makes `format.test.js` runnable under plain `node --test` with no
 * browser and no jsdom.
 *
 * The one rule the whole module exists to enforce: **absent is not zero.**
 *
 * The dashboard this replaced ran every number through `Number(value).toFixed(3)`
 * with a NaN guard, so `null` became `"-"` but `undefined` inside a renamed field
 * became `"NaN"`, and - worse - a metric the API had not computed rendered as
 * `0.000`. A calibration error of `0.000` is a claim of perfect calibration. It
 * has to be impossible to state that by accident, so `MISSING` is returned for
 * every input that is not a finite number, and there is a test per input kind.
 */

/** What every formatter returns when it has nothing to say. */
export const MISSING = "-";

/**
 * Number formatters are constructed once. `new Intl.NumberFormat(...)` costs
 * roughly a hundred microseconds and is called per cell per draw, which is
 * measurable on a table of a few hundred PSI rows.
 */
const cache = new Map();

function formatter(options) {
  const key = JSON.stringify(options);
  let found = cache.get(key);
  if (found === undefined) {
    // `undefined` locale, not `"en-US"`: the reader's own grouping and decimal
    // separators. Digits are pinned by the options, so this changes separators
    // and never precision.
    found = new Intl.NumberFormat(undefined, options);
    cache.set(key, found);
  }
  return found;
}

/**
 * True for anything that must render as `MISSING`.
 *
 * `""` is included because a CSV-derived payload sends an empty cell as an empty
 * string, and `Number("")` is `0` - the exact coercion that turns "not measured"
 * into "measured as zero".
 *
 * Only `number` and `string` get as far as the finiteness check, and that
 * type gate is load-bearing rather than defensive: `Number([])` is `0` and
 * `Number(true)` is `1`, so an endpoint that sent an empty array where a metric
 * belonged would otherwise render a confident `0.000`.
 */
export function isMissing(value) {
  if (value === null || value === undefined || value === "") return true;
  if (typeof value !== "number" && typeof value !== "string") return true;
  return !Number.isFinite(Number(value));
}

/** A metric, fixed decimals. `number(0.6829908, 3)` -> `"0.683"`. */
export function number(value, digits = 3) {
  if (isMissing(value)) return MISSING;
  return formatter({
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value));
}

/**
 * A rate in [0, 1] as a percentage. `percent(0.13345)` -> `"13.3%"`.
 *
 * Multiplying before formatting rather than using `style: "percent"`: the two
 * agree, but `style: "percent"` inserts a locale-specific space before the sign
 * in some locales, and these strings go into fixed-width metric cards.
 */
export function percent(value, digits = 1) {
  if (isMissing(value)) return MISSING;
  return `${number(Number(value) * 100, digits)}%`;
}

/** A row count, grouped. `count(12000)` -> `"12,000"`. */
export function count(value) {
  if (isMissing(value)) return MISSING;
  return formatter({ maximumFractionDigits: 0 }).format(Number(value));
}

/**
 * A signed contribution, for reason codes and drift deltas.
 *
 * The `+` is explicit because the sign *is* the information in a reason code:
 * "increases risk" and "reduces risk" are different sentences in an adverse
 * action notice. `signWhenZero` stays off so an exact zero renders as `0.000`
 * rather than the nonsense `+0.000`.
 */
export function signed(value, digits = 3) {
  if (isMissing(value)) return MISSING;
  const asNumber = Number(value);
  const body = number(Math.abs(asNumber), digits);
  if (asNumber > 0) return `+${body}`;
  if (asNumber < 0) return `-${body}`;
  return body;
}

/**
 * A cost from the threshold table. Grouped, no decimals.
 *
 * Unitless on purpose: the cost matrix is a ratio (a false negative costs 5, a
 * false positive 1), so rendering it as currency would invent a unit the model
 * never had.
 */
export function cost(value) {
  if (isMissing(value)) return MISSING;
  return formatter({ maximumFractionDigits: 0 }).format(Number(value));
}

/**
 * An ISO instant as a readable UTC string. `"2026-08-24T21:57:53.249Z"` ->
 * `"2026-08-24 21:57 UTC"`.
 *
 * **UTC, not local time,** which is the unusual choice and the deliberate one.
 * A run id begins `20260824T215753249Z`, and a header that showed
 * `2026-08-24 14:57` beside a run id reading `T2157` looks like two different
 * runs. The identifier is in UTC, so its rendering is too, and the suffix says
 * so rather than leaving the reader to guess.
 */
export function timestamp(value) {
  if (value === null || value === undefined || value === "") return MISSING;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return MISSING;
  const iso = parsed.toISOString();
  return `${iso.slice(0, 10)} ${iso.slice(11, 16)} UTC`;
}

/**
 * A run id shortened for a control that cannot hold 60 characters.
 *
 * Keeps the timestamp to the minute and the git sha and drops the middle, since
 * model type and tier are displayed as their own fields beside it. Anything
 * that does not look like a run id is returned untouched rather than sliced into
 * something misleading.
 */
export function shortRunId(runId) {
  if (typeof runId !== "string" || runId === "") return MISSING;
  const parts = runId.split("-");
  if (parts.length < 2 || parts[0].length < 16) return runId;
  const stamp = parts[0];
  return `${stamp.slice(0, 8)}·${stamp.slice(9, 13)} ${parts[parts.length - 1]}`;
}

/**
 * PSI bands, from `risk_score.drift`. The thresholds are the conventional ones
 * (0.10 and 0.25) and they are duplicated here rather than fetched, because the
 * API sends the band name per row already - this is only for the legend and for
 * colouring a value the caller computed itself.
 */
export const PSI_MODERATE = 0.1;
export const PSI_SIGNIFICANT = 0.25;

/**
 * A PSI value -> a band name and the token suffix to colour it with.
 *
 * Returned as a token *name* rather than a colour, so dark mode still resolves
 * through the cascade. A caller writes `var(--${tone})`.
 */
export function psiBand(value) {
  if (isMissing(value)) return { band: MISSING, tone: "muted" };
  const psi = Number(value);
  if (psi >= PSI_SIGNIFICANT) return { band: "significant", tone: "danger" };
  if (psi >= PSI_MODERATE) return { band: "moderate", tone: "warn" };
  return { band: "stable", tone: "ok" };
}

/**
 * A snake_case identifier as a title, for a column the API sent no label for.
 *
 * Only ever a fallback: the real labels come from the feature registry through
 * `/api/schema` and `/api/shap-summary`, because a label derived from a column
 * name cannot say that `dti` means "debt-to-income ratio". Known initialisms are
 * upper-cased so the fallback does not read as `Auc Roc`.
 */
const INITIALISMS = new Set(["auc", "roc", "ks", "psi", "dti", "fico", "shap", "id", "csv"]);

export function labelize(name) {
  if (typeof name !== "string" || name === "") return MISSING;
  return name
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((word) =>
      INITIALISMS.has(word.toLowerCase())
        ? word.toUpperCase()
        : word.charAt(0).toUpperCase() + word.slice(1),
    )
    .join(" ");
}

/**
 * A columnar payload -> an array of row objects.
 *
 * The report endpoints send columns (`{bin: [...], rows: [...]}`) rather than
 * rows, because 99 threshold entries as row objects repeat every key 99 times -
 * about four times the bytes for the same numbers. Charts want columns and
 * tables want rows, so the conversion lives here, once, next to a test that
 * pins what happens when two columns disagree about their length: the shortest
 * wins, because a table with a hole in it is worse than a shorter table.
 */
export function toRows(columns) {
  if (columns === null || typeof columns !== "object") return [];
  const names = Object.keys(columns).filter((name) => Array.isArray(columns[name]));
  if (names.length === 0) return [];
  const length = Math.min(...names.map((name) => columns[name].length));
  return Array.from({ length }, (_, index) => {
    const row = {};
    for (const name of names) row[name] = columns[name][index];
    return row;
  });
}

/**
 * One column as an array of finite numbers, with the positions that were not
 * finite reported rather than dropped silently.
 *
 * A chart that plots 8 of 10 calibration bins and says nothing looks like a
 * chart of 8 bins. The caller decides what to do with `dropped`; nothing here
 * decides for it.
 */
export function numericColumn(values) {
  if (!Array.isArray(values)) return { values: [], dropped: 0 };
  const kept = [];
  let dropped = 0;
  for (const value of values) {
    if (isMissing(value)) dropped += 1;
    else kept.push(Number(value));
  }
  return { values: kept, dropped };
}
