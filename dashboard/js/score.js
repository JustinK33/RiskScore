/**
 * Score an applicant: the panel this whole project exists to make possible.
 *
 * The form is generated from `GET /api/schema`, never written by hand. That is the
 * point of the endpoint: the fields, which of them are required, their bounds, and
 * the categories the fitted encoder actually saw all come from the bundle being
 * served, so switching to a model trained on a different tier changes the form
 * rather than silently accepting inputs it ignores.
 *
 * Two decisions worth stating.
 *
 * Categoricals render as `<select>`, not text. `OneHotEncoder` is configured with
 * `handle_unknown="infrequent_if_exist"`, so `"rent"` for `"RENT"` is accepted,
 * pooled into the infrequent bucket, and scored - a plausible probability from a
 * typo, with no error anywhere. A closed choice list makes that unreachable from
 * this page.
 *
 * Reason codes render as a CSS tornado plot rather than a canvas. They are five
 * rows of a signed number with a long label, which is a table; drawing it would
 * mean rotated text, a fallback table beside it, and two things to keep in sync.
 * The bars here are `<div>`s whose width is a percentage, so they reflow and are
 * readable at 320px, and every number is present as text.
 */

import { getSchema, predict } from "./api.js";
import { el, replaceChildren, setBanner } from "./dom.js";
import { labelize, number, percent } from "./format.js";

/**
 * A plausible applicant, for the demo button.
 *
 * Hardcoded, and honest about it: the alternative is persisting the training
 * medians in the bundle, and a form prefilled with the *median* applicant is a
 * worse demo than one prefilled with a recognisable case.
 *
 * Values are in the *control's* dialect, not the extract's: `36` rather than
 * `" 36 months"`, `62.5` rather than `"62.5%"`, `"2015-06"` rather than
 * `"Jun-2015"`. `CanonicalizeFrame` accepts either, but an `<input type="month">`
 * silently renders blank for anything that is not `%Y-%m`, and a number input
 * rejects a percent sign - so a prefill in the raw dialect would half-load and
 * look like a bug in the form.
 *
 * `ponytail:` fixed example. If the demo ever needs to show several archetypes,
 * this becomes a list and the button becomes a select.
 */
export const EXAMPLE_APPLICANT = {
  issue_d: "2015-06",
  loan_amnt: 15000,
  term: 36,
  purpose: "debt_consolidation",
  annual_inc: 62000,
  emp_length: 5,
  home_ownership: "RENT",
  verification_status: "Verified",
  addr_state: "CA",
  dti: 18.2,
  revol_util: 62.5,
  revol_bal: 12000,
  delinq_2yrs: 0,
  inq_last_6mths: 1,
  open_acc: 9,
  total_acc: 21,
  pub_rec: 0,
  earliest_cr_line: "2003-08",
  // Present because the served bundle's schema offers them, even though both real
  // extracts in `data/raw/` lack them entirely. `fill` ignores names the schema does
  // not have, so a bundle trained on the real data simply leaves these two out.
  fico_range_low: 700,
  fico_range_high: 704,
};

/** Tier order for the fieldsets, so the form reads chronologically. */
const TIER_ORDER = ["timeline", "loan_request", "borrower", "lender_priced"];

/**
 * One raw form value to something `/predict` accepts.
 *
 * An empty string becomes `undefined`, not `null` and not `0`: the field was left
 * blank, which for an optional input means "not supplied", and the imputer handles
 * it. Sending `0` for a blank `delinq_2yrs` would be asserting a fact the user
 * never entered - and for `annual_inc` it would be asserting a false one.
 */
export function coerce(field, raw) {
  const text = typeof raw === "string" ? raw.trim() : raw;
  if (text === "" || text === null || text === undefined) return undefined;
  if (field.kind !== "numeric") return String(text);
  const value = Number(text);
  // A non-numeric string in a numeric field is passed through rather than dropped,
  // so the server's 422 names the field. Silently discarding it would score the
  // applicant as if the field were blank and report a decision built on less than
  // the user supplied.
  return Number.isFinite(value) ? value : String(text);
}

/** The control for one field, chosen by kind. */
function control(field) {
  const shared = {
    id: `field-${field.name}`,
    name: field.name,
    required: field.required,
    attrs: { "aria-describedby": `hint-${field.name}` },
  };
  if (field.kind === "categorical" && field.choices?.length) {
    return el("select", shared, [
      // A blank first option only where blank is legal, so a required select
      // cannot be submitted empty.
      field.required ? null : el("option", { value: "", textContent: "not supplied" }),
      ...field.choices.map((choice) => el("option", { value: choice, textContent: choice })),
    ]);
  }
  if (field.kind === "date") {
    // `type="month"` yields "2015-06", which is `%Y-%m` - one of the formats
    // `DEFAULT_DATE_FORMATS` accepts. Both date inputs here are month-precision in
    // the source data, so a day picker would invent precision.
    return el("input", { ...shared, type: "month" });
  }
  if (field.kind === "numeric") {
    return el("input", {
      ...shared,
      type: "number",
      // `any` rather than a per-field step: `dti` is fractional, `open_acc` is an
      // integer, and a wrong step makes the browser reject a valid value.
      step: "any",
      min: field.minimum ?? "",
      max: field.maximum ?? "",
    });
  }
  return el("input", { ...shared, type: "text" });
}

/**
 * The hint under one control.
 *
 * Categoricals get a sentence naming where the list came from, because a reader
 * who wants Wyoming and finds seventeen states needs to know the list is the
 * vocabulary this model was *fitted* on, not a validation rule someone chose.
 * `/predict` would accept `"WY"` - it would be pooled into the infrequent bucket
 * and scored - which is precisely why this page does not offer it.
 */
export function hint(field) {
  if (field.kind !== "categorical" || !field.choices?.length) return field.description;
  return `${field.description} ${field.choices.length} value${
    field.choices.length === 1 ? "" : "s"
  } appeared in training; anything else was never learned.`;
}

/**
 * Build the form body from the schema, grouped into a fieldset per tier.
 *
 * Returns `{node, read, fill}` - the caller never touches the inputs directly, so
 * the coercion rules live in exactly one place.
 */
export function buildForm(schema) {
  const byTier = new Map();
  for (const field of schema.fields) {
    if (!byTier.has(field.tier)) byTier.set(field.tier, []);
    byTier.get(field.tier).push(field);
  }
  const tiers = [...byTier.keys()].sort(
    (left, right) =>
      (TIER_ORDER.indexOf(left) + 1 || 99) - (TIER_ORDER.indexOf(right) + 1 || 99),
  );

  const inputs = new Map();
  const groups = tiers.map((tier) =>
    el("fieldset", { className: "field-group" }, [
      el("legend", { textContent: labelize(tier) }),
      ...byTier.get(tier).map((field) => {
        const input = control(field);
        inputs.set(field.name, { field, input });
        return el("div", { className: "field" }, [
          el("label", { attrs: { for: input.id } }, [
            field.name,
            field.required
              ? el("abbr", {
                  className: "required",
                  textContent: "*",
                  attrs: { title: "required" },
                })
              : null,
          ]),
          input,
          el("small", { id: `hint-${field.name}`, className: "hint", textContent: hint(field) }),
        ]);
      }),
    ]),
  );

  return {
    node: el("div", { className: "form-body" }, groups),
    /** Everything the user actually supplied, coerced. Blank fields are absent. */
    read() {
      const applicant = {};
      for (const [name, { field, input }] of inputs) {
        const value = coerce(field, input.value);
        if (value !== undefined) applicant[name] = value;
      }
      return applicant;
    },
    /** Prefill from a plain object, ignoring names this bundle does not accept. */
    fill(values) {
      for (const [name, { input }] of inputs) {
        input.value = values[name] === undefined ? "" : String(values[name]);
      }
    },
  };
}

/**
 * Reason codes as tornado-plot rows.
 *
 * Widths are relative to the largest absolute contribution in the set, not to the
 * total log-odds: the chart's job is to rank and compare the drivers, and scaling
 * to a total that the baseline dominates would render all five as slivers.
 */
export function reasonBars(reasons) {
  const largest = reasons.reduce(
    (most, reason) => Math.max(most, Math.abs(Number(reason.log_odds) || 0)),
    0,
  );
  return reasons.map((reason) => {
    const logOdds = Number(reason.log_odds) || 0;
    return {
      ...reason,
      increases: logOdds > 0,
      // A floor of 2%, so a contribution that is real but small is still a visible
      // mark rather than nothing at all.
      width: largest === 0 ? 0 : Math.max(2, (Math.abs(logOdds) / largest) * 100),
    };
  });
}

function reasonRow(reason) {
  const bar = el("div", {
    className: `bar ${reason.increases ? "bar-up" : "bar-down"}`,
    style: { width: `${reason.width}%` },
  });
  return el("tr", {}, [
    el("th", { attrs: { scope: "row" } }, [
      el("code", { textContent: reason.feature }),
      el("small", { className: "hint", textContent: reason.label }),
    ]),
    el("td", { className: "num", textContent: String(reason.value ?? "-") }),
    // Two half-width tracks with the bar growing outward from the centre, so
    // "reduces risk" reads left and "increases risk" reads right at a glance.
    //
    // The grid is a `<div>` inside the cell rather than the cell itself: a
    // `display: grid` on a `<td>` stops it being a table-cell, so the browser wraps
    // it in an anonymous one and the row's bottom borders end up drawn at two
    // different heights.
    el("td", { className: "tornado-cell" }, [
      el("div", { className: "tornado" }, [
        el("div", { className: "tornado-left" }, [reason.increases ? null : bar]),
        el("div", { className: "tornado-right" }, [reason.increases ? bar : null]),
      ]),
    ]),
    el("td", {
      className: "num",
      textContent: `${reason.increases ? "+" : ""}${number(reason.log_odds, 3)}`,
    }),
  ]);
}

/** The result block: decision, probability, and the reason-code table. */
export function renderPrediction(prediction) {
  const approved = prediction.decision === "approve";
  const reasons = reasonBars(prediction.reasons || []);

  return [
    el("div", { className: "verdict", dataset: { decision: prediction.decision } }, [
      el("div", { className: "verdict-decision" }, [
        el("span", { className: "chip-label", textContent: "Decision" }),
        el("strong", { textContent: approved ? "Approve" : "Decline" }),
        el("small", {
          textContent: `${percent(prediction.default_probability, 2)} default probability against a ${number(prediction.threshold, 2)} threshold`,
        }),
      ]),
      el("div", { className: "verdict-figure" }, [
        el("span", { className: "chip-label", textContent: "Probability" }),
        el("strong", { textContent: percent(prediction.default_probability, 2) }),
        el("small", {
          // Both log-odds, because the difference between them *is* the
          // explanation: the baseline is what the model says about nobody in
          // particular, and the reasons below sum to the gap.
          textContent: `log-odds ${number(prediction.baseline_log_odds, 2)} → ${number(prediction.total_log_odds, 2)}`,
        }),
      ]),
      el("div", { className: "verdict-figure" }, [
        el("span", { className: "chip-label", textContent: "Latency" }),
        el("strong", { textContent: `${number(prediction.latency_ms, 1)} ms` }),
        el("small", { textContent: "server-side, including the explanation" }),
      ]),
    ]),
    reasons.length === 0
      ? null
      : el("table", { className: "data-table reason-table" }, [
          el("caption", {
            textContent: `Top ${reasons.length} drivers, as exact log-odds contributions`,
          }),
          el("thead", {}, [
            el("tr", {}, [
              el("th", { attrs: { scope: "col" }, textContent: "Feature" }),
              el("th", { attrs: { scope: "col" }, className: "num", textContent: "Value" }),
              // Paired with the `.tornado` cells by class, because both are hidden
              // together below 640px where the plot cannot fit - see app.css. An
              // `nth-child` would silently target the wrong column the moment a
              // column is added.
              el("th", {
                attrs: { scope: "col" },
                className: "tornado-head",
                textContent: "Reduces ← → increases risk",
              }),
              el("th", { attrs: { scope: "col" }, className: "num", textContent: "Log-odds" }),
            ]),
          ]),
          el("tbody", {}, reasons.map(reasonRow)),
        ]),
  ];
}

/**
 * Wire the panel. Fetches the schema once, then scores on submit.
 *
 * Everything user-visible on failure goes through `setBanner` on the panel's own
 * banner rather than the page-level one, because a rejected applicant is a local
 * problem and should not look like the dashboard breaking.
 */
export async function mountScorePanel({ form, body, banner, result, exampleButton, submitButton }) {
  if (!form || !body) return;
  let schema;
  try {
    schema = await getSchema();
  } catch (error) {
    setBanner(banner, `The input schema is unavailable: ${error.message}`, "danger");
    return;
  }

  const built = buildForm(schema);
  replaceChildren(body, built.node);
  exampleButton?.addEventListener("click", () => {
    built.fill(EXAMPLE_APPLICANT);
    setBanner(banner, null);
  });

  form.addEventListener("submit", async (event) => {
    // The form has no action and no method: this is the only submit path, and
    // letting the browser navigate would lose the page. `novalidate` is *not* set,
    // so required fields and numeric bounds are enforced before we get here.
    event.preventDefault();
    if (submitButton) submitButton.disabled = true;
    setBanner(banner, "Scoring", "info");
    try {
      const prediction = await predict(built.read());
      replaceChildren(result, renderPrediction(prediction));
      setBanner(banner, null);
    } catch (error) {
      // A 422 names the offending field in `detail`, which is the single most
      // useful string on the page at that moment, so it is shown verbatim.
      setBanner(banner, error.message, "danger");
      replaceChildren(result, []);
    } finally {
      if (submitButton) submitButton.disabled = false;
    }
  });
}
