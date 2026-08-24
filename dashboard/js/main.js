/**
 * Boot, state, and wiring. The only module that touches both the network and the
 * DOM, and it does neither directly: `api.js` fetches, `dom.js` builds,
 * `panels.js` draws.
 *
 * The single most important property of this file is that **`redraw` never
 * fetches.** Everything a chart needs is in `state`, so a resize, a theme change,
 * and a font finally loading all cost one draw and zero requests. The dashboard
 * this replaces re-requested two endpoints on every `resize` event and discarded
 * the outcome with `.catch(() => {})`, so dragging a window edge was a request
 * storm whose failures were invisible and whose responses could arrive out of
 * order and draw stale data over fresh.
 *
 * The second property is that a partly-available run still renders. Reports are
 * fetched with `allSettled`, so a run trained before a report existed shows every
 * panel it can and names the ones it cannot, rather than failing the page. The old
 * version treated any non-OK response from any of three endpoints as a single
 * fatal "Report artifacts are missing".
 */

import {
  ApiError,
  getCalibration,
  getHealth,
  getMetrics,
  getModel,
  getRun,
  getRuns,
  getThresholdCosts,
  getVintages,
  invalidate,
} from "./api.js";
import { observeResize } from "./charts.js";
import {
  $,
  dataTableDetails,
  definition,
  el,
  replaceChildren,
  setBanner,
  setStatus,
  setText,
} from "./dom.js";
import { count, number, percent, shortRunId, timestamp } from "./format.js";
import { drawCalibration, drawEmbargo, drawThresholdCosts, drawVintages } from "./panels.js";
import { mountScorePanel } from "./score.js";

/**
 * Everything fetched, keyed by nothing but its own name.
 *
 * A plain object rather than a store: there is one page, one run selected at a
 * time, and no component that owns a slice of it. `runId` is the identity of
 * everything else in here - see `load`, which replaces the whole object rather
 * than mutating fields, so a half-applied run cannot be rendered.
 */
let state = {
  runId: null,
  activeRunId: null,
  model: null,
  manifest: null,
  metrics: null,
  calibration: null,
  thresholdCosts: null,
  vintages: null,
  runs: [],
  problems: [],
};

/** Below these, a test-set metric is noise being reported to three decimals. */
const MIN_TEST_ROWS = 1000;
const MIN_TEST_POSITIVES = 50;

const THEME_STORAGE = "riskscore.theme";

/** The run directory files worth offering as links, in reading order. */
const ARTIFACTS = [
  ["model_card.md", "Model card"],
  ["manifest.json", "Manifest"],
  ["metrics.json", "Metrics"],
  ["calibration_test.csv", "Calibration (test)"],
  ["calibration_validation.csv", "Calibration (validation)"],
  ["threshold_costs_validation.csv", "Threshold costs"],
  ["metrics_by_vintage.csv", "Metrics by vintage"],
  ["psi_features.csv", "Feature PSI"],
  ["psi_score.csv", "Score PSI"],
  ["shap_summary.csv", "SHAP summary"],
  ["figures/calibration_test.png", "Calibration figure"],
  ["run.log", "Run log"],
];

// --- loading -------------------------------------------------------------------

/**
 * Fetch everything for one run and render it.
 *
 * `runId` of `null` means the active run, which is also what the report endpoints
 * mean by an absent `run_id`, so the null is passed straight through rather than
 * resolved first - one fewer round trip before the first paint.
 */
async function load(runId = null) {
  setStatus($("#status"), "Loading reports", "loading");

  // Readiness first and on its own, because when the answer is "no bundle" every
  // report below it is a 503 and reporting six of those is worse than reporting
  // the one cause.
  const health = await getHealth();
  if (!health.bundle_loaded) {
    setStatus($("#status"), "No model loaded", "error");
    setBanner(
      $("#healthBanner"),
      `${health.reason || "The service has no active run."} Train one with \`riskscore train\`.`,
      "danger",
    );
    return;
  }
  setBanner($("#healthBanner"), null);

  // `allSettled`, not `all`: a run predating a given report should show every
  // panel it can. The rejected ones are collected and named in the banner.
  const [model, runs, manifest, metrics, calibration, thresholdCosts, vintages] =
    await Promise.allSettled([
      getModel(),
      getRuns(),
      runId ? getRun(runId) : getModel().then((identity) => getRun(identity.run_id)),
      getMetrics(runId),
      getCalibration(runId),
      getThresholdCosts(runId),
      getVintages(runId),
    ]);

  // Unwrapped exactly once each, because `value` records a failure as a side
  // effect and calling it twice on the same rejection would report it twice.
  const problems = [];
  const value = (settled, name) => {
    if (settled.status === "fulfilled") return settled.value;
    const error = settled.reason;
    problems.push(`${name}: ${error instanceof ApiError ? error.message : String(error)}`);
    return null;
  };
  const identity = value(model, "model");
  const history = value(runs, "run history");

  state = {
    runId: runId || identity?.run_id || null,
    activeRunId: history?.active_run_id || null,
    model: identity,
    manifest: value(manifest, "manifest"),
    metrics: value(metrics, "metrics"),
    calibration: value(calibration, "calibration"),
    thresholdCosts: value(thresholdCosts, "threshold costs"),
    vintages: value(vintages, "vintages"),
    runs: history?.runs || [],
    problems,
  };

  render();
}

/** Everything that depends on `state`. Called once per load and per theme change. */
function render() {
  renderRunPicker();
  renderIdentity();
  renderRowFlow();
  renderMetrics();
  renderRunDetails();
  renderEmbargoFacts();
  renderArtifacts();
  renderSanity();
  redraw();

  if (state.problems.length) {
    setBanner(
      $("#healthBanner"),
      `This run is missing ${state.problems.length} report${state.problems.length > 1 ? "s" : ""}. ${state.problems.join("; ")}`,
      "warn",
    );
    setStatus($("#status"), "Loaded with gaps", "warn");
  } else {
    setStatus($("#status"), `Loaded ${shortRunId(state.runId)}`, "ok");
  }
}

/**
 * Draw both charts from `state`. **No fetching, ever.**
 *
 * Each renderer returns its own accessible label and data table, which are
 * installed here rather than inside the renderer, so a panel cannot forget: the
 * label and the fallback are produced by the same call that draws the pixels.
 */
function redraw() {
  paint("#calibrationChart", "#calibrationCaption", "#calibrationFallback", () =>
    drawCalibration($("#calibrationChart"), state.calibration),
  );
  paint("#thresholdChart", "#thresholdCaption", "#thresholdFallback", () =>
    drawThresholdCosts($("#thresholdChart"), state.thresholdCosts, {
      selectedThreshold: state.metrics?.metrics?.selected_threshold,
    }),
  );
  paint("#embargoChart", "#embargoCaption", "#embargoFallback", () =>
    drawEmbargo($("#embargoChart"), state.manifest),
  );
  paint("#vintageChart", "#vintageCaption", "#vintageFallback", () =>
    drawVintages($("#vintageChart"), state.vintages),
  );
}

function paint(canvasSelector, captionSelector, fallbackSelector, draw) {
  const canvas = $(canvasSelector);
  if (!canvas) return;
  const { label, table } = draw();
  canvas.setAttribute("aria-label", label);
  setText(captionSelector, label);
  replaceChildren($(fallbackSelector), table ? dataTableDetails("Show the numbers", table) : []);
}

// --- panels --------------------------------------------------------------------

function renderRunPicker() {
  const select = $("#runSelect");
  if (!select) return;
  replaceChildren(
    select,
    state.runs.map((run) =>
      el("option", {
        value: run.run_id,
        selected: run.run_id === state.runId,
        // The active run is marked, because "which one is serving /predict" is a
        // different question from "which one am I looking at".
        textContent: `${shortRunId(run.run_id)} · ${run.model_type}${
          run.run_id === state.activeRunId ? " · active" : ""
        }`,
      }),
    ),
  );
  select.disabled = state.runs.length < 2;
}

function renderIdentity() {
  const model = state.model || {};
  const manifest = state.manifest || {};
  replaceChildren($("#identityGrid"), [
    definition("Run", state.runId, { mono: true }),
    definition("Trained", timestamp(model.created_at || manifest.created_at)),
    definition("Model", model.model_type || manifest.model_type),
    definition("Feature tier", model.feature_tier || manifest.feature_tier),
    definition("Commit", model.git_commit || manifest.git_commit, { mono: true }),
    definition("Dataset", manifest.dataset_sha256, { mono: true }),
  ]);
}

/**
 * The row funnel: how many loans each filter removed, in order.
 *
 * This is the panel that makes the project's central correction visible. 12,000
 * rows become 9,326 closed loans, and the embargo takes another 2,037 - the
 * immature ones whose outcome is not yet known. Without that step the measured
 * default rate climbs with every vintage, which reads as a deteriorating book and
 * is actually survivorship bias. The embargo's own before/after figures are in
 * the manifest and get their own panel; this strip is where a reader sees that
 * something was removed at all.
 */
function renderRowFlow() {
  const rows = state.manifest?.rows;
  const node = $("#rowFlow");
  if (!node) return;
  if (!rows) {
    replaceChildren(node, []);
    return;
  }
  const stages = [
    ["Raw", rows.raw, "rows read from the extract"],
    ["Closed", rows.closed, "loans with a final status"],
    ["Mature", rows.mature, "outcome known by the snapshot date"],
    ["In-scope term", rows.in_scope_terms, "36-month loans"],
  ];
  const chips = stages.map(([label, value, title], index) => {
    const previous = index === 0 ? null : stages[index - 1][1];
    const delta = previous === null ? null : value - previous;
    return el("div", { className: "chip", attrs: { title } }, [
      el("span", { className: "chip-label", textContent: label }),
      el("strong", { textContent: count(value) }),
      // The sign carries the direction on its own, so there is no colour here.
      // Every stage in this funnel removes rows; four orange numbers in a row
      // would read as four warnings.
      delta === null
        ? null
        : el("small", { textContent: `${delta > 0 ? "+" : "-"}${count(Math.abs(delta))}` }),
    ]);
  });
  const splits = ["train", "validation", "test"].map((name) =>
    el("div", { className: "chip chip-split" }, [
      el("span", { className: "chip-label", textContent: name }),
      el("strong", { textContent: count(rows[name]) }),
      el("small", { textContent: state.manifest?.split_windows?.[name] || "" }),
    ]),
  );
  replaceChildren(node, [
    ...chips,
    // Decorative: the reading order already conveys the sequence, and a screen
    // reader announcing "right arrow" between two counts adds nothing.
    el("div", { className: "chip-arrow", textContent: "→", attrs: { "aria-hidden": "true" } }),
    ...splits,
  ]);
}

function renderMetrics() {
  const metrics = state.metrics?.metrics || {};
  setText("#aucRoc", number(metrics.auc_roc));
  setText("#averagePrecision", number(metrics.average_precision));
  setText("#ksStatistic", number(metrics.ks_statistic));
  setText("#brierScore", number(metrics.brier_score));
  setText("#ece", number(metrics.expected_calibration_error));
  setText("#defaultRate", percent(metrics.default_rate));
  setText("#approvalRate", percent(metrics.approval_rate));
  setText("#selectedThreshold", number(metrics.selected_threshold, 2));

  // The notes carry the comparisons that make each headline number readable, and
  // they are set from data rather than written in the markup because each one is a
  // claim about this run.
  setText(
    "#brierScoreNote",
    `Squared error of the probability. Uncalibrated: ${number(metrics.brier_score_uncalibrated)}`,
  );
  setText(
    "#eceNote",
    `Mean gap, predicted against observed. On validation, in-sample: ${number(
      metrics.expected_calibration_error_validation_in_sample,
    )}`,
  );
  setText(
    "#selectedThresholdNote",
    `Chosen on validation. Total cost there: ${count(metrics.selected_threshold_total_cost)}`,
  );
}

function renderRunDetails() {
  const manifest = state.manifest || {};
  const metrics = state.metrics?.metrics || {};
  const costs = manifest.cost_matrix || {};
  replaceChildren($("#runDetails"), [
    definition("Missed default costs", number(costs.false_negative_cost, 1)),
    definition("Declined good loan costs", number(costs.false_positive_cost, 1)),
    definition("Calibration", metrics.calibration_method),
    definition("Calibration fitted on", metrics.calibration_fitted_on),
    definition("Embargo", manifest.embargo?.snapshot),
    definition("Bundle schema", manifest.bundle_schema_version ?? state.model?.bundle_schema_version),
  ]);
}

/**
 * The embargo rule itself, beside the chart that shows what it bought.
 *
 * `rows_unknown_maturity` is here even when it is zero, and especially then: it
 * counts loans whose term or issue date could not be read, which the embargo has to
 * drop because it cannot prove they matured. A silent zero is the difference
 * between "no rows were unreadable" and "nobody checked".
 */
function renderEmbargoFacts() {
  const embargo = state.manifest?.embargo;
  const node = $("#embargoFacts");
  if (!node) return;
  if (!embargo) {
    replaceChildren(node, []);
    setText("#embargoSummary", "This run predates the embargo record.");
    return;
  }
  replaceChildren(node, [
    definition("Snapshot", embargo.snapshot),
    definition("Rule", "issue_d + term <= snapshot", { mono: true }),
    definition("Immature, removed", count(embargo.rows_immature)),
    definition("Maturity unknown, removed", count(embargo.rows_unknown_maturity)),
  ]);
  // The manifest's own one-line summary, verbatim. It is what `run.log` records and
  // what the model card quotes, so showing anything reworded here would give a
  // reader two versions of one fact to reconcile.
  setText("#embargoSummary", embargo.summary || "");
}

function renderArtifacts() {
  const node = $("#artifactLinks");
  if (!node) return;
  if (!state.runId) {
    replaceChildren(node, []);
    return;
  }
  replaceChildren(
    node,
    ARTIFACTS.map(([name, label]) =>
      el("a", {
        // `/artifacts/{run_id}/{name}` - the run-scoped path. The old links pointed
        // at `/artifacts/metrics/logistic_regression_metrics.json`, a layout that
        // stopped existing when runs became directories, so every one was a 404.
        href: `/artifacts/${encodeURIComponent(state.runId)}/${name}`,
        textContent: label,
        // Not `target="_blank"`: a CSV that the browser downloads and a PNG that it
        // renders behave differently already, and opening either in a new tab is
        // the reader's decision to make with a modifier key.
      }),
    ),
  );
}

/**
 * The sanity banner.
 *
 * The artifact committed to this repository before the rewrite was a test set of
 * 116 rows containing one default, presented with an AUC to four decimals and no
 * caveat at all. A metric computed on 116 rows is not wrong, it is noise, and the
 * dashboard is the last place that can say so before a reader quotes it.
 */
function renderSanity() {
  const node = $("#sanityBanner");
  const rows = state.manifest?.rows;
  const rate = state.metrics?.metrics?.default_rate;
  if (!node || !rows?.test) {
    setBanner(node, null);
    return;
  }
  const positives = Number.isFinite(Number(rate)) ? Math.round(rows.test * Number(rate)) : null;
  const warnings = [];
  if (rows.test < MIN_TEST_ROWS) {
    warnings.push(`the test partition holds ${count(rows.test)} loans (under ${count(MIN_TEST_ROWS)})`);
  }
  if (positives !== null && positives < MIN_TEST_POSITIVES) {
    warnings.push(`only about ${count(positives)} of them defaulted (under ${MIN_TEST_POSITIVES})`);
  }
  setBanner(
    node,
    warnings.length
      ? `Treat these metrics as indicative: ${warnings.join(", and ")}. Confidence intervals on an AUC from a sample this size are wide enough to include chance.`
      : null,
    "warn",
  );
}

// --- theme ---------------------------------------------------------------------

/** The theme actually in effect, whether chosen or inherited from the OS. */
function effectiveTheme() {
  const chosen = document.documentElement.dataset.theme;
  if (chosen === "dark" || chosen === "light") return chosen;
  return globalThis.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/**
 * Reflect the theme in the toggle. Reads state, never writes it.
 *
 * Kept separate from `applyTheme` because boot must not persist anything: an
 * `applyTheme(effectiveTheme())` at startup writes the OS preference into
 * `localStorage`, which pins the theme to whatever the machine happened to be on
 * the first visit and stops the page following the OS ever again. The button
 * label names the *action*, not the current state, which is why it says "Dark
 * theme" while the page is light.
 */
function syncThemeButton(theme) {
  $("#themeToggle")?.setAttribute("aria-pressed", String(theme === "dark"));
  setText("#themeToggleLabel", theme === "dark" ? "Light theme" : "Dark theme");
}

/** Choose a theme explicitly, from a click. This one does persist. */
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_STORAGE, theme);
  } catch {
    /* A blocked storage is not a reason to refuse the toggle for this page view. */
  }
  syncThemeButton(theme);
  // Canvas has no cascade: the colours were baked into pixels at draw time, so a
  // theme change is only complete once the charts are redrawn.
  redraw();
}

function wireTheme() {
  syncThemeButton(effectiveTheme());
  $("#themeToggle")?.addEventListener("click", () => {
    applyTheme(effectiveTheme() === "dark" ? "light" : "dark");
  });
  // Follow the OS while no explicit choice has been made. Without this, a machine
  // switching to dark at sunset leaves the charts drawn in the light palette on a
  // dark page - the CSS follows, the canvas cannot.
  globalThis.matchMedia?.("(prefers-color-scheme: dark)").addEventListener("change", (event) => {
    if (document.documentElement.dataset.theme) return;
    syncThemeButton(event.matches ? "dark" : "light");
    redraw();
  });
}

// --- boot ----------------------------------------------------------------------

/** The run named in the URL, so a link to a past run's reports is shareable. */
function runIdFromUrl() {
  const value = new URLSearchParams(globalThis.location.search).get("run_id");
  return value && value.trim() !== "" ? value : null;
}

function wireRunPicker() {
  $("#runSelect")?.addEventListener("change", async (event) => {
    const runId = event.target.value;
    const url = runId === state.activeRunId ? globalThis.location.pathname : `?run_id=${encodeURIComponent(runId)}`;
    // `replaceState`, not `pushState`: switching runs is filtering a view, and
    // filling the back button with report views is not what a reader means by back.
    globalThis.history.replaceState(null, "", url);
    await load(runId);
  });
}

async function boot() {
  wireTheme();
  wireRunPicker();
  // One observer for both canvases. It fires on a window resize, on a panel
  // reflowing, and once when the layout first settles - which is the moment a
  // canvas finally has a width to be sized against.
  observeResize(
    [$("#calibrationChart"), $("#thresholdChart"), $("#embargoChart"), $("#vintageChart")],
    redraw,
  );

  // Not awaited alongside the reports: the score panel needs only `/api/schema`,
  // and a reader who came to try a prediction should not wait on seven report
  // fetches for the form to appear.
  mountScorePanel({
    form: $("#scoreForm"),
    body: $("#scoreFields"),
    banner: $("#scoreBanner"),
    result: $("#scoreResult"),
    exampleButton: $("#scoreExample"),
    submitButton: $("#scoreSubmit"),
  });

  try {
    await load(runIdFromUrl());
  } catch (error) {
    // Anything that reaches here is a bug in this file rather than a failed
    // request - `load` handles those - so it says so, and the console keeps the
    // stack.
    setStatus($("#status"), "The dashboard failed to render", "error");
    setBanner($("#healthBanner"), String(error?.message || error), "danger");
    throw error;
  }
}

/** Re-read everything from the server. Used after a retrain. */
export async function reload() {
  invalidate();
  await load(state.runId);
}

boot();
