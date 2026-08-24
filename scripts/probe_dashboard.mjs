/**
 * Measure the dashboard in a real browser at every breakpoint.
 *
 * Screenshots catch a wrong colour; they do not catch a 24px horizontal
 * scrollbar, and they cannot tell you *which* element caused one. This drives
 * headless Chrome over the DevTools Protocol, so the assertions are the same ones
 * the plan states: no horizontal overflow at any width in either theme, every
 * canvas actually sized, every panel populated, and no console error.
 *
 * It also drives the score panel end to end - load the example, submit, wait for a
 * verdict - because the form is generated from `/api/schema` at runtime, so nothing
 * short of a real browser talking to a real server proves the generated controls
 * round-trip their values back into a request the model accepts.
 *
 * Not a pytest test, and deliberately not in CI: it needs a running server and an
 * installed Chrome. It is the pixel-perfection check to run by hand after touching
 * the dashboard.
 *
 *   node scripts/probe_dashboard.mjs [base-url]
 *
 * Exits non-zero on the first failure, and names the widest overflowing element
 * rather than just reporting that the page overflows.
 */

import { spawn } from "node:child_process";
import { once } from "node:events";

const BASE = process.argv[2] || "http://127.0.0.1:8125/";
const PORT = 9333;
const WIDTHS = [320, 520, 768, 900, 1024, 1440];
const THEMES = ["light", "dark"];

const CHROME =
  process.env.CHROME ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

/**
 * The page-side probe. Runs in the browser, returns plain JSON.
 *
 * Overflow is attributed by walking every element and comparing its right edge to
 * the document's client width, because "the page scrolls sideways" is not
 * actionable and "`.panel-header > .field.inline` ends at 344px in a 296px
 * viewport" is.
 */
const PROBE = `(() => {
  const doc = document.documentElement;
  const limit = doc.clientWidth;
  const offenders = [];
  for (const node of document.querySelectorAll("body *")) {
    const rect = node.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) continue;
    // 1px of slack: a fractional layout width rounds up and is not a scrollbar.
    if (rect.right > limit + 1) {
      offenders.push({
        selector: node.tagName.toLowerCase() + (node.id ? "#" + node.id : "") +
          (node.className && typeof node.className === "string"
            ? "." + node.className.trim().split(/\\s+/).join(".")
            : ""),
        right: Math.round(rect.right),
      });
    }
  }
  offenders.sort((a, b) => b.right - a.right);
  const canvases = [...document.querySelectorAll("canvas")].map((canvas) => ({
    id: canvas.id,
    cssWidth: Math.round(canvas.getBoundingClientRect().width),
    bitmapWidth: canvas.width,
    label: canvas.getAttribute("aria-label") || "",
  }));
  return JSON.stringify({
    clientWidth: limit,
    scrollWidth: doc.scrollWidth,
    theme: doc.dataset.theme || "(os)",
    background: getComputedStyle(document.body).backgroundColor,
    status: document.querySelector("#status")?.textContent || "",
    statusState: document.querySelector("#status")?.dataset.state || "",
    metrics: [...document.querySelectorAll(".metric strong")].map((n) => n.textContent),
    identityRows: document.querySelectorAll("#identityGrid .def").length,
    flowChips: document.querySelectorAll("#rowFlow .chip").length,
    detailRows: document.querySelectorAll("#runDetails .def").length,
    artifacts: document.querySelectorAll("#artifactLinks a").length,
    tables: [...document.querySelectorAll(".data-fallback table")].map(
      (t) => t.querySelectorAll("tbody tr").length,
    ),
    banners: [...document.querySelectorAll(".banner:not([hidden])")].map((n) => n.textContent),
    scoreFields: document.querySelectorAll("#scoreFields .field").length,
    // Every categorical must be a select. A text box there is the typo-pooling
    // hole the /api/schema choices list exists to close, and it looks identical
    // in a screenshot.
    scoreSelects: document.querySelectorAll("#scoreFields select").length,
    scoreGroups: document.querySelectorAll("#scoreFields fieldset").length,
    verdict: document.querySelector("#scoreResult .verdict")?.dataset.decision || "",
    reasonRows: document.querySelectorAll("#scoreResult .reason-table tbody tr").length,
    // A bar with no width is a reason code the reader cannot see; the renderer has
    // a 2% floor precisely so this can never be 0. Bars in the tornado column that
    // the narrow layout drops are not rendered at all, so they are excluded rather
    // than counted - the log-odds text is the reading at those widths.
    narrowBars: [...document.querySelectorAll("#scoreResult .bar")].filter(
      (bar) => bar.getClientRects().length > 0 && bar.getBoundingClientRect().width < 1,
    ).length,
    offenders: offenders.slice(0, 5),
    canvases,
  });
})()`;

/**
 * Drive the score panel the way a reader does: load the example, submit, wait for
 * a verdict. Returns "" on success or the reason it failed.
 *
 * This is the only automated check that the generated form is submittable at all.
 * `buildForm` reads its values back out of live DOM nodes, so a control whose
 * `value` does not round-trip - a `type="month"` given "Jun-2015", a required
 * select with no matching option - produces either a browser validation block or a
 * 422, and both are invisible to a unit test.
 */
const SCORE = `(async () => {
  const example = document.querySelector("#scoreExample");
  const form = document.querySelector("#scoreForm");
  if (!example || !form) return "the score panel is absent";
  example.click();
  const missing = [...form.elements].filter((node) => node.willValidate && !node.checkValidity());
  if (missing.length) {
    return "the example does not satisfy the form: " + missing.map((n) => n.name).join(", ");
  }
  form.requestSubmit();
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (document.querySelector("#scoreResult .verdict")) return "";
    const banner = document.querySelector("#scoreBanner:not([hidden])");
    if (banner && banner.dataset.tone === "danger") return "POST /predict: " + banner.textContent;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return "no verdict rendered within 10s";
})()`;

/** One CDP session against a fresh tab. */
async function session(width, height, theme) {
  const target = await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, {
    method: "PUT",
  }).then((r) => r.json());
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await once(socket, "open");

  let id = 0;
  const waiting = new Map();
  const events = [];
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && waiting.has(message.id)) {
      waiting.get(message.id)(message);
      waiting.delete(message.id);
    } else if (message.method) {
      events.push(message);
    }
  });
  const send = (method, params = {}) =>
    new Promise((resolve) => {
      const next = ++id;
      waiting.set(next, resolve);
      socket.send(JSON.stringify({ id: next, method, params }));
    });

  await send("Runtime.enable");
  await send("Log.enable");
  // The profile directory persists between runs, and the server sends an ETag with
  // no `Cache-Control`, so Chrome heuristically reuses a stylesheet it already has.
  // That made this script report a CSS fix as still broken - a probe that reads
  // stale bytes is worse than no probe, so the cache is off for every session.
  await send("Network.enable");
  await send("Network.setCacheDisabled", { cacheDisabled: true });
  await send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: false,
  });
  // The theme is emulated rather than written to localStorage, so this measures the
  // OS-preference path - the one that has no JS involved and therefore no flash.
  await send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-color-scheme", value: theme }],
  });

  await send("Page.enable");
  await send("Page.navigate", { url: BASE });
  // The dashboard's own status region is the readiness signal: it is only set once
  // every fetch has settled and every panel has rendered.
  const deadline = Date.now() + 15000;
  let result;
  for (;;) {
    const response = await send("Runtime.evaluate", {
      expression: PROBE,
      returnByValue: true,
      awaitPromise: false,
    });
    result = response.result?.result?.value ? JSON.parse(response.result.result.value) : null;
    if (result && result.statusState !== "loading") break;
    if (Date.now() > deadline) throw new Error(`the page never finished loading at ${width}px`);
    await new Promise((resolve) => setTimeout(resolve, 200));
  }

  // Score after the report panels have settled, then re-probe: the verdict and the
  // reason table are the tallest things this page can add, and their overflow has to
  // be measured with them present.
  const scored = await send("Runtime.evaluate", {
    expression: SCORE,
    returnByValue: true,
    awaitPromise: true,
  });
  const scoreError = scored.result?.result?.value ?? "the score probe did not return";
  const after = await send("Runtime.evaluate", { expression: PROBE, returnByValue: true });
  result = JSON.parse(after.result.result.value);

  const consoleErrors = events
    .filter((event) => event.method === "Log.entryAdded" && event.params.entry.level === "error")
    .map((event) => event.params.entry.text);

  socket.close();
  await fetch(`http://127.0.0.1:${PORT}/json/close/${target.id}`);
  return { ...result, scoreError, consoleErrors };
}

const chrome = spawn(
  CHROME,
  [
    "--headless=new",
    "--disable-gpu",
    "--hide-scrollbars",
    `--remote-debugging-port=${PORT}`,
    "--user-data-dir=/tmp/riskscore-probe-profile",
    "about:blank",
  ],
  { stdio: "ignore" },
);

/** Wait for the debugging endpoint rather than sleeping a guessed interval. */
async function waitForChrome() {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      await fetch(`http://127.0.0.1:${PORT}/json/version`);
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw new Error("Chrome never opened its debugging port");
}

let failures = 0;
try {
  await waitForChrome();
  for (const theme of THEMES) {
    for (const width of WIDTHS) {
      const report = await session(width, 1200, theme);
      const problems = [];
      if (report.scrollWidth > report.clientWidth + 1) {
        problems.push(
          `overflows by ${report.scrollWidth - report.clientWidth}px: ` +
            report.offenders.map((o) => `${o.selector} ends at ${o.right}`).join(", "),
        );
      }
      for (const canvas of report.canvases) {
        if (canvas.cssWidth < 100) problems.push(`${canvas.id} is only ${canvas.cssWidth}px wide`);
        if (canvas.bitmapWidth === 0) problems.push(`${canvas.id} has no bitmap`);
        if (!canvas.label || canvas.label.endsWith("loading")) {
          problems.push(`${canvas.id} has no aria-label`);
        }
      }
      if (report.statusState === "error") problems.push(`status: ${report.status}`);
      if (report.metrics.some((value) => value === "-")) problems.push("a metric card is empty");
      if (report.identityRows !== 6) problems.push(`${report.identityRows} identity rows, want 6`);
      if (report.artifacts !== 12) problems.push(`${report.artifacts} artifact links, want 12`);
      if (report.tables.some((rows) => rows === 0)) problems.push("an empty fallback table");
      if (report.scoreFields < 4) problems.push(`${report.scoreFields} score fields, want the schema's`);
      if (report.scoreSelects < 1) problems.push("no categorical rendered as a select");
      if (report.scoreError) problems.push(report.scoreError);
      if (!report.verdict) problems.push("no decision in the verdict block");
      if (report.reasonRows === 0) problems.push("a verdict with no reason codes");
      if (report.narrowBars) problems.push(`${report.narrowBars} reason bar(s) render at zero width`);
      if (report.consoleErrors.length) problems.push(`console: ${report.consoleErrors[0]}`);

      const mark = problems.length ? "FAIL" : "ok  ";
      failures += problems.length ? 1 : 0;
      console.log(
        `${mark} ${theme.padEnd(5)} ${String(width).padStart(4)}px  ` +
          `scroll=${report.scrollWidth} bg=${report.background} ` +
          `canvas=${report.canvases.map((c) => c.cssWidth).join("/")} ` +
          `tables=${report.tables.join("/")} ` +
          `score=${report.scoreFields}f/${report.scoreGroups}g/${report.scoreSelects}s ` +
          `verdict=${report.verdict || "none"}/${report.reasonRows}r ` +
          `banners=${report.banners.length}`,
      );
      for (const problem of problems) console.log(`       - ${problem}`);
    }
  }
} finally {
  chrome.kill();
}

console.log(failures === 0 ? "\nall widths clean" : `\n${failures} failing configuration(s)`);
process.exit(failures === 0 ? 0 : 1);
