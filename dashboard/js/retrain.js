/**
 * Retrain from a CSV: upload, start, poll, reload.
 *
 * This replaces the one POST the old dashboard had, which read a CSV out of a
 * `<textarea>`, retrained on the request thread with no credential, and overwrote
 * the canonical `reports/` tree - including the pickle the service was serving -
 * while every other request waited on the GIL. The three-step flow here exists
 * because each step is a different risk: storing bytes is capped and
 * content-addressed, starting a fit is authenticated and single-flight, and
 * watching one is an ordinary poll that cannot change anything.
 *
 * The panel is only mounted when `/readyz` reports both features on. A form whose
 * every submission is a 403 is worse than no form: it reads as a broken feature
 * rather than a switched-off one, and the operator who needs it is reading the
 * runbook, not this page.
 *
 * Nothing here is optimistic. The new run is not selected, the panels are not
 * patched, and no status is inferred from a timer - the job's own status decides,
 * and a success reloads every report from the server, because a retrain has
 * changed what every panel on the page describes.
 */

import { setApiKey, startRetrain, uploadDataset, waitForJob } from "./api.js";
import { setBanner, setText } from "./dom.js";

/** The two the service accepts. Mirrors `RetrainIn.model_type`. */
const MODEL_TYPES = ["logistic_regression", "xgboost"];

/** The tier option that maps to `include_lender_priced: true`. */
const LENDER_PRICED_TIER = "with_lender_priced";

/**
 * One line of progress, from a job payload.
 *
 * The stage is not reported by the service - a fit is one child process with no
 * checkpoints to report from - so this says what is true rather than inventing a
 * percentage: which model, on which dataset, and how long it has been running.
 */
export function jobLine(job, { elapsedSeconds = null } = {}) {
  if (!job) return "";
  const what = `${job.model_type} on ${job.dataset}`;
  const forHowLong = elapsedSeconds === null ? "" : ` (${Math.round(elapsedSeconds)}s)`;
  switch (job.status) {
    case "running":
      return `Fitting ${what}${forHowLong}. This is a full training run, so it takes as long as \`riskscore train\` does.`;
    case "succeeded":
      return `Finished: ${job.run_id}. It is published and active; every panel below now describes it.`;
    case "timed_out":
      return `The fit was killed after exceeding the server's timeout. ${job.detail || ""}`.trim();
    case "failed":
      // The child's exception text, verbatim: it is the only description of what
      // went wrong, and this route already required a credential to reach.
      return `The fit failed: ${job.detail || "no detail was reported."}`;
    default:
      return `Unknown job status: ${job.status}.`;
  }
}

/**
 * Wire the panel. Returns without doing anything when the service has the
 * features switched off, which is the default and the safe case.
 *
 * `now` is injectable so the elapsed counter is testable without a clock.
 */
export function mountRetrainPanel({
  section,
  form,
  banner,
  progress,
  submitButton,
  mutatingRoutes = "none",
  onDone = null,
  now = () => Date.now(),
}) {
  if (!section || !form) return;
  // Both, not either: the browser has no server-side dataset id to refer to, so
  // uploading is the only way it can name one, and a retrain route with uploads
  // off is reachable only from the CLI.
  if (mutatingRoutes !== "both") return;
  section.hidden = false;

  const modelSelect = form.elements.namedItem("model_type");
  if (modelSelect) {
    // Populated here rather than in the HTML so the markup cannot drift from the
    // list the service accepts.
    modelSelect.replaceChildren(
      ...MODEL_TYPES.map((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        return option;
      }),
    );
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = form.elements.namedItem("dataset")?.files?.[0];
    if (!file) {
      setBanner(banner, "Choose a CSV to train on.", "warn");
      return;
    }
    // Stored before the request rather than read from the field per call: the key
    // then covers the upload, the start, and every poll, and it lives in
    // `sessionStorage` so it does not outlive the tab.
    setApiKey(form.elements.namedItem("api_key")?.value?.trim() || "");

    if (submitButton) submitButton.disabled = true;
    setBanner(banner, null);
    const started = now();
    try {
      setText(progress, `Uploading ${file.name}.`);
      const dataset = await uploadDataset(file);
      setText(
        progress,
        dataset.existing
          ? `Those exact bytes were already stored as ${dataset.dataset_id}, so nothing was uploaded twice.`
          : `Stored ${dataset.dataset_id} (${dataset.size_bytes} bytes).`,
      );

      const receipt = await startRetrain({
        datasetId: dataset.dataset_id,
        modelType: modelSelect?.value,
        // A tier by name rather than a checkbox: the page calls the two tiers
        // `origination_only` and `with_lender_priced` everywhere else, and a
        // checkbox labelled `include_lender_priced` is the same choice spelled a
        // third way.
        includeLenderPriced: form.elements.namedItem("feature_tier")?.value === LENDER_PRICED_TIER,
      });

      const job = await waitForJob(receipt.job_id, {
        onUpdate: (update) =>
          setText(progress, jobLine(update, { elapsedSeconds: (now() - started) / 1000 })),
      });

      setText(progress, jobLine(job));
      if (job.status !== "succeeded") {
        setBanner(banner, jobLine(job), "danger");
        return;
      }
      // Only now, and only from the server: the run is published and active, so
      // every report on the page describes the previous model until it is re-read.
      await onDone?.(job.run_id);
    } catch (error) {
      setBanner(banner, error.message, "danger");
    } finally {
      if (submitButton) submitButton.disabled = false;
    }
  });
}
