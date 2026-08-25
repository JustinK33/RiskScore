/**
 * `node --test` from `dashboard/`. No browser, no server, no file picker.
 *
 * Two things are worth testing here and the rest is DOM wiring a unit test cannot
 * reach honestly. `jobLine` is the only description an operator gets of a fit that
 * takes minutes, and a failed job whose line reads like a success - or whose
 * `detail` is dropped - leaves them with a stale model and no reason. And the mount
 * guard decides whether the panel is offered at all, which is a security-shaped
 * question: the default has to be off.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { jobLine, mountRetrainPanel } from "./retrain.js";

const RUNNING = {
  job_id: "j1",
  status: "running",
  dataset: "ab12cd34",
  model_type: "logistic_regression",
  submitted_at: "2026-08-25T00:00:00Z",
};

test("a running job names the model, the dataset, and how long it has been going", () => {
  const line = jobLine(RUNNING, { elapsedSeconds: 42.4 });

  assert.match(line, /logistic_regression on ab12cd34/);
  assert.match(line, /\(42s\)/);
});

test("a succeeded job names the run, because that is what to look at next", () => {
  const line = jobLine({ ...RUNNING, status: "succeeded", run_id: "20260825T0000Z-lr-x-abc1234" });

  assert.match(line, /20260825T0000Z-lr-x-abc1234/);
});

test("a failed job carries the child's own detail verbatim", () => {
  const line = jobLine({ ...RUNNING, status: "failed", detail: "no rows survived the embargo" });

  assert.match(line, /failed/);
  assert.match(line, /no rows survived the embargo/);
});

test("a failed job with no detail says so rather than reading as a success", () => {
  const line = jobLine({ ...RUNNING, status: "failed" });

  assert.match(line, /failed/);
  assert.doesNotMatch(line, /undefined|null/);
});

test("a timed-out job is distinguished from a failed one", () => {
  const line = jobLine({ ...RUNNING, status: "timed_out" });

  assert.match(line, /killed/);
});

test("an unrecognised status is reported, not silently rendered as running", () => {
  const line = jobLine({ ...RUNNING, status: "sideways" });

  assert.match(line, /Unknown job status: sideways/);
});

test("no job is an empty line rather than the string undefined", () => {
  assert.equal(jobLine(null), "");
});

/** The two nodes the guard touches, with only the surface it uses. */
const stubPanel = () => ({
  section: { hidden: true },
  form: { addEventListener: () => {}, elements: { namedItem: () => null } },
});

test("the panel stays hidden unless the service reports both features on", () => {
  for (const mutatingRoutes of ["none", "upload", "retrain", undefined]) {
    const nodes = stubPanel();
    mountRetrainPanel({ ...nodes, mutatingRoutes });

    assert.equal(nodes.section.hidden, true, `mounted for ${mutatingRoutes}`);
  }
});

test("the panel appears when both are on", () => {
  const nodes = stubPanel();
  mountRetrainPanel({ ...nodes, mutatingRoutes: "both" });

  assert.equal(nodes.section.hidden, false);
});
