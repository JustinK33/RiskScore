#!/usr/bin/env python3
"""Drive the whole project end to end on synthetic data, and assert the contract.

This is the only automated proof that the artifact contract holds. ``reports/``
is gitignored, so nothing in the test suite ever sees a real run tree published
by the real CLI - the suite builds bundles through :func:`train_run` in a
``tmp_path`` and asserts on the objects. That leaves a specific gap: a filename
constant renamed in one place, a report the dashboard fetches that the pipeline
stopped writing, a card template variable nobody substituted. Each of those is
invisible to unit tests and fatal in a clone-and-run demo.

So this walks the documented path with nothing but a synthetic extract:

    make-sample-data -> compare -> activate -> card -> serve -> predict

and then checks the things a reader would check: every artifact the API
allowlists exists on disk, every report endpoint answers, ``/predict`` returns a
probability and reason codes that sum back, and no ``$placeholder`` survived into
the model card.

Runs in CI on every push, and locally in about a minute:

    python scripts/smoke_e2e.py [--workdir DIR] [--rows N]

Exits 0 when every check passes, 1 with the failures listed. Leaves the work
directory behind when given one, so a failure can be inspected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

#: Every file the service allowlists at ``/artifacts/{run_id}/{name}``, which is
#: also every file the dashboard's ARTIFACTS list links. Spelled literally rather
#: than imported from :mod:`risk_score.artifacts`: importing the constants would
#: make this script agree with the code by construction, and agreeing with the
#: code is precisely what is not being tested. A rename has to be made here too,
#: and that second edit is the check.
EXPECTED_ARTIFACTS = (
    "manifest.json",
    "metrics.json",
    "model_card.md",
    "calibration_validation.csv",
    "calibration_test.csv",
    "threshold_costs_validation.csv",
    "metrics_by_vintage.csv",
    "psi_score.csv",
    "psi_features.csv",
    "shap_summary.csv",
    "figures/calibration_test.png",
    "run.log",
)

#: The bundle is deliberately NOT in the list above - it must exist on disk and
#: must never be served. Both halves are asserted.
BUNDLE = "model.joblib"

#: Report endpoints the dashboard reads. ``/api/comparison`` is included because
#: this run tree comes from `compare`, so unlike a plain train run it must have
#: one - a 404 here would mean the comparison was published somewhere the service
#: does not look.
REPORT_PATHS = (
    "/healthz",
    "/readyz",
    "/api/model",
    "/api/schema",
    "/api/metrics",
    "/api/calibration",
    "/api/threshold-costs",
    "/api/vintages",
    "/api/drift",
    "/api/shap-summary",
    "/api/comparison",
    "/api/runs",
)

#: One applicant, in the raw extract's own string dialect - the harder case, and
#: the one a client copying values out of a Lending Club CSV sends. Kept in step
#: with the `applicant` fixture in tests/conftest.py by hand.
APPLICANT = {
    "issue_d": "Jun-2015",
    "loan_amnt": 15000,
    "term": " 36 months",
    "purpose": "debt_consolidation",
    "annual_inc": 62000,
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "addr_state": "CA",
    "dti": 18.2,
    "delinq_2yrs": 0,
    "earliest_cr_line": "Aug-2003",
    "inq_last_6mths": 1,
    "open_acc": 9,
    "pub_rec": 0,
    "revol_bal": 12000,
    "revol_util": "62.5%",
    "total_acc": 21,
}

HOST = "127.0.0.1"
PORT = 8399
BASE = f"http://{HOST}:{PORT}"

problems: list[str] = []


def check(condition: bool, message: str) -> bool:
    """Record a failure without stopping, and report whether it passed.

    Every check runs, because four broken artifacts should produce four lines in
    one run rather than four invocations. The return value is for the few checks
    whose successors would be meaningless - there is nothing to say about a
    payload that never arrived.
    """
    if not condition:
        problems.append(message)
    return condition


def run(*command: str) -> str:
    """Run a CLI command, echo it, and fail loudly with its own output."""
    print(f"$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        # The child's stderr is the diagnosis. Swallowing it and raising
        # "command failed" is how a CI log becomes useless.
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(f"{command[0]} {command[1]} exited {completed.returncode}")
    return completed.stdout


def get(path: str, timeout: float = 10.0) -> tuple[int, object]:
    """GET a path, returning the status and the parsed body (or the raw text).

    A status of ``0`` means the connection itself failed. That is a legitimate
    answer rather than an error, because :func:`wait_for_ready` polls before the
    listener exists and a refused connection is what "not yet" looks like.
    """
    request = urllib.request.Request(f"{BASE}{path}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # `errors="replace"` because one allowlisted artifact is a PNG. Nothing
            # here inspects a figure's bytes, only that it arrived non-empty, and a
            # decode error would otherwise abort the run on a passing check.
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        status = error.code
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        return 0, f"{type(error).__name__}: {error}"
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def post(path: str, payload: dict[str, object]) -> tuple[int, object]:
    """POST JSON and return the status and parsed body."""
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8") or "null")


def wait_for_ready(process: subprocess.Popen[bytes], log: Path, deadline: float = 60.0) -> None:
    """Poll /readyz until the bundle is loaded, or fail with the server's own log.

    Polls a real signal rather than sleeping a guessed interval, and checks that
    the child is still alive on every pass - otherwise a service that refused to
    boot produces a 60-second timeout instead of the reason it refused.
    """
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if process.poll() is not None:
            sys.stderr.write(log.read_text(encoding="utf-8"))
            raise SystemExit(f"the service exited {process.returncode} before becoming ready")
        status, body = get("/readyz", timeout=2.0)
        if status == 200 and isinstance(body, dict) and body.get("bundle_loaded"):
            return
        time.sleep(0.5)
    sys.stderr.write(log.read_text(encoding="utf-8"))
    raise SystemExit("the service never became ready")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, help="keep the run tree here instead of a tempdir")
    # 4000 rows is the floor that still leaves positives in all three partitions
    # after the embargo, at roughly half the wall time of the 8000-row default.
    parser.add_argument("--rows", type=int, default=4000, help="synthetic loans (default: 4000)")
    args = parser.parse_args()

    keep = args.workdir is not None
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="riskscore-smoke-"))
    workdir.mkdir(parents=True, exist_ok=True)
    reports = workdir / "reports"
    dataset = workdir / "loans.csv"

    try:
        return smoke(workdir, reports, dataset, args.rows)
    finally:
        if keep:
            print(f"\nwork directory kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def smoke(workdir: Path, reports: Path, dataset: Path, rows: int) -> int:
    run("riskscore", "make-sample-data", str(dataset), "--rows", str(rows), "--seed", "7")
    check(dataset.exists(), "make-sample-data wrote no CSV")

    # `compare` rather than `train`, because it is the only path that publishes
    # comparison.json, and a comparison the dashboard cannot find is a panel that
    # says nothing has been compared on a tree where something has.
    run(
        "riskscore",
        "compare",
        str(dataset),
        "--output-dir",
        str(reports),
        "--tiers",
        "both",
        "--model",
        "logistic_regression",
        "--cache-dir",
        str(workdir / "cache"),
    )

    check((reports / "registry.json").exists(), "compare wrote no registry.json")
    check((reports / "comparison.json").exists(), "compare wrote no comparison.json at the root")

    registry = json.loads((reports / "registry.json").read_text(encoding="utf-8"))
    entries = registry["runs"] if isinstance(registry, dict) else registry
    if not check(bool(entries), "the registry is empty"):
        return report()

    # Nothing is activated by measuring, so `compare` leaves active_run.json
    # absent by design - asserted here rather than assumed, because the service
    # would then boot with no bundle and every check below would fail obscurely.
    check(
        not (reports / "active_run.json").exists(),
        "compare activated a run; choosing what to serve is supposed to be explicit",
    )

    run_id = str(entries[0]["run_id"])
    run("riskscore", "activate", run_id, "--output-dir", str(reports))
    check((reports / "active_run.json").exists(), "activate wrote no active_run.json")

    run_dir = reports / "runs" / run_id
    for name in EXPECTED_ARTIFACTS:
        path = run_dir / name
        check(path.exists(), f"{name} is missing from the published run")
        if path.exists():
            check(path.stat().st_size > 0, f"{name} is empty")
    check((run_dir / BUNDLE).exists(), f"{BUNDLE} is missing from the published run")

    # Every variant, not just the activated one: a report written only for the
    # baseline is a run picker whose other options render as dashes.
    for entry in entries:
        other = reports / "runs" / str(entry["run_id"])
        missing = [name for name in EXPECTED_ARTIFACTS if not (other / name).exists()]
        check(not missing, f"run {entry['run_id']} is missing {', '.join(missing)}")

    card = run("riskscore", "card", run_id, "--output-dir", str(reports))
    # A template variable that reached the output means the card claims a fact it
    # does not have, which is worse than omitting it.
    for marker in ("$", "{{", "None", "nan"):
        check(marker not in card, f"the model card contains {marker!r}")
    check(len(card) > 1000, f"the model card is only {len(card)} characters")
    check(run_id in card, "the model card does not name its own run")

    log = workdir / "serve.log"
    environment = dict(os.environ)
    # Every default is the safe one and this asserts it: no upload, no retrain,
    # bound to loopback. The service refuses to boot on an unsafe combination, so
    # a regression there fails here rather than in production.
    environment.pop("RISKSCORE_ALLOW_UPLOAD", None)
    environment.pop("RISKSCORE_ALLOW_RETRAIN", None)
    with log.open("wb") as sink:
        service = subprocess.Popen(
            [
                "riskscore",
                "serve",
                "--output-dir",
                str(reports),
                "--host",
                HOST,
                "--port",
                str(PORT),
            ],
            stdout=sink,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    try:
        wait_for_ready(service, log)
        check_service(run_id)
    finally:
        service.terminate()
        try:
            service.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # A service that ignores SIGTERM would hang CI. Killing it is right
            # here and is itself worth reporting.
            service.kill()
            problems.append("the service did not exit on SIGTERM")

    return report()


def check_service(run_id: str) -> None:
    """Every assertion that needs the service running."""
    for path in REPORT_PATHS:
        status, body = get(path)
        check(status == 200, f"GET {path} answered {status}")
        check(body is not None, f"GET {path} returned an empty body")

    status, model = get("/api/model")
    if check(isinstance(model, dict), "GET /api/model is not an object") and isinstance(
        model, dict
    ):
        check(model.get("run_id") == run_id, f"the service is serving {model.get('run_id')!r}")

    # The dashboard's own readiness contract. `mutating_routes` decides whether
    # the retrain panel mounts at all, and "none" is what a default install must
    # report - a default that offered the routes would be the security
    # regression this field exists to make visible.
    status, ready = get("/readyz")
    if isinstance(ready, dict):
        check(
            ready.get("mutating_routes") == "none",
            f"a default service reports mutating_routes={ready.get('mutating_routes')!r}",
        )

    # The applicant is the body itself, and `explain`/`top_k` are query
    # parameters - so a bare `POST /predict` with a raw extract row scores it,
    # which is what makes the endpoint usable from curl without a wrapper object.
    status, predicted = post("/predict?top_k=5", APPLICANT)
    if not check(status == 200, f"POST /predict answered {status}: {predicted!r}"):
        return
    assert isinstance(predicted, dict)

    probability = predicted.get("default_probability")
    check(
        isinstance(probability, (int, float)) and 0.0 < float(probability) < 1.0,
        f"/predict returned default_probability={probability!r}",
    )
    check(
        predicted.get("decision") in {"approve", "decline"},
        f"decision={predicted.get('decision')!r}",
    )
    check(isinstance(predicted.get("threshold"), float), "/predict returned no threshold")
    # Identity on every response, because "which model declined this" is the first
    # question asked about any decline and it cannot be reconstructed later.
    identity = predicted.get("model")
    if check(isinstance(identity, dict), "/predict carries no model identity"):
        assert isinstance(identity, dict)
        check(identity.get("run_id") == run_id, "/predict names a different run than /api/model")
    check(isinstance(predicted.get("latency_ms"), float), "/predict reports no latency")

    reasons = predicted.get("reasons")
    if check(isinstance(reasons, list) and bool(reasons), "/predict returned no reason codes"):
        assert isinstance(reasons, list)
        check(len(reasons) <= 5, f"top_k=5 returned {len(reasons)} reason codes")
        for reason in reasons:
            check(bool(reason.get("feature")), "a reason code has no feature name")
            check(bool(reason.get("label")), f"{reason.get('feature')} has no human label")
            check("log_odds" in reason, "a reason code has no log-odds contribution")
            check(
                reason.get("direction") in {"increases risk", "reduces risk", "no effect"},
                f"direction={reason.get('direction')!r}",
            )

    # The exactness claim, checked rather than trusted: the contributions plus the
    # baseline must reproduce the model's own log-odds. This is the property that
    # separates real SHAP from a plausible-looking bar chart, and it is the one
    # thing about the explanation a reader cannot verify by looking.
    #
    # Asked again at the maximum top_k, because the five above are truncated and a
    # truncated set is supposed to not sum. Only asserted when the response came
    # back under the cap, since at the cap it would be truncated too.
    _, full = post("/predict?top_k=50", APPLICANT)
    if isinstance(full, dict) and isinstance(full.get("reasons"), list):
        contributions = full["reasons"]
        if check(len(contributions) < 50, f"{len(contributions)} reasons, so the set is truncated"):
            summed = float(full["baseline_log_odds"]) + sum(
                float(reason["log_odds"]) for reason in contributions
            )
            check(
                abs(summed - float(full["total_log_odds"])) < 1e-6,
                f"reason codes sum to {summed}, the model says {full['total_log_odds']}",
            )

    # Reason codes roughly triple the latency of a single call, so the cheap path
    # has to actually be available.
    status, plain = post("/predict?explain=false", APPLICANT)
    check(status == 200, f"POST /predict?explain=false answered {status}")
    if isinstance(plain, dict):
        check(plain.get("reasons") == [], "explain=false still returned reason codes")

    # A missing required field must be a 422 naming the field, not a 500 and not a
    # prediction from an imputed value. The dashboard's whole error story is built
    # on the field name being in the response.
    incomplete = {key: value for key, value in APPLICANT.items() if key != "loan_amnt"}
    status, body = post("/predict", incomplete)
    check(status == 422, f"a missing required field answered {status}, want 422")
    check("loan_amnt" in json.dumps(body), "the 422 does not name the missing field")

    # An unknown field is rejected rather than ignored, because a typo silently
    # dropped is a score computed from an imputed value with nothing to show it.
    status, body = post("/predict", {**APPLICANT, "loan_amount": 15000})
    check(status == 422, f"an unknown field answered {status}, want 422")

    status, batch = post("/predict/batch", {"applicants": [APPLICANT, incomplete]})
    if check(status == 200, f"POST /predict/batch answered {status}: {batch!r}") and isinstance(
        batch, dict
    ):
        # Per-row errors inline: failing all rows because one is malformed is a
        # worse answer than one score and one message.
        check(batch.get("scored") == 1, f"batch scored {batch.get('scored')}, want 1")
        check(batch.get("failed") == 1, f"batch failed {batch.get('failed')}, want 1")

    # The pickle must exist on disk and must never be reachable over HTTP.
    status, _ = get(f"/artifacts/{run_id}/{BUNDLE}")
    check(status == 404, f"the model pickle is served ({status})")
    for name in EXPECTED_ARTIFACTS:
        status, _ = get(f"/artifacts/{run_id}/{name}")
        check(status == 200, f"GET /artifacts/{run_id}/{name} answered {status}")

    # The dashboard is served by this same process, which is the sentence in its
    # own footer and the reason there is no second server.
    for path in ("/", "/styles/tokens.css", "/styles/app.css", "/js/main.js"):
        status, _ = get(path)
        check(status == 200, f"GET {path} answered {status}")


def report() -> int:
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        print(f"\n{len(problems)} failing check(s)")
        return 1
    print("\nend-to-end smoke clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
