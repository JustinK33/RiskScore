"""End-to-end tests for the scoring service, through a real ASGI client.

Through ``TestClient`` rather than by calling the handlers, because most of what
this package adds is *not* in the handlers. The body cap, the request id, the
host allow-list and the error shape are all middleware and exception handlers,
and a test that calls ``predict(...)`` directly exercises none of them.

The interesting cases here are the ones a unit test cannot see:

* both real extract dialects - ``' 36 months'`` and ``36`` - must produce the
  *identical* probability, because that equality is the entire reason
  ``CanonicalizeFrame`` owns parsing;
* a typo must be a 422, not a score computed from an imputed median;
* no error body may contain a filesystem path.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from risk_score.api import Settings, create_app
from risk_score.api.app import summarize_validation_errors
from risk_score.api.deps import require_api_key
from risk_score.api.reports import ARTIFACT_FILENAMES, REPORTS, RUN_ID_PATTERN
from risk_score.pipeline import RunResult

#: Every report a plain ``riskscore train`` writes. ``comparison`` is excluded
#: because only ``riskscore compare`` produces it, and its absence is asserted
#: separately as a 404.
TRAIN_REPORTS = tuple(name for name in REPORTS if name != "comparison")

# --- 1. scoring ----------------------------------------------------------------


def test_predict_scores_an_applicant(client: TestClient, applicant: dict[str, object]) -> None:
    response = client.post("/predict", json=applicant)

    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["decision"] in {"approve", "decline"}
    # The threshold travels with the bundle, so a response can be re-derived from
    # its own contents: the decision must follow from the probability and the
    # threshold that came back with it.
    expected = "decline" if body["default_probability"] >= body["threshold"] else "approve"
    assert body["decision"] == expected
    assert body["latency_ms"] > 0.0


def test_predict_accepts_both_extract_dialects(
    client: TestClient, applicant: dict[str, object]
) -> None:
    """The string dialect and the numeric one are the same applicant.

    This is the regression test for the bug found by actually posting to the
    service: ``_TYPE_BY_PARSE_KIND`` forced these three fields to ``float``, so
    ``emp_length: '5 years'`` - the primary real-world spelling - was a 422. Equal
    probabilities are the assertion that matters; a 200 alone would have passed
    even if the strings were being parsed differently.
    """
    numeric = {**applicant, "term": 36, "emp_length": 5, "revol_util": 62.5}

    strings = client.post("/predict", json=applicant)
    numbers = client.post("/predict", json=numeric)

    assert strings.status_code == 200, strings.text
    assert numbers.status_code == 200, numbers.text
    assert strings.json()["default_probability"] == numbers.json()["default_probability"], (
        "the two extract dialects must parse to the same applicant, exactly"
    )


def test_predict_accepts_a_source_alias(client: TestClient, applicant: dict[str, object]) -> None:
    """``funded_amnt`` is what one extract calls ``loan_amnt``."""
    aliased = {key: value for key, value in applicant.items() if key != "loan_amnt"}
    aliased["funded_amnt"] = applicant["loan_amnt"]

    response = client.post("/predict", json=aliased)

    assert response.status_code == 200, response.text
    assert response.json()["default_probability"] == pytest.approx(
        client.post("/predict", json=applicant).json()["default_probability"]
    )


def test_predict_refuses_an_unknown_field(client: TestClient, applicant: dict[str, object]) -> None:
    """The single most important validation behaviour in the service.

    ``anual_inc`` accepted as extra data means ``annual_inc`` is absent, imputed
    to a median, and the caller gets a confident 200 describing a *different
    applicant*. ``extra="forbid"`` is what makes that a 422 instead.
    """
    typo = {key: value for key, value in applicant.items() if key != "annual_inc"}
    typo["anual_inc"] = 62000

    response = client.post("/predict", json=typo)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "anual_inc" in detail
    assert "not permitted" in detail


def test_predict_names_a_missing_required_field(
    client: TestClient, applicant: dict[str, object]
) -> None:
    without = {key: value for key, value in applicant.items() if key != "loan_amnt"}

    response = client.post("/predict", json=without)

    assert response.status_code == 422
    assert "loan_amnt" in response.json()["detail"]


def test_predict_reasons_are_ordered_by_absolute_contribution(
    client: TestClient, applicant: dict[str, object]
) -> None:
    body = client.post("/predict", json=applicant, params={"top_k": 5}).json()

    reasons = body["reasons"]
    assert reasons, "a logistic-regression bundle carries a background and can explain"
    assert len(reasons) <= 5
    magnitudes = [abs(reason["log_odds"]) for reason in reasons]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert body["baseline_log_odds"] is not None


def test_predict_without_explain_omits_reasons(
    client: TestClient, applicant: dict[str, object]
) -> None:
    body = client.post("/predict", json=applicant, params={"explain": "false"}).json()

    assert body["reasons"] == []
    assert body["baseline_log_odds"] is None
    assert body["total_log_odds"] is None


def test_predict_options_reject_an_unknown_query_param(
    client: TestClient, applicant: dict[str, object]
) -> None:
    """``explain`` misspelled must not silently mean "reasons on".

    Same argument as ``extra="forbid"`` on the body: a knob that is ignored when
    misspelled is a knob nobody can rely on.
    """
    response = client.post("/predict", json=applicant, params={"expalin": "false"})

    assert response.status_code == 422


def test_predict_reports_the_model_that_scored(
    client: TestClient, applicant: dict[str, object], trained_run: RunResult
) -> None:
    identity = client.post("/predict", json=applicant).json()["model"]

    assert identity["run_id"] == trained_run.metadata.run_id
    assert identity["feature_tier"] == "origination_only"
    assert identity["bundle_schema_version"] >= 1


# --- 2. batch ------------------------------------------------------------------


def test_batch_scores_every_row(client: TestClient, applicant: dict[str, object]) -> None:
    response = client.post("/predict/batch", json={"applicants": [applicant, applicant]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scored"] == 2
    assert body["failed"] == 0
    assert [row["index"] for row in body["rows"]] == [0, 1]
    first, second = (row["prediction"]["default_probability"] for row in body["rows"])
    assert first == second


def test_batch_matches_single_scoring(client: TestClient, applicant: dict[str, object]) -> None:
    """A vectorized pass and a one-row pass must agree.

    Not a tautology: the batch builds one frame for many rows, and a
    ``groupby``-shaped bug or a stray fit would show up here as a different
    number for the same applicant.
    """
    single = client.post("/predict", json=applicant, params={"explain": "false"}).json()
    batch = client.post("/predict/batch", json={"applicants": [applicant]}).json()

    assert batch["rows"][0]["prediction"]["default_probability"] == pytest.approx(
        single["default_probability"]
    )


def test_batch_reports_bad_rows_inline(client: TestClient, applicant: dict[str, object]) -> None:
    """Row 1 is broken; rows 0 and 2 must still come back scored."""
    broken = {key: value for key, value in applicant.items() if key != "loan_amnt"}

    body = client.post("/predict/batch", json={"applicants": [applicant, broken, applicant]}).json()

    assert body["scored"] == 2
    assert body["failed"] == 1
    assert [row["index"] for row in body["rows"]] == [0, 1, 2]
    assert body["rows"][1]["prediction"] is None
    assert "loan_amnt" in body["rows"][1]["error"]
    assert body["rows"][0]["prediction"] is not None
    assert body["rows"][2]["prediction"] is not None


def test_batch_refuses_more_rows_than_configured(
    api_settings: Settings, applicant: dict[str, object]
) -> None:
    settings = api_settings.model_copy(update={"max_batch_rows": 2})
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.post("/predict/batch", json={"applicants": [applicant] * 3})

    assert response.status_code == 422
    assert "at most 2 rows" in response.json()["detail"]


def test_batch_refuses_an_empty_list(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"applicants": []})

    assert response.status_code == 422


# --- 3. identity, schema, health -----------------------------------------------


def test_model_identity(client: TestClient, trained_run: RunResult) -> None:
    body = client.get("/api/model").json()

    assert body["run_id"] == trained_run.metadata.run_id
    assert body["model_type"] == trained_run.metadata.model_type


def test_schema_describes_every_accepted_input(
    client: TestClient, trained_run: RunResult, applicant: dict[str, object]
) -> None:
    body = client.get("/api/schema").json()

    spec = trained_run.bundle.feature_spec
    assert [field["name"] for field in body["fields"]] == list(spec.raw_inputs)
    assert body["feature_tier"] == "origination_only"
    assert body["engineered"] == list(spec.engineered)
    # The form the dashboard builds from this must be postable as-is: every
    # required field the schema advertises is a key the fixture applicant has.
    required = {field["name"] for field in body["fields"] if field["required"]}
    assert required <= set(applicant)


def test_schema_kinds_are_form_kinds(client: TestClient) -> None:
    """Seven parse kinds collapse to the four a form can render."""
    kinds = {field["kind"] for field in client.get("/api/schema").json()["fields"]}

    assert kinds <= {"numeric", "categorical", "date", "text"}


def test_healthz_and_readyz_when_loaded(client: TestClient) -> None:
    health = client.get("/healthz").json()
    ready = client.get("/readyz")

    assert health["status"] == "ok"
    assert health["bundle_loaded"] is True
    assert ready.status_code == 200


def test_healthz_is_ok_and_readyz_is_503_with_no_bundle(tmp_path: Path) -> None:
    """The distinction the two probes exist for.

    A liveness probe that failed here would make an orchestrator restart a
    process that would come up in exactly this state again, forever. Readiness is
    what takes the replica out of rotation.
    """
    settings = Settings(reports_dir=tmp_path / "empty", log_level="WARNING")
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")
        predict = client.post("/predict", json={})

    assert health.status_code == 200
    assert health.json() == {
        "status": "degraded",
        "bundle_loaded": False,
        "run_id": None,
        "reason": "no active run under empty/",
    }
    assert ready.status_code == 503
    assert predict.status_code == 503
    assert "riskscore train" in predict.json()["detail"]


def test_require_bundle_refuses_to_start_without_one(tmp_path: Path) -> None:
    settings = Settings(reports_dir=tmp_path / "empty", require_bundle=True, log_level="WARNING")

    with pytest.raises(RuntimeError, match="REQUIRE_BUNDLE"):
        create_app(settings)


# --- 4. middleware and error shape ---------------------------------------------


def test_request_id_is_echoed(client: TestClient, applicant: dict[str, object]) -> None:
    response = client.post("/predict", json=applicant, headers={"X-Request-ID": "abc-123"})

    assert response.headers["x-request-id"] == "abc-123"


def test_request_id_is_generated_and_present_on_errors(client: TestClient) -> None:
    response = client.post("/predict", json={"nonsense": 1})

    assert response.status_code == 422
    assert response.headers["x-request-id"]
    # The body's id and the header's id are the same id, which is what makes a
    # user reporting "it said 422" joinable to the log line.
    assert response.json()["request_id"] == response.headers["x-request-id"]


def test_oversized_declared_body_is_refused_unread(api_settings: Settings) -> None:
    """413 from the declared length, before the body is read.

    Asserted through the *declared* length rather than by actually sending a
    megabyte: refusing on ``Content-Length`` is the behaviour worth having, since
    it costs one response and no allocation.
    """
    settings = api_settings.model_copy(update={"max_body_bytes": 64})
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.post("/predict", content=b"x" * 128)

    assert response.status_code == 413
    assert "64 bytes" in response.json()["detail"]
    assert response.headers["x-request-id"]


def test_malformed_content_length_is_400(api_settings: Settings) -> None:
    """A negative length is the input that hangs a naive ``read(length)``.

    Sent through the raw ASGI app rather than ``TestClient``, because httpx
    computes ``Content-Length`` itself and will not send a bad one - which is
    exactly why this check lives in middleware and not in a handler.
    """
    app = create_app(api_settings)
    received: list[MutableMapping[str, Any]] = []
    scope: MutableMapping[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/predict",
        "raw_path": b"/predict",
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "headers": [(b"host", b"testserver"), (b"content-length", b"-1")],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }

    async def send(message: MutableMapping[str, Any]) -> None:
        received.append(message)

    async def receive() -> MutableMapping[str, Any]:
        return {"type": "http.request", "body": b"{}", "more_body": False}

    anyio.run(lambda: app(scope, receive, send))

    start = next(message for message in received if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in received if message["type"] == "http.response.body"
    )
    assert start["status"] == 400
    assert json.loads(body)["detail"] == "Malformed Content-Length."


def test_unknown_host_is_refused(api_settings: Settings) -> None:
    """DNS rebinding: an attacker's domain resolving to 127.0.0.1."""
    with TestClient(create_app(api_settings), raise_server_exceptions=False) as client:
        response = client.get("/healthz", headers={"Host": "evil.example.com"})

    assert response.status_code == 400


def test_no_error_body_leaks_a_path(client: TestClient, applicant: dict[str, object]) -> None:
    """An error body is the one place a service volunteers information.

    Checked against the *actual* reports directory, so this fails if a handler
    starts including a resolved path in a message rather than merely if somebody
    writes the literal string ``/Users``.
    """
    reports = str(client.app.state.settings.reports_dir)  # type: ignore[attr-defined]
    bodies = [
        client.post("/predict", json={}).text,
        client.post("/predict", json={**applicant, "oops": 1}).text,
        client.get("/api/model").text,
        client.post("/predict/batch", json={"applicants": [{}]}).text,
    ]

    for body in bodies:
        assert reports not in body
        assert "Traceback" not in body


def test_error_bodies_carry_only_detail_and_request_id(client: TestClient) -> None:
    body = client.post("/predict", json={"nonsense": 1}).json()

    assert set(body) == {"detail", "request_id"}


def test_validation_summary_counts_beyond_the_reported_limit() -> None:
    errors = [{"loc": ("body", f"f{index}"), "msg": "bad"} for index in range(9)]

    summary = summarize_validation_errors(errors)

    assert summary.startswith("f0: bad; ")
    assert "f4: bad" in summary
    assert "f5" not in summary
    assert summary.endswith("and 4 more")


def test_validation_summary_never_quotes_a_value() -> None:
    """The reason this function exists rather than ``str(error)``.

    FastAPI's default 422 body includes ``input``. A 422 that quotes an
    applicant's income back at whoever sent it puts that value in every proxy log
    between them.
    """
    summary = summarize_validation_errors(
        [{"loc": ("body", "annual_inc"), "msg": "bad", "input": 987654}]
    )

    assert "987654" not in summary
    assert summary == "annual_inc: bad"


# --- 5. reports ----------------------------------------------------------------


@pytest.mark.parametrize("name", TRAIN_REPORTS)
def test_every_train_report_is_served(
    client: TestClient, trained_run: RunResult, name: str
) -> None:
    """Parametrized over the allowlist, so a new report cannot be added untested.

    The run id in the envelope is the assertion that matters beyond the 200: a
    payload that does not say which run it describes is a chart nobody can trust
    after a retrain.
    """
    response = client.get(f"/api/{name}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_id"] == trained_run.metadata.run_id
    assert set(body) - {"run_id"}, "a report with no parts is a route that does nothing"


def test_report_tables_are_columnar(client: TestClient) -> None:
    """One array per column, not one object per row.

    The encoding is the point: the threshold cost table is 99 rows of six numbers,
    and a row-per-object body repeats every key 99 times.
    """
    validation = client.get("/api/threshold-costs").json()["validation"]

    assert isinstance(validation, dict)
    assert "threshold" in validation
    lengths = {len(column) for column in validation.values()}
    assert len(lengths) == 1, "every column of one table must have the same length"
    assert lengths.pop() > 1


def test_metrics_payload_carries_the_embargo_counts(client: TestClient) -> None:
    """The project's headline correction has to be visible over HTTP.

    An embargo that only appears in a local CSV is a claim; one the dashboard can
    render is evidence.
    """
    metrics = client.get("/api/metrics").json()["metrics"]

    assert "embargo_rows_immature" in metrics
    assert "default_rate_by_vintage_before_embargo" in metrics
    assert "default_rate_by_vintage_after_embargo" in metrics


def test_report_etag_answers_304(client: TestClient) -> None:
    first = client.get("/api/metrics")
    etag = first.headers["etag"]

    second = client.get("/api/metrics", headers={"If-None-Match": etag})

    assert etag.startswith('W/"'), "gzip is applied downstream, so the validator is weak"
    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["etag"] == etag


def test_report_etag_tolerates_a_proxy_rewriting_the_validator(client: TestClient) -> None:
    """A proxy may strip ``W/`` or send a list; both still match."""
    etag = client.get("/api/metrics").headers["etag"]
    opaque = etag.removeprefix("W/")

    assert client.get("/api/metrics", headers={"If-None-Match": opaque}).status_code == 304
    assert (
        client.get("/api/metrics", headers={"If-None-Match": f'"other", {etag}'}).status_code == 304
    )


def test_report_etags_differ_per_report(client: TestClient) -> None:
    """Two reports of one run must not share a validator.

    They would if the ETag came from the run's mtime alone, and a client that had
    fetched metrics would then be told its calibration copy was current.
    """
    etags = {name: client.get(f"/api/{name}").headers["etag"] for name in TRAIN_REPORTS}

    assert len(set(etags.values())) == len(etags)


def test_report_cache_notices_a_rewritten_file(
    client: TestClient, trained_run: RunResult, tmp_path: Path
) -> None:
    """The cache is keyed on file identity, so a rewrite must change the answer.

    Written to a copy of the run rather than to the session's own directory,
    which every other test treats as read-only. This is the property a TTL cache
    cannot have: no stale window, and nothing to tune.
    """
    import shutil

    reports = tmp_path / "reports"
    shutil.copytree(trained_run.run_dir.parent.parent, reports)
    run_dir = reports / "runs" / trained_run.metadata.run_id
    settings = Settings(reports_dir=reports, log_level="WARNING")

    with TestClient(create_app(settings), raise_server_exceptions=False) as local:
        before = local.get("/api/vintages")
        table = run_dir / "metrics_by_vintage.csv"
        table.write_text(table.read_text() + table.read_text().splitlines()[-1] + "\n")
        after = local.get("/api/vintages")

    assert before.status_code == after.status_code == 200
    assert after.headers["etag"] != before.headers["etag"]
    assert (
        len(after.json()["vintages"]["partition"])
        == len(before.json()["vintages"]["partition"]) + 1
    )


def test_active_run_revalidates_and_a_named_run_is_immutable(
    client: TestClient, trained_run: RunResult
) -> None:
    """A run directory never changes; "the active run" is a pointer that moves."""
    active = client.get("/api/metrics")
    named = client.get("/api/metrics", params={"run_id": trained_run.metadata.run_id})

    assert active.headers["cache-control"] == "no-cache"
    assert "immutable" in named.headers["cache-control"]
    assert named.json() == active.json()


def test_comparison_is_404_on_a_plain_train_run(client: TestClient) -> None:
    """404, not ``{}``: "nothing was compared" is not "the models tied"."""
    response = client.get("/api/comparison")

    assert response.status_code == 404
    assert set(response.json()) == {"detail", "request_id"}


#: Stand-in for what ``riskscore compare`` writes. Fabricated rather than
#: produced by a real comparison, which needs two full fits and XGBoost - the
#: thing under test is where the endpoint *looks*, not what compare computes.
COMPARISON_JSON = {"generated_at": "2026-01-01T00:00:00Z", "variants": ["lr", "xgb"]}


def _reports_with_comparison(trained_run: RunResult, tmp_path: Path) -> Path:
    """A copy of the trained report tree with a ``comparison.json`` at its root."""
    import shutil

    reports = tmp_path / "reports"
    shutil.copytree(trained_run.run_dir.parent.parent, reports)
    (reports / "comparison.json").write_text(json.dumps(COMPARISON_JSON), encoding="utf-8")
    return reports


def test_comparison_is_read_from_the_report_root(trained_run: RunResult, tmp_path: Path) -> None:
    """``comparison.json`` lives beside ``registry.json``, not inside a run.

    It describes several runs and is only complete once all of them are published,
    so no run directory owns it - see :mod:`risk_score.reporting`. This test is
    the one that would have caught the endpoint reading it from the run directory,
    where it will never be.
    """
    reports = _reports_with_comparison(trained_run, tmp_path)
    with TestClient(
        create_app(Settings(reports_dir=reports, log_level="WARNING")),
        raise_server_exceptions=False,
    ) as local:
        response = local.get("/api/comparison")

    assert response.status_code == 200, response.text
    assert response.json()["comparison"] == COMPARISON_JSON
    # No run id on a document about several runs, and no immutable caching on a
    # file the next comparison overwrites.
    assert "run_id" not in response.json()
    assert response.headers["cache-control"] == "no-cache"


def test_comparison_ignores_a_run_id(trained_run: RunResult, tmp_path: Path) -> None:
    """``?run_id=`` is not part of this route's contract, so it cannot 404 on it."""
    reports = _reports_with_comparison(trained_run, tmp_path)
    with TestClient(
        create_app(Settings(reports_dir=reports, log_level="WARNING")),
        raise_server_exceptions=False,
    ) as local:
        response = local.get("/api/comparison", params={"run_id": "../../etc"})

    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "run_id",
    ["../../etc", "..", ".", "nope", "runs/x", "a" * 200, ""],
)
def test_report_refuses_a_bad_run_id(client: TestClient, run_id: str) -> None:
    """Traversal, absent runs and oversized ids are all one 404.

    One status for all of them on purpose: distinguishing "malformed" from
    "absent" tells a scanner which of its guesses had the right shape.
    """
    response = client.get("/api/metrics", params={"run_id": run_id})

    assert response.status_code in {404, 422}
    assert "/" not in response.json()["detail"]


def test_report_run_id_never_escapes_the_runs_directory(tmp_path: Path) -> None:
    """A symlink out of ``runs/`` is refused by the containment check.

    ``RUN_ID_PATTERN`` accepts ``"sneaky"`` - it is ordinary characters - so a 404
    here can only have come from the resolved-path check. That is the point of
    having two guards: the pattern anticipates traversal syntax, and containment
    catches what it did not anticipate.
    """
    assert RUN_ID_PATTERN.match("sneaky"), "otherwise this test proves the wrong guard"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "metrics.json").write_text("{}")
    runs = tmp_path / "reports" / "runs"
    runs.mkdir(parents=True)
    (runs / "sneaky").symlink_to(outside, target_is_directory=True)

    settings = Settings(reports_dir=tmp_path / "reports", log_level="WARNING")
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.get("/api/metrics", params={"run_id": "sneaky"})

    assert response.status_code == 404


def test_reports_are_503_with_no_active_run(tmp_path: Path) -> None:
    settings = Settings(reports_dir=tmp_path / "empty", log_level="WARNING")
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.get("/api/metrics")

    assert response.status_code == 503
    assert "riskscore train" in response.json()["detail"]


def test_report_payloads_contain_no_nan_token(client: TestClient) -> None:
    """``json.dumps`` writes a bare ``NaN`` by default, which ``JSON.parse`` rejects.

    One missing metric would then take out a whole dashboard panel, so the payload
    is serialized with ``allow_nan=False`` over already-nulled values. Asserted on
    the raw text: ``response.json()`` accepts ``NaN`` and would hide the bug.
    """
    for name in TRAIN_REPORTS:
        text = client.get(f"/api/{name}").text
        assert "NaN" not in text
        assert "Infinity" not in text


# --- 6. run history and artifacts ----------------------------------------------


def test_run_history_lists_the_active_run(client: TestClient, trained_run: RunResult) -> None:
    response = client.get("/api/runs")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["active_run_id"] == trained_run.metadata.run_id
    assert [run["run_id"] for run in body["runs"]] == [trained_run.metadata.run_id]
    entry = body["runs"][0]
    # The registry carries the headline metrics, which is what lets a history table
    # render from one file rather than from one manifest read per row.
    assert entry["metrics"]["auc_roc"] is not None
    assert entry["feature_tier"] == "origination_only"


def test_run_history_is_empty_rather_than_an_error_before_any_run(tmp_path: Path) -> None:
    """A fresh clone has no registry. That is a state, not a failure.

    503 here would make the dashboard's run selector the thing that breaks on a
    first visit, which is the visit that matters most.
    """
    settings = Settings(reports_dir=tmp_path / "empty", log_level="WARNING")
    with TestClient(create_app(settings), raise_server_exceptions=False) as local:
        response = local.get("/api/runs")

    assert response.status_code == 200
    assert response.json() == {"active_run_id": None, "runs": []}


def test_manifest_is_served_verbatim(client: TestClient, trained_run: RunResult) -> None:
    response = client.get(f"/api/runs/{trained_run.metadata.run_id}")

    assert response.status_code == 200, response.text
    on_disk = json.loads((trained_run.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert response.json() == on_disk
    # Re-serializing a manifest through a schema written later is how a recorded
    # field silently stops being reported, so byte-level equality is the assertion.
    assert response.json()["embargo"]


def test_model_card_is_served_as_markdown(client: TestClient, trained_run: RunResult) -> None:
    response = client.get(f"/api/runs/{trained_run.metadata.run_id}/card")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text == (trained_run.run_dir / "model_card.md").read_text(encoding="utf-8")
    assert "$" not in response.text, "a template placeholder survived into the card"


@pytest.mark.parametrize("name", sorted(ARTIFACT_FILENAMES))
def test_every_allowlisted_artifact_is_served(
    client: TestClient, trained_run: RunResult, name: str
) -> None:
    """The allowlist and the run directory must agree.

    An entry that no run writes is a 404 nobody notices until a dashboard panel is
    blank; a file every run writes but the list omits is unreachable. Both are
    caught here, and both were real risks while the filenames were spelled twice.
    """
    response = client.get(f"/artifacts/{trained_run.metadata.run_id}/{name}")

    assert response.status_code == 200, f"{name}: {response.text[:200]}"
    assert response.content == (trained_run.run_dir / name).read_bytes()
    assert "immutable" in response.headers["cache-control"]


def test_the_figure_is_served_as_an_image(client: TestClient, trained_run: RunResult) -> None:
    """The one allowlist entry containing a separator, which is the interesting one."""
    response = client.get(f"/artifacts/{trained_run.metadata.run_id}/figures/calibration_test.png")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    # Inline, not an attachment: the dashboard renders this in the page.
    assert "attachment" not in response.headers.get("content-disposition", "")


def test_the_model_pickle_is_not_served(client: TestClient, trained_run: RunResult) -> None:
    """The one file in a run directory that must not be reachable.

    A pickle offered over HTTP is an invitation to unpickle something a stranger
    chose. It is in the same directory as everything above, so this is the
    assertion that the allowlist is an allowlist.
    """
    assert (trained_run.run_dir / "model.joblib").is_file(), "otherwise this proves nothing"

    response = client.get(f"/artifacts/{trained_run.metadata.run_id}/model.joblib")

    assert response.status_code == 404
    assert "model.joblib" not in response.json()["detail"]


@pytest.mark.parametrize(
    "name",
    [
        "../../../etc/passwd",
        "..%2f..%2fmodel.joblib",
        "figures/../model.joblib",
        "metrics.json/../model.joblib",
        "registry.json",
        "",
        "METRICS.JSON",
    ],
)
def test_an_artifact_name_outside_the_allowlist_is_404(
    client: TestClient, trained_run: RunResult, name: str
) -> None:
    """Traversal, case games and files from the parent directory are all one 404.

    ``registry.json`` is in the list on purpose: it exists, one directory up, and
    an allowlist checked after joining rather than before would have served it.
    """
    response = client.get(f"/artifacts/{trained_run.metadata.run_id}/{name}")

    assert response.status_code in {404, 422}
    if response.headers["content-type"].startswith("application/json"):
        assert "/" not in response.json().get("detail", "")


def test_an_artifact_from_an_unknown_run_is_404(client: TestClient) -> None:
    response = client.get("/artifacts/nope/metrics.json")

    assert response.status_code == 404
    assert response.json()["detail"] == "No such run."


def test_a_manifest_from_an_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/card").status_code == 404


# --- 7. OpenAPI ----------------------------------------------------------------


def test_openapi_describes_the_loaded_model(client: TestClient, trained_run: RunResult) -> None:
    document = client.get("/openapi.json").json()

    applicant = document["components"]["schemas"]["Applicant"]
    assert set(applicant["properties"]) == set(trained_run.bundle.feature_spec.raw_inputs)
    predict = document["paths"]["/predict"]["post"]["requestBody"]["content"]
    assert predict["application/json"]["schema"] == {"$ref": "#/components/schemas/Applicant"}
    batch = document["components"]["schemas"]["BatchRequest"]["properties"]["applicants"]
    assert batch["items"] == {"$ref": "#/components/schemas/Applicant"}


def test_docs_can_be_switched_off(api_settings: Settings) -> None:
    settings = api_settings.model_copy(update={"docs_enabled": False})
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


# --- 8. the mounted dashboard ---------------------------------------------------


def test_the_dashboard_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>" in response.text


def test_the_dashboard_does_not_shadow_the_api(client: TestClient) -> None:
    """The mount is at ``/``, so it matches anything no route claimed first.

    Starlette resolves routes in order, so mounting before the routers would turn
    every endpoint into a 404 from the filesystem. This is the assertion that the
    mount goes last, and a 404 from an unknown path is what proves the mount is
    actually there to be shadowed.
    """
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/metrics").status_code == 200
    assert client.get("/no-such-page").status_code == 404


def test_a_missing_dashboard_leaves_the_api_working(trained_run: RunResult, tmp_path: Path) -> None:
    """A container shipping the API only must still boot.

    An operator who removed the dashboard wanted a scoring service, not a startup
    error naming a directory they deleted on purpose.
    """
    settings = Settings(
        reports_dir=trained_run.run_dir.parent.parent,
        dashboard_dir=tmp_path / "absent",
        log_level="WARNING",
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as local:
        assert local.get("/healthz").status_code == 200
        assert local.get("/").status_code == 404


# --- 9. settings ---------------------------------------------------------------


def test_public_bind_is_refused_without_the_flag() -> None:
    with pytest.raises(ValidationError, match="ALLOW_PUBLIC_BIND"):
        Settings(host="0.0.0.0")


def test_public_bind_with_mutating_routes_needs_a_key() -> None:
    with pytest.raises(ValidationError, match="refusing to start"):
        Settings(host="0.0.0.0", allow_public_bind=True, allow_retrain=True)


def test_mutating_routes_on_loopback_need_no_key() -> None:
    """A local demo is not a remote code execution surface."""
    settings = Settings(allow_retrain=True, allow_upload=True)

    assert settings.mutating_routes_enabled() == "both"
    assert settings.bind_is_public is False


def test_allowed_hosts_accepts_a_comma_separated_string() -> None:
    settings = Settings(allowed_hosts="localhost, example.com ")  # type: ignore[arg-type]

    assert settings.allowed_hosts == ("localhost", "example.com")


def test_unknown_log_level_is_refused() -> None:
    with pytest.raises(ValidationError, match="log_level must be one of"):
        Settings(log_level="chatty")


def test_api_key_check_is_false_when_unconfigured() -> None:
    """Fails closed: an unconfigured key must not match ``None`` or ``""``."""
    settings = Settings()

    assert settings.check_api_key(None) is False
    assert settings.check_api_key("") is False
    assert Settings(api_key="secret").check_api_key("secret") is True
    assert Settings(api_key="secret").check_api_key("Secret") is False


def test_require_api_key_distinguishes_unconfigured_from_wrong() -> None:
    """503 for a bolted door, 401 for a wrong key.

    401 with no key configured would invite a client to keep guessing at a route
    that cannot be authorized at all.
    """
    with pytest.raises(HTTPException) as unconfigured:
        require_api_key(Settings(), "anything")
    assert unconfigured.value.status_code == 503

    with pytest.raises(HTTPException) as wrong:
        require_api_key(Settings(api_key="right"), "wrong")
    assert wrong.value.status_code == 401
    assert wrong.value.headers == {"WWW-Authenticate": "X-API-Key"}

    # The success path raises nothing. Asserted by calling it, since a gate that
    # rejects a valid key is the failure nobody notices until a deploy.
    require_api_key(Settings(api_key="right"), "right")
