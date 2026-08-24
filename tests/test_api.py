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
from risk_score.pipeline import RunResult

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


# --- 5. OpenAPI ----------------------------------------------------------------


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


# --- 6. settings ---------------------------------------------------------------


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
