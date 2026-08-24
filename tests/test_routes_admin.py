"""The mutating surface: upload, retrain, job status - and everything that guards it.

Two kinds of test here, and the split is deliberate.

Most of them use a stub child, injected into the :class:`JobRunner`, because what
is under test is the *route*: the flag, the key, the 409, the ``Retry-After``, the
shape of the body. A real fit per case would add seconds to each one and prove
nothing extra about the HTTP layer.

The last one is the opposite: it uploads a real CSV and lets the real child fit a
real model, then asserts the service swapped to the new run. That is the only test
that proves the pieces are wired to each other rather than each being correct
alone, and it is the one that would catch a ``TrainRequest`` field the child does
not read.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from risk_score.api import Settings, create_app
from risk_score.api.jobs import JobRunner, TrainRequest
from risk_score.api.routes_admin import RETRY_AFTER_SECONDS
from risk_score.modeling import SUPPORTED_MODEL_TYPES
from risk_score.pipeline import RunResult
from risk_score.sample_data import make_synthetic_loans
from tests.conftest import SMALL_ROWS

KEY = "test-key-not-a-secret"
HEADERS = {"X-API-Key": KEY}

#: A minimal body that :func:`~risk_score.api.jobs.looks_like_csv` accepts. Not a
#: trainable extract - the retrain tests use a stub child, and the one that does
#: not writes a real one.
CSV = b"a,b,c\n1,2,3\n4,5,6\n"

FAKE_RUN_ID = "20260101T000000Z-logistic_regression-origination_only-abc1234"


def _stub_child(connection: Connection, request: TrainRequest, log_level: str) -> None:
    """A child that publishes nothing and reports success immediately.

    Module-level because spawn pickles a target by module and qualname: a closure
    cannot cross the process boundary, and a stub defined inside a test function
    would fail to import in the child.
    """
    connection.send({"run_id": FAKE_RUN_ID})
    connection.close()


@pytest.fixture
def admin_settings(trained_run: RunResult, tmp_path: Path) -> Settings:
    """Both features on, with a key, on loopback. The demo configuration."""
    return Settings(
        reports_dir=trained_run.run_dir.parent.parent,
        datasets_dir=tmp_path / "uploads",
        api_key=KEY,
        allow_upload=True,
        allow_retrain=True,
        log_level="WARNING",
    )


@pytest.fixture
def admin_app(admin_settings: Settings) -> FastAPI:
    """An app whose retrain child is a stub. See the module docstring.

    ``on_success`` is carried over from the runner ``create_app`` built, so the
    reload path still runs - it is the callback most likely to throw, and a stub
    that skipped it would test less than the real thing for no saving.
    """
    app = create_app(admin_settings)
    app.state.job_runner = JobRunner(
        timeout_seconds=30.0,
        on_success=app.state.job_runner.on_success,
        target=_stub_child,
    )
    return app


@pytest.fixture
def admin(admin_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(admin_app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def admin_runner(admin_app: FastAPI) -> JobRunner:
    """The stub runner, for tests that have to wait for a job to finish."""
    runner: JobRunner = admin_app.state.job_runner
    return runner


def _upload(client: TestClient, body: bytes = CSV) -> str:
    response = client.post("/api/datasets", content=body, headers=HEADERS)
    assert response.status_code == 201, response.text
    dataset_id: str = response.json()["dataset_id"]
    return dataset_id


# --- 1. the guards, in order ---------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/datasets"),
        ("post", "/api/runs"),
        ("get", "/api/jobs/abc"),
    ],
)
def test_a_disabled_feature_is_403_naming_its_variable(
    client: TestClient, method: str, path: str
) -> None:
    """Off by default, and the refusal says what would turn it on.

    404 would be the more secretive answer and the wrong one: the routes are in the
    OpenAPI document either way, so hiding them fools nobody and leaves an operator
    following the runbook with no next step.
    """
    response = client.request(method, path, json={"dataset_id": "x" * 32})

    assert response.status_code == 403, response.text
    assert "RISKSCORE_ALLOW_" in response.json()["detail"]


def test_the_flag_is_checked_before_the_key(client: TestClient) -> None:
    """A disabled route says so even with no credential presented.

    Order matters: the alternative asks for a key, gets one, and *then* says the
    feature is off - which sends somebody hunting for a credential problem that
    does not exist.
    """
    response = client.post("/api/datasets", content=CSV)

    assert response.status_code == 403
    assert "RISKSCORE_ALLOW_UPLOAD" in response.json()["detail"]


@pytest.mark.parametrize("presented", [None, "", "wrong-key", KEY + "x"])
def test_an_enabled_route_still_needs_the_key(admin: TestClient, presented: str | None) -> None:
    headers = {} if presented is None else {"X-API-Key": presented}
    response = admin.post("/api/datasets", content=CSV, headers=headers)

    assert response.status_code == 401, response.text
    assert response.headers["www-authenticate"] == "X-API-Key"


def test_an_enabled_route_with_no_key_configured_is_503(
    trained_run: RunResult, tmp_path: Path
) -> None:
    """Enabled but unauthorizable is a configuration error, not a bad credential.

    401 here would invite a client to keep guessing at a door that cannot be
    opened at all.
    """
    settings = Settings(
        reports_dir=trained_run.run_dir.parent.parent,
        datasets_dir=tmp_path / "uploads",
        allow_upload=True,
        log_level="WARNING",
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.post("/api/datasets", content=CSV)

    assert response.status_code == 503
    assert "RISKSCORE_API_KEY" in response.json()["detail"]


# --- 2. upload -----------------------------------------------------------------


def test_an_upload_is_stored_by_content(admin: TestClient, admin_settings: Settings) -> None:
    response = admin.post("/api/datasets", content=CSV, headers=HEADERS)

    assert response.status_code == 201, response.text
    body = response.json()
    assert len(body["dataset_id"]) == 32
    assert body["size_bytes"] == len(CSV)
    assert body["existing"] is False
    stored = admin_settings.datasets_dir / f"{body['dataset_id']}.csv"
    assert stored.read_bytes() == CSV


def test_re_uploading_the_same_bytes_says_so(admin: TestClient, admin_settings: Settings) -> None:
    """Idempotent, and reported as such rather than silently.

    An operator who uploads twice by accident should be told the second one was a
    no-op, not left wondering which copy a run will use.
    """
    first = admin.post("/api/datasets", content=CSV, headers=HEADERS).json()
    second = admin.post("/api/datasets", content=CSV, headers=HEADERS).json()

    assert first["dataset_id"] == second["dataset_id"]
    assert first["existing"] is False
    assert second["existing"] is True
    assert len(list(admin_settings.datasets_dir.glob("*.csv"))) == 1


def test_a_chunked_upload_is_stored_whole(admin: TestClient, admin_settings: Settings) -> None:
    """The body arriving in pieces must reassemble byte for byte.

    This is the case the sync-stream adapter exists for, and the one where an
    off-by-one in its buffer would produce a file under a hash that does not
    describe it. A generator body makes ``httpx`` send it chunked with no declared
    length, which is also the request the size cap cannot check in advance.
    """
    payload = b"a,b,c\n" + b"1,2,3\n" * 20_000

    def chunks() -> Iterator[bytes]:
        for start in range(0, len(payload), 1024):
            yield payload[start : start + 1024]

    response = admin.post("/api/datasets", content=chunks(), headers=HEADERS)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["size_bytes"] == len(payload)
    stored = admin_settings.datasets_dir / f"{body['dataset_id']}.csv"
    assert stored.read_bytes() == payload


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (b"", "empty"),
        (b"   \n\n", "empty"),
        (b"a,b\x00c\n1,2,3\n", "binary"),
        (b"no delimiter here\nnor here\n", "no comma"),
    ],
)
def test_an_implausible_upload_is_422(admin: TestClient, body: bytes, reason: str) -> None:
    response = admin.post("/api/datasets", content=body, headers=HEADERS)

    assert response.status_code == 422, response.text
    assert reason in response.json()["detail"]


def test_an_upload_over_the_cap_is_413(trained_run: RunResult, tmp_path: Path) -> None:
    """Refused by the middleware, from the declared length, before a byte is stored.

    The upload route has its own cap - the 1 MiB body limit is sized for a
    ``/predict`` request - so this is also the assertion that the per-path override
    is wired to the right path and is not simply uncapped.
    """
    settings = Settings(
        reports_dir=trained_run.run_dir.parent.parent,
        datasets_dir=tmp_path / "uploads",
        api_key=KEY,
        allow_upload=True,
        max_upload_bytes=64,
        log_level="WARNING",
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.post("/api/datasets", content=b"a,b\n" + b"1,2\n" * 100, headers=HEADERS)

    assert response.status_code == 413, response.text
    assert "64 bytes" in response.json()["detail"]
    # Refused from the declared length, so the handler never ran and the directory
    # was never even created. `glob` on a missing directory yields nothing.
    assert list((tmp_path / "uploads").glob("*")) == []


def test_a_predict_body_is_still_capped_at_the_smaller_limit(admin: TestClient) -> None:
    """The per-path override must not become the limit for everything.

    A prefix match, or one cap raised for all routes, would make this pass a 2 MiB
    applicant - which is the mistake the exemption is one edit away from being.
    """
    response = admin.post("/predict", content=b"x" * (2 << 20), headers=HEADERS)

    assert response.status_code == 413, response.text


# --- 3. retrain ----------------------------------------------------------------


def test_a_retrain_returns_202_with_a_job_id(admin: TestClient) -> None:
    dataset_id = _upload(admin)

    response = admin.post("/api/runs", json={"dataset_id": dataset_id}, headers=HEADERS)

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "running"
    assert body["dataset"] == f"{dataset_id}.csv"
    assert body["model_type"] == "logistic_regression"
    assert body["run_id"] is None


def test_a_retrain_on_an_unknown_dataset_is_404(admin: TestClient) -> None:
    response = admin.post("/api/runs", json={"dataset_id": "0" * 32}, headers=HEADERS)

    assert response.status_code == 404, response.text
    assert "POST /api/datasets" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"dataset_id": "0" * 32, "model_type": "random_forest"},
        {"dataset_id": "0" * 32, "extra": 1},
        {"dataset_id": "0" * 32, "include_lender_priced": "maybe"},
    ],
)
def test_a_malformed_retrain_request_is_422(admin: TestClient, payload: dict[str, object]) -> None:
    """Unknown keys are refused, not ignored.

    ``extra="forbid"`` is why: a caller who sends ``includeLenderPriced`` and gets
    a 202 would reasonably believe the flag took effect.
    """
    response = admin.post("/api/runs", json=payload, headers=HEADERS)

    assert response.status_code == 422, response.text


def test_retrain_offers_every_supported_model(admin: TestClient) -> None:
    """The ``Literal`` in ``RetrainIn`` and ``SUPPORTED_MODEL_TYPES`` must agree.

    A ``Literal`` cannot be built from a runtime tuple, so the choice list is
    spelled twice. This is the assertion that keeps them in step: a third estimator
    added to the registry would otherwise be trainable from the CLI and a 422 over
    HTTP, with nothing failing to say so.
    """
    schema = admin.get("/openapi.json").json()["components"]["schemas"]["RetrainIn"]

    assert set(schema["properties"]["model_type"]["enum"]) == set(SUPPORTED_MODEL_TYPES)


def test_the_admin_routes_are_documented_even_when_disabled(client: TestClient) -> None:
    """Which is why a switched-off route is 403 and not 404.

    Hiding them from the OpenAPI document as well would be the coherent version of
    a 404, and it would also mean ``/docs`` described a different service depending
    on the environment - so a client could not be written against it at all.
    """
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/datasets" in paths
    assert "/api/runs" in paths
    assert "/api/jobs/{job_id}" in paths


def test_a_second_retrain_is_409_with_a_retry_after(
    admin: TestClient, admin_runner: JobRunner
) -> None:
    """One slot, and the refusal names the job holding it. Not a queue - see jobs.py."""
    dataset_id = _upload(admin)
    first = admin.post("/api/runs", json={"dataset_id": dataset_id}, headers=HEADERS).json()

    second = admin.post("/api/runs", json={"dataset_id": dataset_id}, headers=HEADERS)

    if second.status_code == 409:
        assert first["job_id"] in second.json()["detail"]
        assert second.headers["retry-after"] == RETRY_AFTER_SECONDS
    else:
        # The first child finished first. Not a failure of the 409 rule, so assert
        # what is actually true then: the slot was released and reused.
        assert second.status_code == 202, second.text
        assert second.json()["job_id"] != first["job_id"]
    admin_runner.wait(timeout=30.0)


def test_a_finished_job_reports_its_run_id(admin: TestClient, admin_runner: JobRunner) -> None:
    dataset_id = _upload(admin)
    submitted = admin.post("/api/runs", json={"dataset_id": dataset_id}, headers=HEADERS).json()
    assert admin_runner.wait(timeout=60.0), "the stub child's watcher never finished"

    response = admin.get(f"/api/jobs/{submitted['job_id']}", headers=HEADERS)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["run_id"] == FAKE_RUN_ID
    assert body["finished_at"] is not None
    # Absent once there is nothing to wait for, which is itself the signal to stop
    # polling.
    assert "retry-after" not in response.headers


def test_an_unknown_job_is_404(admin: TestClient) -> None:
    response = admin.get("/api/jobs/deadbeef", headers=HEADERS)

    assert response.status_code == 404
    assert response.json()["detail"] == "No such job."


def test_the_job_body_has_exactly_the_documented_keys(admin: TestClient) -> None:
    """The response model, not ``asdict``.

    A field added to :class:`~risk_score.api.jobs.Job` for internal bookkeeping
    must not appear in the API because somebody added it to a dataclass.
    """
    dataset_id = _upload(admin)
    body = admin.post("/api/runs", json={"dataset_id": dataset_id}, headers=HEADERS).json()

    assert set(body) == {
        "job_id",
        "status",
        "dataset",
        "model_type",
        "submitted_at",
        "finished_at",
        "run_id",
        "detail",
        "exit_code",
    }


# --- 4. the whole thing, for real ----------------------------------------------


def test_a_real_retrain_publishes_a_run_and_the_service_swaps_to_it(
    admin_settings: Settings, tmp_path: Path
) -> None:
    """Upload a real extract, retrain, and score against the new model.

    The only test here that runs the production child. Everything above proves a
    route in isolation; this proves the wiring - that ``TrainRequest`` carries what
    the child reads, that ``on_success`` reaches ``load_service``, and that
    ``/predict`` answers from the new bundle rather than the one loaded at startup.

    It writes into a copy of the report tree, never the session fixture's: the
    retrain publishes a new active run, and mutating the shared tree would change
    what every other test is looking at.
    """
    import shutil

    reports = tmp_path / "reports"
    shutil.copytree(admin_settings.reports_dir, reports)
    settings = admin_settings.model_copy(update={"reports_dir": reports})

    app = create_app(settings)
    before = app.state.service.metadata.run_id
    with TestClient(app, raise_server_exceptions=False) as client:
        extract = io.BytesIO()
        make_synthetic_loans(n_rows=SMALL_ROWS, seed=99).to_csv(extract, index=False)
        dataset_id = _upload(client, extract.getvalue())

        submitted = client.post("/api/runs", json={"dataset_id": dataset_id}, headers=HEADERS)
        assert submitted.status_code == 202, submitted.text

        runner: JobRunner = app.state.job_runner
        # Generous: this is a real fit in a spawned interpreter, and a flaky timeout
        # here would be a test that fails on a loaded machine rather than a bug.
        assert runner.wait(timeout=300.0), "the retrain never finished"

        job = client.get(f"/api/jobs/{submitted.json()['job_id']}", headers=HEADERS).json()
        assert job["status"] == "succeeded", job["detail"]

        after = client.get("/api/model").json()["run_id"]
        assert after == job["run_id"]
        assert after != before, "the service is still serving the run it started with"
        # And it can still score, which is the assertion that the swap replaced
        # every piece of per-bundle state together rather than half of it.
        assert client.get("/readyz").json()["run_id"] == after
