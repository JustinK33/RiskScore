"""The retrain runner and the dataset store, without HTTP in the way.

The runner is tested against stub children rather than real fits. Every case
worth asserting here - the single-flight refusal, a child that raises, a child
the kernel kills, a child that never finishes - is about what the *parent* does,
and paying ten seconds of scikit-learn per case to find out would mean these
paths get tested once and then never again.

The stub children are module-level functions because spawn pickles a target by
module and qualname; a closure or a local function cannot cross the process
boundary. They also take the real signature, so a change to it breaks them.
"""

from __future__ import annotations

import io
import os
import time
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from risk_score.api.jobs import (
    HISTORY,
    DatasetRef,
    Job,
    JobBusy,
    JobRunner,
    TrainRequest,
    UploadRejected,
    _child,
    looks_like_csv,
    resolve_dataset,
    store_dataset,
)

#: Long enough that a child cannot outrun the assertion, short enough that the
#: timeout test does not dominate the suite.
STUB_TIMEOUT = 5.0

#: Every stub child finishes in milliseconds, so waiting this long means the
#: watcher thread is wedged rather than busy.
WATCH_TIMEOUT = 30.0

FAKE_RUN_ID = "20260101T000000Z-logistic_regression-origination_only-abc1234"

#: The exit code `_dying_child` uses. Arbitrary and non-zero; asserted exactly,
#: because "the job recorded some failure" would pass even if the runner threw
#: the code away, which is the field an operator needs to tell an OOM kill from a
#: raised exception.
DEATH_CODE = 9


# --- stub children -------------------------------------------------------------


def _publishing_child(connection: Connection, request: TrainRequest, log_level: str) -> None:
    """Reports a run id and exits, as a successful fit does."""
    connection.send({"run_id": FAKE_RUN_ID})
    connection.close()


def _failing_child(connection: Connection, request: TrainRequest, log_level: str) -> None:
    """Reports an error string, as `_child`'s own except block does."""
    connection.send({"error": "ValueError: the target column is constant"})
    connection.close()


def _dying_child(connection: Connection, request: TrainRequest, log_level: str) -> None:
    """Exits without reporting anything. This is what an OOM kill looks like."""
    os._exit(DEATH_CODE)


def _hanging_child(connection: Connection, request: TrainRequest, log_level: str) -> None:
    """Never finishes, so the parent has to enforce the timeout."""
    time.sleep(600)


def _request(tmp_path: Path) -> TrainRequest:
    return TrainRequest(
        dataset=tmp_path / "loans.csv",
        output_dir=tmp_path / "reports",
        model_type="logistic_regression",
    )


def _runner(target: object, **kwargs: object) -> JobRunner:
    return JobRunner(
        timeout_seconds=kwargs.pop("timeout_seconds", STUB_TIMEOUT),  # type: ignore[arg-type]
        target=target,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# --- 1. the runner -------------------------------------------------------------


def test_a_successful_child_publishes_its_run_and_fires_the_callback(tmp_path: Path) -> None:
    activated: list[str] = []
    runner = _runner(_publishing_child, on_success=activated.append)

    job = runner.submit(_request(tmp_path))
    assert job.status == "running"
    assert runner.wait(WATCH_TIMEOUT), "the watcher thread did not finish"

    finished = runner.get(job.job_id)
    assert finished is not None
    assert finished.status == "succeeded"
    assert finished.run_id == FAKE_RUN_ID
    assert finished.detail is None
    assert finished.finished_at is not None
    # The callback is what swaps the served bundle. A job that says "succeeded"
    # while /predict still answers from the previous model is the failure this
    # asserts against.
    assert activated == [FAKE_RUN_ID]


def test_the_slot_is_released_once_the_child_finishes(tmp_path: Path) -> None:
    runner = _runner(_publishing_child)
    runner.submit(_request(tmp_path))
    assert runner.wait(WATCH_TIMEOUT)
    assert runner.active is None
    # And a second retrain is accepted, which is the observable half of it.
    runner.submit(_request(tmp_path))
    assert runner.wait(WATCH_TIMEOUT)


def test_a_second_submit_while_one_runs_is_refused(tmp_path: Path) -> None:
    runner = _runner(_hanging_child)
    first = runner.submit(_request(tmp_path))
    try:
        with pytest.raises(JobBusy) as raised:
            runner.submit(_request(tmp_path))
        # The running job's id, so the caller of the 409 can poll the job that is
        # in its way rather than guess.
        assert raised.value.job_id == first.job_id
        assert first.job_id in str(raised.value)
    finally:
        runner.timeout_seconds = 0.0
        runner.wait(WATCH_TIMEOUT)


def test_a_raising_child_is_failed_with_its_reason(tmp_path: Path) -> None:
    activated: list[str] = []
    runner = _runner(_failing_child, on_success=activated.append)

    job = runner.submit(_request(tmp_path))
    assert runner.wait(WATCH_TIMEOUT)

    finished = runner.get(job.job_id)
    assert finished is not None
    assert finished.status == "failed"
    assert finished.detail == "ValueError: the target column is constant"
    assert finished.run_id is None
    # Nothing published, so nothing swapped.
    assert activated == []


def test_a_killed_child_is_failed_with_its_exit_code(tmp_path: Path) -> None:
    """The OOM case: no message on the pipe and the process is simply gone.

    Distinguished from a raised exception by ``exit_code``, because the two have
    completely different fixes - one is a bug in the pipeline, the other is a
    machine too small for the extract.
    """
    runner = _runner(_dying_child)
    job = runner.submit(_request(tmp_path))
    assert runner.wait(WATCH_TIMEOUT)

    finished = runner.get(job.job_id)
    assert finished is not None
    assert finished.status == "failed"
    assert finished.exit_code == DEATH_CODE
    assert "without reporting" in (finished.detail or "")


def test_a_hanging_child_is_killed_and_the_slot_released(tmp_path: Path) -> None:
    runner = _runner(_hanging_child, timeout_seconds=0.5)
    job = runner.submit(_request(tmp_path))
    assert runner.wait(WATCH_TIMEOUT)

    finished = runner.get(job.job_id)
    assert finished is not None
    assert finished.status == "timed_out"
    assert "Exceeded" in (finished.detail or "")
    # The whole point of the timeout: the single slot is usable again.
    assert runner.active is None


def test_a_failing_callback_does_not_fail_the_job(tmp_path: Path) -> None:
    """A reload failure is ``/readyz``'s problem, not the job's.

    The run is on disk either way, and a job reported as failed would invite an
    operator to retrain a model that trained perfectly well.
    """

    def explode(run_id: str) -> None:
        raise RuntimeError("could not unpickle the new bundle")

    runner = _runner(_publishing_child, on_success=explode)
    job = runner.submit(_request(tmp_path))
    assert runner.wait(WATCH_TIMEOUT)

    finished = runner.get(job.job_id)
    assert finished is not None
    assert finished.status == "succeeded"


def test_history_is_newest_first(tmp_path: Path) -> None:
    runner = _runner(_publishing_child)
    first = runner.submit(_request(tmp_path))
    assert runner.wait(WATCH_TIMEOUT)
    second = runner.submit(_request(tmp_path))
    assert runner.wait(WATCH_TIMEOUT)

    assert [job.job_id for job in runner.recent()] == [second.job_id, first.job_id]


def test_history_is_bounded_and_drops_the_oldest(tmp_path: Path) -> None:
    """The bound, without paying for ``HISTORY + 3`` interpreter startups.

    Reaching into ``_remember`` rather than submitting 35 jobs: the eviction rule
    is arithmetic on an ``OrderedDict``, and spawning a process 35 times to observe
    it would make this the slowest test in the suite by an order of magnitude.
    """
    runner = _runner(_publishing_child)
    jobs = [
        Job(
            job_id=f"job{index:03d}",
            status="succeeded",
            dataset="loans.csv",
            model_type="logistic_regression",
            submitted_at=f"2026-01-01T00:00:{index:02d}Z",
        )
        for index in range(HISTORY + 3)
    ]
    for job in jobs:
        runner._remember(job)

    remembered = [job.job_id for job in runner.recent()]
    assert len(remembered) == HISTORY
    assert remembered[0] == jobs[-1].job_id, "newest first"
    assert jobs[0].job_id not in remembered, "the oldest is evicted"


def test_an_unknown_job_id_is_none() -> None:
    runner = _runner(_publishing_child)
    assert runner.get("nope") is None
    assert runner.recent() == []
    # Nothing has been submitted, so there is no watcher to wait for.
    assert runner.wait(0.0)


def test_job_status_serializes_to_plain_json_types(tmp_path: Path) -> None:
    runner = _runner(_publishing_child)
    job = runner.submit(_request(tmp_path))
    assert runner.wait(WATCH_TIMEOUT)

    finished = runner.get(job.job_id)
    assert finished is not None
    payload = finished.as_dict()
    assert set(payload) == {
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
    # The dataset is reported by name, never by path: a job status is the one
    # response that would otherwise leak the server's directory layout.
    assert payload["dataset"] == "loans.csv"
    assert "/" not in str(payload["dataset"])


# --- 2. sniffing an upload -----------------------------------------------------


def test_a_plausible_csv_header_is_accepted() -> None:
    looks_like_csv(b"loan_amnt,term,purpose\n15000, 36 months,debt_consolidation\n")


@pytest.mark.parametrize(
    ("head", "because"),
    [
        (b"", "empty"),
        (b"   \n\n", "whitespace only"),
        (b"\x80\x81\x82,a,b\n", "not UTF-8"),
        (b"\x80\x93 an em dash in latin-1", "not UTF-8"),
        (b"loan_amnt\tterm\tpurpose\n", "tab separated, no comma"),
        (b"just a sentence\n", "no delimiter at all"),
    ],
)
def test_an_implausible_upload_is_refused(head: bytes, because: str) -> None:
    with pytest.raises(UploadRejected):
        looks_like_csv(head)


def test_a_pickle_is_refused_as_binary() -> None:
    """The one that matters: a joblib bundle handed to ``pd.read_csv``.

    Refused on the NUL byte rather than on a magic-number list, so parquet, zip
    and every other binary format is covered by the same check.
    """
    import pickle

    with pytest.raises(UploadRejected, match="binary"):
        looks_like_csv(pickle.dumps({"a": 1}))


# --- 3. storing an upload ------------------------------------------------------

CSV = b"loan_amnt,term\n15000, 36 months\n20000, 36 months\n"


def test_a_stored_dataset_is_named_by_its_content(tmp_path: Path) -> None:
    first = store_dataset(io.BytesIO(CSV), tmp_path, max_bytes=1 << 20)
    second = store_dataset(io.BytesIO(CSV), tmp_path, max_bytes=1 << 20)

    assert isinstance(first, DatasetRef)
    assert first.dataset_id == second.dataset_id, "the same bytes must get the same id"
    assert first.path == second.path
    assert first.size_bytes == len(CSV)
    assert first.path.read_bytes() == CSV
    # One file, not two: an id that is a content address cannot collide with a
    # different dataset, so re-uploading is idempotent rather than duplicative.
    assert sorted(path.name for path in tmp_path.iterdir()) == [first.path.name]


def test_different_content_gets_a_different_id(tmp_path: Path) -> None:
    first = store_dataset(io.BytesIO(CSV), tmp_path, max_bytes=1 << 20)
    second = store_dataset(io.BytesIO(CSV + b"30000, 36 months\n"), tmp_path, max_bytes=1 << 20)
    assert first.dataset_id != second.dataset_id


def test_the_directory_is_created_on_demand(tmp_path: Path) -> None:
    target = tmp_path / "does" / "not" / "exist"
    reference = store_dataset(io.BytesIO(CSV), target, max_bytes=1 << 20)
    assert reference.path.is_file()


def test_an_oversized_upload_is_refused_and_leaves_nothing_behind(tmp_path: Path) -> None:
    with pytest.raises(UploadRejected, match="exceeds"):
        store_dataset(io.BytesIO(b"a,b\n" + b"x" * 4096), tmp_path, max_bytes=64)
    # No `.part` file: a refused upload that leaves its bytes on disk is a way to
    # fill a disk with requests that were all answered with an error.
    assert list(tmp_path.iterdir()) == []


def test_a_non_csv_upload_is_refused_after_streaming_and_leaves_nothing(tmp_path: Path) -> None:
    with pytest.raises(UploadRejected):
        store_dataset(io.BytesIO(b"\x00\x01\x02binary"), tmp_path, max_bytes=1 << 20)
    assert list(tmp_path.iterdir()) == []


def test_a_stream_that_breaks_midway_leaves_nothing(tmp_path: Path) -> None:
    """A dropped connection must not leave a file under a hash of partial bytes."""

    # A plain class, not an `io` subclass: `store_dataset` declares the `Reader`
    # protocol, which is one `read` method, precisely so the real caller - an ASGI
    # body adapter - does not have to pretend to be a file.
    class Breaking:
        def read(self, size: int = -1) -> bytes:
            raise OSError("connection reset")

    with pytest.raises(OSError, match="connection reset"):
        store_dataset(Breaking(), tmp_path, max_bytes=1 << 20)
    assert list(tmp_path.iterdir()) == []


def test_a_size_cap_is_enforced_across_chunks(tmp_path: Path) -> None:
    """The cap counts what arrived, not what was declared.

    A stream handing back small chunks is the shape a chunked upload takes, and a
    cap checked only against ``Content-Length`` would accept all of it.
    """

    class Dribbling:
        def __init__(self) -> None:
            self.sent = 0

        def read(self, size: int = -1) -> bytes:
            self.sent += 8
            return b"a,b,c,d\n"

    with pytest.raises(UploadRejected, match="exceeds"):
        store_dataset(Dribbling(), tmp_path, max_bytes=100)
    assert list(tmp_path.iterdir()) == []


# --- 4. resolving a dataset id -------------------------------------------------


def test_a_stored_dataset_resolves_by_id(tmp_path: Path) -> None:
    reference = store_dataset(io.BytesIO(CSV), tmp_path, max_bytes=1 << 20)
    assert resolve_dataset(reference.dataset_id, tmp_path) == reference.path.resolve()


@pytest.mark.parametrize(
    "dataset_id",
    [
        "",
        "..",
        "../../etc/passwd",
        "loans",
        "0" * 31,
        "0" * 33,
        "0" * 31 + "G",
        "0" * 31 + "A",
        "0123456789abcdef0123456789abcde/",
    ],
)
def test_a_bad_dataset_id_is_a_key_error(dataset_id: str, tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        resolve_dataset(dataset_id, tmp_path)


def test_an_absent_dataset_is_a_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        resolve_dataset("0" * 32, tmp_path)


def test_a_symlinked_dataset_never_escapes_the_directory(tmp_path: Path) -> None:
    """Second guard: the pattern cannot see a symlink, so containment is checked.

    The id here is perfectly well formed - that is the point. Only the resolved
    parent check refuses it.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.csv"
    secret.write_bytes(CSV)

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    dataset_id = "0" * 32
    (uploads / f"{dataset_id}.csv").symlink_to(secret)

    with pytest.raises(KeyError):
        resolve_dataset(dataset_id, uploads)


# --- 3. the real child ---------------------------------------------------------


def _run_child(request: TrainRequest) -> dict[str, object]:
    """Call `_child` in this process and return whatever it sent back.

    In-process, which the stubs above deliberately are not. Those test the parent;
    this tests the child, and the two halves of the runner are only ever connected
    through the pipe. Running it here rather than spawning means coverage sees it
    - a spawned child's lines are invisible to coverage.py without subprocess
    instrumentation - and it costs nothing, because `_child` closes the connection
    itself in a `finally`.
    """
    parent, child = Pipe(duplex=False)
    try:
        _child(child, request, "WARNING")
        assert parent.poll(), "the child sent nothing before closing the pipe"
        received: dict[str, object] = parent.recv()
        return received
    finally:
        parent.close()


def test_the_real_child_reports_the_run_it_published(raw_csv: Path, tmp_path: Path) -> None:
    """A genuine fit through `_child`, in this process.

    `tests/test_routes_admin.py::test_a_real_retrain_publishes_a_run_and_the_service_swaps_to_it`
    already runs this code end to end, but it runs it *spawned*, where coverage.py
    cannot see it without subprocess instrumentation - so the child read as
    untested while being the most consequential twenty lines in the module. Here
    the same call is made directly, which also lets the assertions be about what
    came back through the pipe rather than about the service that ends up loaded.
    """
    output_dir = tmp_path / "reports"
    result = _run_child(
        TrainRequest(
            dataset=raw_csv,
            output_dir=output_dir,
            model_type="logistic_regression",
            cache_dir=tmp_path / "cache",
        )
    )

    run_id = result["run_id"]
    assert isinstance(run_id, str)
    assert "logistic_regression" in run_id and "origination_only" in run_id
    assert (output_dir / "runs" / run_id / "model.joblib").is_file()
    assert "error" not in result


def test_the_real_child_sends_the_failure_back_instead_of_raising(tmp_path: Path) -> None:
    """A child that raises must answer the pipe, not just die.

    This is the path an operator actually meets: the parent has no traceback, only
    what came back through the pipe, so an exception that escaped `_child` would
    surface as a bare exit code and the whole point of a job status - saying why -
    would be lost.
    """
    result = _run_child(_request(tmp_path))

    assert "run_id" not in result
    assert str(result["error"]).startswith("FileNotFoundError:")
