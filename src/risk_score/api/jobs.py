"""Running a retrain off the request thread, and storing the data it runs on.

The dashboard this replaces retrained the model *on the request thread*: an
unauthenticated POST fitted a model on an uploaded CSV and overwrote the
canonical report tree, and every ``/predict`` in the meantime waited behind it.
This module is the answer to all three parts of that.

**A process, not a thread**, for three independent reasons. A fit holds the GIL
for minutes, so a thread would stall the event loop and every score with it.
matplotlib's pyplot carries global state that is not thread-safe, and the run
writes figures. And a process can be killed, which is the only way to enforce a
timeout on scikit-learn code that does not check for cancellation.

**Spawn, not fork.** The parent has an event loop, a thread pool and an open
pickle; forking that copies every one of them into a child that is about to spend
ten minutes in BLAS. Spawn costs about a second of interpreter startup, which is
nothing against a fit, and it is the default on macOS anyway - so fork would only
ever be the *untested* path.

**One at a time**, and a second request is a 409 rather than a queue. Two
concurrent fits on the 1.19 GB extract is an out-of-memory kill, and a queue
would accept work it cannot promise to do; a 409 tells the caller the truth
immediately.

Throughout, ``/predict`` keeps serving the bundle it already has. The new run is
staged under a temporary name and published by one ``os.replace``, so no request
ever sees a half-built model - and the service only swaps to it after the child
exits successfully.

Not reported here: per-stage progress. A child process cannot write into the
parent's memory, so a "now fitting the calibrator" string needs an IPC channel -
a ``multiprocessing.Manager`` proxy, which is a third process - for a status
line. ponytail: status, timing and the error, and the elapsed seconds a poller
can already compute. Upgrade path if it is ever wanted: pass a manager ``dict``
to :func:`_child` and have the run's stage callbacks write to it.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import re
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from risk_score.artifacts import now_iso

_log = logging.getLogger(__name__)

JobStatus = Literal["running", "succeeded", "failed", "timed_out"]

#: How many finished jobs to remember. The run *registry* is the durable record
#: of what was trained; this is only so a poller that asks about the job it just
#: submitted gets an answer after the process is gone.
HISTORY: Final = 32

#: How long to wait for a terminated child to actually die before giving up on it.
#: A fit stuck in a BLAS call ignores SIGTERM until it returns; after this the
#: slot is released anyway, because holding it forever is the failure mode the
#: timeout exists to prevent.
TERMINATE_GRACE_SECONDS: Final = 10.0


class JobBusy(RuntimeError):
    """A retrain is already running. Surfaces as a 409."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"A retrain is already running (job {job_id}).")
        self.job_id = job_id


@dataclass
class Job:
    """One retrain, as a poller sees it.

    Mutable, unlike almost everything else in this project, because it *is* the
    status: the watcher thread updates it in place and the route reads it. Guarded
    by the runner's lock.
    """

    job_id: str
    status: JobStatus
    dataset: str
    model_type: str
    submitted_at: str
    finished_at: str | None = None
    run_id: str | None = None
    detail: str | None = None
    #: Set only on a timeout or a killed child, so an operator can tell "the fit
    #: raised" from "the kernel killed it", which have completely different fixes.
    exit_code: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainRequest:
    """What the child is asked to do, as picklable plain data.

    Deliberately not a ``RunConfig``: that object is reachable from the estimator
    registry and pickling it would send far more across the process boundary than
    four scalars. The child builds the config itself, from the same defaults the
    CLI uses.
    """

    dataset: Path
    output_dir: Path
    model_type: str
    include_lender_priced: bool = False
    cache_dir: Path | None = None
    keep_runs: int = 20


def _child(connection: Connection, request: TrainRequest, log_level: str) -> None:
    """The whole of what runs in the spawned process.

    Module-level and importable, which spawn requires - a closure or a bound
    method cannot be pickled to the child.

    Every exception is caught and sent back as a string. An uncaught one would
    leave the parent with an exit code and no reason, and "why did the retrain
    fail" is the entire question a job status exists to answer. The payload is a
    few hundred bytes, so it cannot fill the pipe buffer and deadlock against the
    parent's ``join``.
    """
    from risk_score.config import RunConfig
    from risk_score.logging_setup import configure_logging
    from risk_score.pipeline import train_run

    configure_logging(level=log_level)
    try:
        result = train_run(
            request.dataset,
            config=RunConfig(include_lender_priced=request.include_lender_priced),
            output_dir=request.output_dir,
            model_type=request.model_type,
            cache_dir=request.cache_dir,
            keep_runs=request.keep_runs,
        )
        connection.send({"run_id": result.run_id})
    # Broad on purpose: an uncaught exception would leave the parent with an
    # exit code and no reason, and "why did the retrain fail" is the entire
    # question a job status exists to answer.
    except Exception as error:
        _log.exception("retrain failed")
        connection.send({"error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


#: What a child process must look like. Only ``_child`` implements it in
#: production; the alias exists so the injection point in :class:`JobRunner` is
#: typed rather than ``Any``.
ChildTarget = Callable[[Connection, "TrainRequest", str], None]


class JobRunner:
    """One retrain slot, its history, and the thread that watches the child.

    Constructed once per app and held on ``app.state``. Not a module global: two
    apps in one test process would otherwise share a slot, and a test that
    submitted a job would make an unrelated one return 409.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        log_level: str = "INFO",
        on_success: Callable[[str], None] | None = None,
        target: ChildTarget = _child,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.log_level = log_level
        # What the child runs. Injected only so the runner's own behaviour - the
        # timeout, the kill, a child that dies without reporting - can be tested
        # without a ten-second model fit per case. Production never passes it.
        self.target = target
        # Called in the watcher thread once a run is published, to reload the
        # bundle and drop the report cache. Injected rather than imported so this
        # module knows nothing about FastAPI.
        self.on_success = on_success
        self._lock = threading.Lock()
        self._active: Job | None = None
        self._history: OrderedDict[str, Job] = OrderedDict()
        # Spawn explicitly rather than relying on the platform default: fork is
        # still the default on Linux, and forking a process that holds an event
        # loop and a thread pool is the bug this avoids.
        self._context = mp.get_context("spawn")
        self._watcher: threading.Thread | None = None

    @property
    def active(self) -> Job | None:
        with self._lock:
            return self._active

    def submit(self, request: TrainRequest) -> Job:
        """Start a retrain, or refuse because one is running.

        The slot is claimed under the lock *before* the process is started, so two
        simultaneous POSTs cannot both find it free.
        """
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            status="running",
            dataset=request.dataset.name,
            model_type=request.model_type,
            submitted_at=now_iso(),
        )
        with self._lock:
            if self._active is not None:
                raise JobBusy(self._active.job_id)
            self._active = job
            self._remember(job)

        parent, child = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=self.target,
            args=(child, request, self.log_level),
            name=f"riskscore-retrain-{job.job_id}",
            daemon=True,
        )
        process.start()
        # Closed in the parent immediately: while any copy of the write end is
        # open, a read on the pipe blocks instead of reporting EOF, so a child
        # that died without sending would hang the watcher rather than be noticed.
        child.close()

        self._watcher = threading.Thread(
            target=self._watch,
            args=(job, process, parent),
            name=f"riskscore-watch-{job.job_id}",
            daemon=True,
        )
        self._watcher.start()
        _log.info("retrain job %s started (pid %s)", job.job_id, process.pid)
        return job

    def _watch(self, job: Job, process: Any, parent: Connection) -> None:
        """Wait for the child, enforce the timeout, record what happened.

        A thread rather than an asyncio task: ``Process.join`` is blocking, and the
        alternative is polling ``is_alive()`` on the event loop, which is a busy
        wait that also delays the answer by up to its interval.
        """
        try:
            process.join(self.timeout_seconds)
            if process.is_alive():
                _log.error(
                    "retrain job %s exceeded %.0fs; killing", job.job_id, self.timeout_seconds
                )
                process.terminate()
                process.join(TERMINATE_GRACE_SECONDS)
                if process.is_alive():
                    # SIGKILL, because a fit inside a BLAS call does not return to
                    # check for SIGTERM and the slot has to be released.
                    process.kill()
                    process.join(TERMINATE_GRACE_SECONDS)
                self._finish(job, "timed_out", detail=f"Exceeded {self.timeout_seconds:.0f}s.")
                return

            payload = self._receive(parent)
            if payload is None:
                # No message and the process is gone: it did not reach its own
                # except block. An OOM kill looks exactly like this.
                self._finish(
                    job,
                    "failed",
                    detail="The retrain process exited without reporting. It was probably killed.",
                    exit_code=process.exitcode,
                )
            elif "error" in payload:
                self._finish(job, "failed", detail=str(payload["error"]))
            else:
                run_id = str(payload["run_id"])
                self._finish(job, "succeeded", run_id=run_id)
                if self.on_success is not None:
                    # After the slot is released and the status is written, so a
                    # poller sees "succeeded" even if the reload itself fails.
                    try:
                        self.on_success(run_id)
                    # A reload failure is /readyz's problem, not this job's.
                    except Exception:
                        _log.exception("could not activate run %s", run_id)
        # Broad, and the most important handler in this module: an exception
        # escaping this thread would leave `_active` set with no process behind
        # it, so every subsequent retrain would be refused with a 409 naming a job
        # that finished long ago. The slot has to be released on every path.
        except Exception:
            _log.exception("retrain watcher for job %s failed", job.job_id)
            self._finish(
                job,
                "failed",
                detail="The retrain could not be monitored. See the server log.",
                only_if_running=True,
            )
        finally:
            parent.close()
            process.close()

    @staticmethod
    def _receive(parent: Connection) -> dict[str, Any] | None:
        """The child's one message, or ``None`` if it never sent one.

        ``poll`` returns true at end of file as well as on data - a closed pipe is
        readable, it just reads as EOF - so the ``recv`` has to be guarded. Without
        this, a child killed before it could report raised ``EOFError`` in the
        watcher thread, which is precisely the case the ``None`` branch exists for.
        """
        if not parent.poll():
            return None
        try:
            received: dict[str, Any] = parent.recv()
        except EOFError:
            return None
        return received

    def _finish(
        self,
        job: Job,
        status: JobStatus,
        *,
        run_id: str | None = None,
        detail: str | None = None,
        exit_code: int | None = None,
        only_if_running: bool = False,
    ) -> None:
        with self._lock:
            # Set by the fallback handler, which cannot know whether the failure
            # happened before or after the job was recorded. Overwriting a
            # "succeeded" with "failed" because the callback that runs afterwards
            # threw would be the worse of the two mistakes.
            if only_if_running and job.status != "running":
                return
            job.status = status
            job.finished_at = now_iso()
            job.run_id = run_id
            job.detail = detail
            job.exit_code = exit_code
            self._active = None
        _log.info("retrain job %s %s%s", job.job_id, status, f": {detail}" if detail else "")

    def _remember(self, job: Job) -> None:
        """Add to the bounded history. Called with the lock held."""
        self._history[job.job_id] = job
        while len(self._history) > HISTORY:
            self._history.popitem(last=False)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._history.get(job_id)

    def recent(self) -> list[Job]:
        """Newest first, which is the order a status panel wants."""
        with self._lock:
            return list(reversed(self._history.values()))

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the current job finishes. For tests and for shutdown.

        Returns whether the watcher is done, so a caller can tell "finished" from
        "gave up waiting" rather than inferring it from a status that may not have
        been written yet.
        """
        watcher = self._watcher
        if watcher is None:
            return True
        watcher.join(timeout)
        return not watcher.is_alive()


#: Jobs a retrain can be in that mean "come back later" rather than "here is the
#: answer". Used by the status route to set ``Retry-After``.
PENDING_STATUSES: Final = frozenset({"running"})


# --- the inputs a job runs on --------------------------------------------------

#: A dataset id is the first 32 hex characters of the file's SHA-256, so it is a
#: safe path segment by construction. Validated anyway on the way back in, for the
#: same reason the run id is: one guard on a filesystem path is one guard too few.
DATASET_ID_PATTERN: Final = re.compile(r"\A[0-9a-f]{32}\Z")

#: Read in chunks this size while hashing and writing an upload. 1 MiB, so a 64
#: MiB extract is 64 iterations rather than one 64 MiB allocation - the whole
#: point of streaming it is not holding it.
UPLOAD_CHUNK_BYTES: Final = 1 << 20

#: How much of an upload to look at before deciding it is a CSV. A header plus a
#: few rows; enough for ``csv.Sniffer`` and for the comma count to mean something,
#: small enough to read before committing to the rest of the file.
SNIFF_BYTES: Final = 64 << 10


class UploadRejected(ValueError):
    """An upload that will not be stored, with a reason fit to return."""


class Reader(Protocol):
    """The only thing :func:`store_dataset` needs from what it is storing.

    A protocol rather than ``IO[bytes]`` because the real caller is not a file: it
    is an ASGI request body adapted to a blocking ``read``, and declaring
    ``IO[bytes]`` would demand ``seek``, ``tell``, ``fileno`` and a context manager
    that the adapter has no way to provide and this function never calls.
    """

    def read(self, size: int = ..., /) -> bytes: ...


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """One stored dataset, named by content rather than by filename.

    Content-addressed for two reasons that both matter here: an upload cannot
    overwrite an existing dataset, so "somebody replaced the file the model
    trained on" is not a state that exists; and the same extract uploaded twice
    gets the same id, so a run's manifest points at something a reviewer can
    verify.
    """

    dataset_id: str
    path: Path
    size_bytes: int
    #: Whether these exact bytes were already stored. Decided inside
    #: :func:`store_dataset`, immediately before the rename, because afterwards the
    #: two cases are indistinguishable - the destination name is the content, so a
    #: re-upload renames onto a file identical to itself.
    existing: bool = False


def looks_like_csv(head: bytes) -> None:
    """Raise unless the first chunk is plausibly a delimited text table.

    Sniffing rather than trusting the filename or the ``Content-Type``: both are
    supplied by the caller, and the thing being prevented is handing a joblib
    pickle to ``pd.read_csv`` and finding out what happens.

    Deliberately shallow. This is not validation - the pipeline's own column
    contract is, and it runs in a process that can be killed. This only refuses
    what is obviously not a CSV, because the cost of accepting one is a wasted
    retrain slot rather than a security problem.
    """
    if not head.strip():
        raise UploadRejected("The upload is empty.")
    # A pickle, a parquet file and a zip all start with a recognizable magic
    # number, and none of them is text. Checking for NUL catches all three plus
    # every other binary format without a list to maintain.
    if b"\x00" in head:
        raise UploadRejected("The upload is binary, not CSV.")
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError as error:
        raise UploadRejected("The upload is not UTF-8 text.") from error
    first = text.splitlines()[0] if text.splitlines() else ""
    if first.count(",") < 1:
        raise UploadRejected("The first line has no comma, so it is not a CSV header.")


def _drain(stream: Reader, temporary: Path, max_bytes: int) -> tuple[str, int, bytes]:
    """Copy the stream to ``temporary``, returning its digest, size, and first chunk.

    Separate from :func:`store_dataset` so the caller's cleanup handler wraps a
    single call rather than the whole loop - a ``raise`` inside a ``try`` whose
    ``except`` is a cleanup handler reads as control flow and is one edit away
    from being caught by the wrong branch.
    """
    digest = hashlib.sha256()
    size = 0
    head = b""
    with temporary.open("wb") as handle:
        while chunk := stream.read(UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise UploadRejected(f"The upload exceeds {max_bytes} bytes.")
            if len(head) < SNIFF_BYTES:
                head += chunk[: SNIFF_BYTES - len(head)]
            digest.update(chunk)
            handle.write(chunk)
        handle.flush()
        # fsync before the rename, so a crash cannot leave a correctly named
        # dataset whose bytes never reached the disk - which is the one failure a
        # content address cannot detect after the fact.
        os.fsync(handle.fileno())
    return digest.hexdigest()[:32], size, head


def store_dataset(stream: Reader, directory: Path, *, max_bytes: int) -> DatasetRef:
    """Stream an upload to a content-addressed file, or refuse it.

    Written to a temporary name in the destination directory and renamed once
    complete, so a connection that drops halfway leaves a ``.part`` file rather
    than a truncated dataset under a hash that does not describe it. Same
    directory, because a rename is only atomic within one filesystem.

    The size cap is enforced *while reading*, not from ``Content-Length``: a
    chunked upload declares nothing, and a declared length is the caller's
    number. The partial file is removed on any failure, including a cancelled
    request - hence ``BaseException`` rather than ``Exception``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".upload-{uuid.uuid4().hex}.part"
    try:
        dataset_id, size, head = _drain(stream, temporary, max_bytes)
        looks_like_csv(head)
        destination = directory / f"{dataset_id}.csv"
        existing = destination.is_file()
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return DatasetRef(dataset_id=dataset_id, path=destination, size_bytes=size, existing=existing)


def resolve_dataset(dataset_id: str, directory: Path) -> Path:
    """The stored path for a dataset id, or ``KeyError``.

    Two guards, as with a run id: the pattern rejects anything that is not a hash,
    and the resolved path is checked for containment in case a symlink got there
    another way.
    """
    if not DATASET_ID_PATTERN.match(dataset_id):
        raise KeyError(dataset_id)
    path = (directory / f"{dataset_id}.csv").resolve()
    if not path.is_file() or directory.resolve() != path.parent:
        raise KeyError(dataset_id)
    return path
