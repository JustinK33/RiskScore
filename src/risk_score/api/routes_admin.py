"""The three routes that change what the server will do next, and nothing else.

Every route here is off by default, needs a key, and is separated from
:mod:`risk_score.api.routes_public` for the same reason the public file says it
is: the interesting question about this service is what an unauthenticated caller
can reach, and that question is answered by reading one file. This is the other
one - the whole of the mutating surface, in one place, short enough to audit in a
sitting.

The feature is genuinely dangerous and is treated that way. ``POST /api/runs``
is remote-triggered execution over caller-supplied data that ends by writing a
pickle the service will later load. Four independent things stand between a
request and that:

* ``RISKSCORE_ALLOW_UPLOAD`` / ``RISKSCORE_ALLOW_RETRAIN``, both off by default,
  checked per route;
* ``X-API-Key``, and :class:`~risk_score.api.settings.Settings` refuses to start
  at all if a mutating route is enabled on a non-loopback bind with no key;
* a ``dataset_id`` rather than a path, so a caller names content that was already
  stored rather than any file the process can read;
* the fit itself in a separate, killable process with one slot.

**403, not 404, for a switched-off route.** The routes appear in ``/docs``
regardless - they are part of the service's contract - so pretending they do not
exist would be a lie a client can see through in one request. The message names
the environment variable, because "not found" for a documented route is the least
actionable answer an API can give somebody following the runbook.

The flag is checked *before* the key, so an operator who has not enabled the
feature is told that rather than being asked for a credential that would not help.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Annotated, Any

import anyio.from_thread
import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from risk_score.api.deps import SettingsDep, require_api_key
from risk_score.api.jobs import (
    PENDING_STATUSES,
    Job,
    JobBusy,
    JobRunner,
    TrainRequest,
    UploadRejected,
    resolve_dataset,
    store_dataset,
)
from risk_score.api.schemas import ERROR_RESPONSES, DatasetOut, JobOut, RetrainIn
from risk_score.api.settings import Settings

_log = logging.getLogger(__name__)

router = APIRouter()

#: How long a poller should wait before asking about a running job again. A fit is
#: minutes, so a shorter interval is only load; the header exists so a client does
#: not have to guess, and a well-behaved one then polls at the server's pace.
RETRY_AFTER_SECONDS = "5"

#: Spelled numerically for the reason :mod:`risk_score.api.app` documents:
#: starlette renamed the constant, and importing it by name from ``app`` would be
#: circular anyway - ``app`` imports this module.
HTTP_422 = 422

#: Documented once for the three routes here. 403 and 409 are the two answers
#: unique to this file, and both are ordinary operating states rather than bugs.
_ADMIN_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    403: {"description": "The feature is switched off. The detail names the variable."},
    409: {"description": "A retrain is already running. Only one runs at a time."},
}


def _require_enabled(enabled: bool, variable: str) -> None:
    """403 naming the environment variable that would switch this route on."""
    if not enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This route is disabled. Set {variable}=1 to enable it.",
        )


def require_upload(settings: SettingsDep) -> None:
    _require_enabled(settings.allow_upload, "RISKSCORE_ALLOW_UPLOAD")


def require_retrain(settings: SettingsDep) -> None:
    _require_enabled(settings.allow_retrain, "RISKSCORE_ALLOW_RETRAIN")


def get_runner(request: Request) -> JobRunner:
    """The one job runner, built in ``create_app``.

    A 503 rather than an attribute error if it is absent, which happens only in a
    partially constructed app - a test building one by hand, in practice.
    """
    runner: JobRunner | None = getattr(request.app.state, "job_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No retrain runner is configured.",
        )
    return runner


RunnerDep = Annotated[JobRunner, Depends(get_runner)]

#: Applied per route rather than to the whole router, because the *order* is
#: load-bearing: the feature flag is checked before the credential.
UPLOAD_GUARDS = [Depends(require_upload), Depends(require_api_key)]
RETRAIN_GUARDS = [Depends(require_retrain), Depends(require_api_key)]


@router.post(
    "/api/datasets",
    response_model=DatasetOut,
    responses=_ADMIN_RESPONSES,
    dependencies=UPLOAD_GUARDS,
    tags=["admin"],
    status_code=status.HTTP_201_CREATED,
    summary="Store a CSV extract to train on later",
)
async def upload_dataset(request: Request, settings: SettingsDep) -> DatasetOut:
    """Stream a CSV to a content-addressed file and return its id.

    The raw request stream, not ``UploadFile`` and not a multipart form. A
    multipart parser would be a third-party dependency and a second size accounting
    to get right, for a route whose entire body is one file; ``request.stream()``
    is already chunked, which is what lets the cap be enforced against bytes that
    have arrived rather than against a number the caller supplied.

    The body-size middleware does not apply here - its limit is 1 MiB, sized for a
    ``/predict`` body - so this route carries its own, larger cap. That is why the
    route is exempted in :func:`~risk_score.api.app.create_app`.

    Re-uploading the same bytes is a no-op that returns the same id, and says so.
    """
    stream = _SyncStream(request)
    try:
        # In a worker thread, which is what makes `_SyncStream` legal: it schedules
        # each chunk read back onto this loop, and `from_thread.run` needs to be
        # called from off the loop to do that. Hashing, writing and fsyncing a 64
        # MiB file on the event loop would also stall every concurrent /predict.
        reference = await anyio.to_thread.run_sync(
            partial(
                store_dataset, stream, settings.datasets_dir, max_bytes=settings.max_upload_bytes
            )
        )
    except UploadRejected as error:
        # 422 rather than 400: the request was well-formed and its body was not what
        # the route accepts, which is the same answer a malformed applicant gets.
        raise HTTPException(status_code=HTTP_422, detail=str(error)) from error

    _log.info(
        "stored dataset %s (%d bytes%s)",
        reference.dataset_id,
        reference.size_bytes,
        ", already present" if reference.existing else "",
    )
    return DatasetOut(
        dataset_id=reference.dataset_id,
        size_bytes=reference.size_bytes,
        existing=reference.existing,
    )


@router.post(
    "/api/runs",
    response_model=JobOut,
    responses=_ADMIN_RESPONSES,
    dependencies=RETRAIN_GUARDS,
    tags=["admin"],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a retrain on a stored dataset",
)
async def start_retrain(
    payload: RetrainIn,
    request: Request,
    settings: SettingsDep,
    runner: RunnerDep,
) -> JobOut:
    """202 with a job id, immediately. The fit happens in another process.

    202 rather than 200 because nothing has been trained yet: the response is a
    receipt, and ``GET /api/jobs/{job_id}`` is where the answer arrives. Returning
    200 with the run's metrics would mean holding the connection open for the
    length of a fit, which is the behaviour this replaces.

    409 when one is already running - see :mod:`risk_score.api.jobs` for why that
    is a refusal rather than a queue.
    """
    try:
        dataset = resolve_dataset(payload.dataset_id, settings.datasets_dir)
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such dataset. Upload it first with POST /api/datasets.",
        ) from error

    try:
        job = runner.submit(
            TrainRequest(
                dataset=dataset,
                output_dir=settings.reports_dir,
                model_type=payload.model_type,
                include_lender_priced=payload.include_lender_priced,
            )
        )
    except JobBusy as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
            # A busy slot clears when the running fit ends, so this is the same
            # "come back later" a running job's status carries.
            headers={"Retry-After": RETRY_AFTER_SECONDS},
        ) from error
    return _job_out(job)


@router.get(
    "/api/jobs/{job_id}",
    response_model=JobOut,
    responses=_ADMIN_RESPONSES,
    dependencies=RETRAIN_GUARDS,
    tags=["admin"],
    summary="One retrain's status",
)
async def job_status(job_id: str, response: Response, runner: RunnerDep) -> JobOut:
    """Where a submitted retrain's outcome shows up, including why it failed.

    Behind the key with the route that starts a job, because ``detail`` is the
    child's exception text - a server-side error message, and the one place this
    service would otherwise volunteer an internal failure to a caller.

    ``Retry-After`` while it is still running, so a poller does not have to invent
    an interval. Absent once the job is finished, which is itself the signal that
    there is nothing more to wait for.
    """
    job = runner.get(job_id)
    if job is None:
        # No distinction between "never existed" and "fell off the end of a
        # 32-entry history": both mean this service cannot tell you, and the run
        # registry is the durable record.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such job.")
    if job.status in PENDING_STATUSES:
        response.headers["Retry-After"] = RETRY_AFTER_SECONDS
    return _job_out(job)


def _job_out(job: Job) -> JobOut:
    """A :class:`Job` as a response body.

    Through the model rather than returning ``job.as_dict()`` so a field added to
    the dataclass cannot appear in the API without somebody deciding it should.
    """
    return JobOut(**job.as_dict())


async def _next_chunk(chunks: AsyncIterator[bytes]) -> bytes:
    """One chunk, as a coroutine.

    ``anyio.from_thread.run`` takes a coroutine *function*, and ``__anext__`` is an
    awaitable-returning method rather than one - so this wraps it. Also the one
    place the async iteration protocol appears, which keeps the adapter below
    readable.
    """
    return await anext(chunks)


class _SyncStream:
    """The request body as a blocking ``read``, for :func:`store_dataset`.

    ``store_dataset`` is synchronous, and deliberately: it hashes, writes and
    fsyncs, which is filesystem work that an ``async def`` cannot overlap with
    anything useful. Rather than duplicate it as a coroutine, the async stream is
    adapted to the file-like interface it wants.

    ``anyio.from_thread.run`` is what makes that safe. The caller hands this to
    ``store_dataset`` inside ``to_thread.run_sync``, so ``read`` runs off the loop,
    and each chunk read is scheduled back onto the loop that owns the request.
    Awaiting the iterator from the thread directly would touch loop state from off
    the loop, which is undefined behaviour rather than an error you get told about.

    ponytail: no seek, no tell, no context manager - ``store_dataset`` calls
    ``read`` and nothing else. If it ever needs more, subclass ``io.RawIOBase``.
    """

    def __init__(self, request: Request) -> None:
        self._chunks = request.stream()
        self._buffer = b""
        self._done = False

    def read(self, size: int = -1) -> bytes:
        while not self._done and (size < 0 or len(self._buffer) < size):
            try:
                chunk = anyio.from_thread.run(_next_chunk, self._chunks)
            except StopAsyncIteration:
                self._done = True
                break
            if not chunk:
                # An empty chunk means end of body in the ASGI protocol, but the
                # iterator yields one more time before stopping, so this cannot be
                # treated as "keep going" or the loop never ends.
                self._done = True
                break
            self._buffer += chunk
        if size < 0:
            taken, self._buffer = self._buffer, b""
            return taken
        taken, self._buffer = self._buffer[:size], self._buffer[size:]
        return taken


def build_runner(settings: Settings, on_success: Callable[[str], None] | None = None) -> JobRunner:
    """The runner ``create_app`` installs. Here so the wiring lives with the routes."""
    return JobRunner(
        timeout_seconds=settings.job_timeout_seconds,
        log_level=settings.log_level,
        on_success=on_success,
    )
