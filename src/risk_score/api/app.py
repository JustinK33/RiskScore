"""The application factory: one function that assembles a configured service.

A factory rather than a module-level ``app = FastAPI()`` because a module-level
app is configured by import side effects. It reads the environment at import
time, so a test cannot build a second one with different settings, and the
uvicorn ``--reload`` worker and the parent disagree about which one is real.

Two things happen here that do not happen anywhere else.

**The bundle is loaded before the app is returned**, not in the lifespan
handler. That is deliberate: the pydantic model describing an applicant is
generated from the bundle's own ``FeatureSpec``, so having the bundle in hand at
construction time is what lets ``/docs`` describe the model that will actually
score the request. A failure to load is recorded rather than raised - unless
``require_bundle`` is set - because a fresh clone has no bundle and should still
boot far enough to say so.

**The middleware stack is assembled in one place, outermost first**::

    RequestContext   request id, body size cap, X-Request-ID on the way out
    TrustedHost      Host header allow-list (DNS rebinding)
    GZip             responses over 1 KiB
    <routes>

Order matters. The body cap has to sit outside anything that reads the body, and
the request id has to be bound before any handler - including an error handler -
can log.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from risk_score.api.routes_public import router as public_router
from risk_score.api.schemas import ErrorOut, build_applicant_model
from risk_score.api.scoring import ScoringService
from risk_score.api.settings import Settings
from risk_score.artifacts import load_active_bundle, read_active_run_id
from risk_score.logging_setup import bind_request_id, configure_logging, request_id_var

_log = logging.getLogger(__name__)

#: Below this, compression costs more than it saves and the ``Content-Encoding``
#: round trip is pure latency. A ``/predict`` response with reason codes is
#: around 1.5 KiB, so the interesting payloads are compressed and a health check
#: is not.
GZIP_MINIMUM_BYTES = 1024

#: Scope key the request id is stashed under, for the rare consumer that has a
#: scope but no contextvar (an exception handler running in another task).
SCOPE_REQUEST_ID = "riskscore_request_id"

#: Spelled numerically because starlette renamed both constants:
#: ``HTTP_422_UNPROCESSABLE_ENTITY`` is deprecated in favour of
#: ``HTTP_422_UNPROCESSABLE_CONTENT`` and ``HTTP_413_REQUEST_ENTITY_TOO_LARGE``
#: in favour of ``HTTP_413_CONTENT_TOO_LARGE``. Referring to either by name makes
#: the import version-dependent for no gain; the numbers have never moved.
HTTP_413 = 413
HTTP_422 = 422

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class BodyTooLarge(Exception):
    """A request body exceeded ``max_body_bytes`` while being read.

    Raised from the wrapped ``receive`` rather than returned as a response,
    because by the time the overrun is detected a handler is already awaiting the
    body. Starlette's exception middleware turns it into a 413 through the
    handler registered in :func:`create_app`.
    """

    def __init__(self, limit: int) -> None:
        super().__init__(f"Request body exceeds {limit} bytes.")
        self.limit = limit


class RequestContextMiddleware:
    """Request id, body size cap, and the response header, as raw ASGI.

    Raw ASGI rather than ``BaseHTTPMiddleware`` for one concrete reason: the body
    cap has to see the request as a stream of chunks. ``BaseHTTPMiddleware``
    hands a handler a fully buffered body, which means the memory has already
    been spent by the time anything can object - the cap would only ever refuse
    requests it had already accepted.

    A declared ``Content-Length`` over the limit is refused before the
    application is called at all, so an oversized upload costs one response and
    no allocation. A chunked body with no declared length is counted as it
    arrives.
    """

    def __init__(self, app: Any, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        with bind_request_id(_header(scope, b"x-request-id")) as request_id:
            scope[SCOPE_REQUEST_ID] = request_id
            declared = _content_length(scope)
            if declared is _MALFORMED:
                await _send_error(
                    scope,
                    send,
                    status.HTTP_400_BAD_REQUEST,
                    "Malformed Content-Length.",
                    request_id,
                )
                return
            if isinstance(declared, int) and declared > self.max_body_bytes:
                await _send_error(
                    scope,
                    send,
                    HTTP_413,
                    f"Request body exceeds {self.max_body_bytes} bytes.",
                    request_id,
                )
                return
            await self.app(
                scope,
                _counting_receive(receive, self.max_body_bytes),
                _stamping_send(send, request_id),
            )


#: Sentinel for a ``Content-Length`` that is present but not a number. Distinct
#: from ``None`` (absent, which is legal for a chunked body) because the two get
#: different answers: 400 versus counting as we go.
_MALFORMED = object()


def _header(scope: Scope, name: bytes) -> str | None:
    headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
    for key, value in headers:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _content_length(scope: Scope) -> int | object | None:
    raw = _header(scope, b"content-length")
    if raw is None:
        return None
    # `isdigit` rather than `int()` in a try block: `int(" -1 ")` succeeds and a
    # negative length is the input that makes a naive `read(length)` hang.
    return int(raw) if raw.isdigit() else _MALFORMED


def _counting_receive(receive: Receive, limit: int) -> Receive:
    """``receive``, wrapped to refuse a body that grows past ``limit``."""
    seen = 0

    async def wrapped() -> Message:
        nonlocal seen
        message = await receive()
        if message["type"] == "http.request":
            seen += len(message.get("body", b""))
            if seen > limit:
                raise BodyTooLarge(limit)
        return message

    return wrapped


def _stamping_send(send: Send, request_id: str) -> Send:
    """``send``, wrapped to add ``X-Request-ID`` to the response.

    On every response including errors, because the id is only useful to somebody
    reporting a failure - and a failure is exactly when a service is least likely
    to have added it.
    """

    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            headers.append((b"x-request-id", request_id.encode("latin-1")))
            message = {**message, "headers": headers}
        await send(message)

    return wrapped


async def _send_error(scope: Scope, send: Send, code: int, detail: str, request_id: str) -> None:
    """A complete ASGI error response, for refusals made before the app runs.

    The request body is never read: a client that declared 900 MB is answered and
    the connection closed rather than drained, which is the whole point of
    checking the declared length instead of counting bytes.
    """
    response = JSONResponse(
        status_code=code,
        content=ErrorOut(detail=detail, request_id=request_id).model_dump(),
        headers={"X-Request-ID": request_id},
    )
    await response(scope, _no_body, send)


async def _no_body() -> Message:
    """A ``receive`` for a response that never reads a request body."""
    return {"type": "http.request", "body": b"", "more_body": False}


def _error_body(detail: str) -> dict[str, str]:
    """The single error shape, with whichever request id is in scope."""
    return ErrorOut(detail=detail, request_id=request_id_var.get() or "-").model_dump()


def load_service(app: FastAPI) -> None:
    """Load the active bundle into ``app.state``, recording failure rather than raising.

    Called at construction and again after a retrain publishes a new run. Every
    piece of per-bundle state is replaced together - service, applicant model,
    and the cached OpenAPI document - because a half-swapped state would serve
    scores from a new model against the old model's input contract.

    Failures are recorded in ``load_error`` and reported by ``/readyz`` and by the
    503 from :func:`~risk_score.api.deps.get_service`. The exception list is
    broad on purpose: unpickling a bundle written by a different library version
    fails in ways joblib does not narrow, and a service that cannot load a model
    should say so rather than lose the reason in a traceback.
    """
    settings: Settings = app.state.settings
    app.state.service = None
    app.state.applicant_model = None
    app.state.load_error = None
    # Cleared so /docs regenerates against the new input contract; FastAPI caches
    # the document on first request and would otherwise describe the old model
    # forever.
    app.openapi_schema = None

    if read_active_run_id(settings.reports_dir) is None:
        app.state.load_error = f"no active run under {settings.reports_dir.name}/"
        _log.warning("no active run: %s", app.state.load_error)
        return
    try:
        bundle = load_active_bundle(settings.reports_dir)
        service = ScoringService(bundle)
        model = build_applicant_model(bundle.feature_spec)
    # Broad, and recorded rather than swallowed - see the docstring.
    except Exception as error:
        app.state.load_error = f"{type(error).__name__}: {error}"
        _log.exception("failed to load the active bundle")
        return

    app.state.service = service
    app.state.applicant_model = model
    _log.info(
        "loaded run %s (%s, %s tier, %d features, reason codes %s)",
        bundle.metadata.run_id,
        bundle.metadata.model_type,
        bundle.metadata.feature_tier,
        len(bundle.feature_spec.model_features),
        "on" if service.can_explain else f"off: {service.explainer_error}",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """A configured application, with its model already loaded.

    ``settings`` is injected for tests; in production it is built here from the
    environment so that exactly one instance exists.
    """
    settings = settings or Settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    app = FastAPI(
        title="RiskScore",
        version="1.0.0",
        summary="Credit default risk scoring with reason codes.",
        description=(
            "Scores one loan application at origination against a versioned model "
            "bundle. Every response carries the run id that produced it, so a "
            "decision can be reproduced. The accepted request body is generated "
            "from the active model's own feature spec - see `GET /api/schema`."
        ),
        # Turned off together: an OpenAPI document is a map of the service, and
        # /docs without /openapi.json is a blank page rather than a hardening
        # measure.
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    app.state.service = None
    app.state.applicant_model = None
    app.state.load_error = None
    load_service(app)

    if settings.require_bundle and app.state.service is None:
        # A container that cannot score should fail to start, not pass its
        # health check while returning 503 to every caller.
        raise RuntimeError(
            f"RISKSCORE_REQUIRE_BUNDLE is set and no bundle loaded: {app.state.load_error}"
        )

    # Added last-to-first: `add_middleware` prepends, so the final call is the
    # outermost layer. RequestContext must be outermost - everything inside it,
    # including TrustedHost's refusal, should carry a request id.
    app.add_middleware(GZipMiddleware, minimum_size=GZIP_MINIMUM_BYTES)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_body_bytes)

    _install_handlers(app)
    _install_openapi(app)
    app.include_router(public_router)

    _log.info(
        "serving on %s:%d (docs %s, mutating routes: %s)",
        settings.host,
        settings.port,
        "on" if settings.docs_enabled else "off",
        settings.mutating_routes_enabled(),
    )
    if settings.bind_is_public:
        _log.warning("bound to %s, which is reachable off this host", settings.host)
    return app


def _install_handlers(app: FastAPI) -> None:
    """One error shape for every failure the service can produce.

    FastAPI's defaults are close but not the same: ``{"detail": ...}`` with no
    request id, and a validation error body that echoes the input. Echoing the
    input is the part worth replacing - a 422 that quotes an applicant's income
    back at whoever sent it puts that value in a proxy log.
    """

    @app.exception_handler(BodyTooLarge)
    async def _too_large(request: Request, error: BodyTooLarge) -> JSONResponse:
        return JSONResponse(
            status_code=HTTP_413,
            content=_error_body(str(error)),
        )

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, error: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=_error_body(str(error.detail)),
            headers=error.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=HTTP_422,
            content=_error_body(summarize_validation_errors(error.errors())),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, error: Exception) -> JSONResponse:
        # The traceback goes to the log with the request id; the caller gets the
        # id and nothing else. This is the handler that keeps a filesystem path
        # or the dataset's date range out of an error body.
        _log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body("Internal error. Quote the request id when reporting this."),
        )


#: How many field names a validation message lists before giving up and counting.
#: A caller posting an object with 40 wrong fields is not helped by 40 names.
MAX_REPORTED_FIELDS = 5


def summarize_validation_errors(errors: Sequence[Any]) -> str:
    """Field names and reasons, with no submitted values.

    ``loc`` is the path pydantic walked, e.g. ``("body", "applicants", 3,
    "annual_inc")``. The leading ``body`` is dropped because every one of these
    is about the body, and an index is kept because "row 3" is the only way a
    batch caller finds the offending row.
    """
    parts: list[str] = []
    for error in errors[:MAX_REPORTED_FIELDS]:
        location = [str(item) for item in error.get("loc", ()) if item != "body"]
        where = ".".join(location) or "body"
        parts.append(f"{where}: {error.get('msg', 'invalid')}")
    if len(errors) > MAX_REPORTED_FIELDS:
        parts.append(f"and {len(errors) - MAX_REPORTED_FIELDS} more")
    return "; ".join(parts) or "The request body did not match the input contract."


def _install_openapi(app: FastAPI) -> None:
    """Describe the *loaded* model's inputs in the OpenAPI document.

    The predict routes declare their body as a free-form object, because the
    accepted fields are not known until a bundle is loaded and a retrain can
    change them. Declaring them statically would mean a third hand-maintained
    copy of the column list - the thing
    :mod:`risk_score.api.schemas` exists to avoid.

    So the schema is patched in afterwards, from the same generated model that
    validates the request. ``/docs`` then lists every accepted field with its
    description and bounds, and it follows the active run rather than the source
    tree. Patch failures are logged and ignored: broken documentation is a much
    smaller problem than a service that will not answer ``/openapi.json``.
    """

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema
        document = get_openapi(
            title=app.title,
            version=app.version,
            summary=app.summary,
            description=app.description,
            routes=app.routes,
        )
        model = app.state.applicant_model
        if model is not None:
            try:
                _patch_applicant_schema(document, model)
            except (KeyError, TypeError, AttributeError):
                _log.exception("could not describe the applicant body in OpenAPI")
        app.openapi_schema = document
        return document

    app.openapi = openapi  # type: ignore[method-assign]


def _patch_applicant_schema(document: dict[str, Any], model: type[Any]) -> None:
    """Point both predict bodies at the generated ``Applicant`` schema."""
    generated = model.model_json_schema(ref_template="#/components/schemas/{model}")
    schemas = document.setdefault("components", {}).setdefault("schemas", {})
    schemas.update(generated.pop("$defs", {}))
    schemas["Applicant"] = generated

    reference = {"$ref": "#/components/schemas/Applicant"}
    body = document["paths"]["/predict"]["post"]["requestBody"]["content"]
    body["application/json"]["schema"] = reference
    # The batch body keeps its wrapper object; only the item type is unknown until
    # a bundle is loaded.
    batch = schemas.get("BatchRequest", {}).get("properties", {}).get("applicants")
    if batch is not None:
        batch["items"] = reference
