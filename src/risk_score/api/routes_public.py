"""Every route that only reads: scoring, identity, reports, and health.

Read-only in the sense that matters for a threat model - nothing here writes to
disk, starts a process, or changes what the next request will see. That is what
makes the file boundary useful: a reviewer asking "what can an unauthenticated
caller reach" reads this file and is done.

Scoring is read-only despite being a POST. The verb is POST because the request
body is an applicant's financial details, and a GET would put them in the query
string, which proxies log and browsers keep in history.

The report routes are deliberately thin. Every one of them is a name and a
docstring over :func:`risk_score.api.reports.report_response`, because caching,
ETag revalidation and path containment are the same problem for all seven and
seven copies of that logic is seven places for one of them to be missing a guard.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from risk_score.api.deps import ApplicantModelDep, ServiceDep, SettingsDep
from risk_score.api.reports import report_response
from risk_score.api.schemas import (
    ERROR_RESPONSES,
    BatchOut,
    BatchRequest,
    BatchRowOut,
    ErrorOut,
    HealthOut,
    ModelIdentity,
    PredictionOut,
    PredictOptions,
    ReasonCodeOut,
    SchemaOut,
    describe_spec,
)
from risk_score.api.scoring import Score, ScoringService

_log = logging.getLogger(__name__)

router = APIRouter()

OptionsDep = Annotated[PredictOptions, Query()]


@router.post(
    "/predict",
    response_model=PredictionOut,
    responses=ERROR_RESPONSES,
    tags=["scoring"],
    summary="Score one applicant",
)
async def predict(
    payload: dict[str, Any],
    options: OptionsDep,
    service: ServiceDep,
    applicant_model: ApplicantModelDep,
) -> PredictionOut:
    """One applicant in, one calibrated probability and its reasons out.

    Declared as a free-form object and validated against ``applicant_model``
    inside the handler, because the accepted fields come from the loaded bundle
    and a retrain can change them. ``/docs`` still lists them: the generated
    schema is patched into the OpenAPI document at startup.

    ``async def`` with synchronous work inside is deliberate and correct here.
    Scoring is single-digit milliseconds of CPU, so handing it to the threadpool
    would cost more in context switches than it saves, and the GIL makes the
    threadpool no more parallel than the event loop for numpy-bound work. A
    retrain is the opposite case and runs in a process pool.
    """
    applicant = _validated(applicant_model, payload)
    score = service.score(applicant, with_reasons=options.explain, top_k=options.top_k)
    return _prediction(score, service)


@router.post(
    "/predict/batch",
    response_model=BatchOut,
    responses=ERROR_RESPONSES,
    tags=["scoring"],
    summary="Score many applicants in one pass",
)
async def predict_batch(
    payload: BatchRequest,
    options: OptionsDep,
    service: ServiceDep,
    applicant_model: ApplicantModelDep,
    settings: SettingsDep,
) -> BatchOut:
    """Up to ``max_batch_rows`` applicants, with per-row errors inline.

    Two-pass on purpose: every row is validated first, then the valid ones are
    scored in a single vectorized call. Validating inside the scoring loop would
    mean a batch of 1000 rows makes 1000 passes through the sklearn pipeline,
    which is roughly two orders of magnitude slower than one pass over 1000 rows.

    Reason codes default **off** for a batch. SHAP over 1000 rows is the dominant
    cost and a batch is a scoring job, not a decision somebody has to justify to
    an applicant.
    """
    if len(payload.applicants) > settings.max_batch_rows:
        raise RequestValidationError(
            [
                {
                    "loc": ("body", "applicants"),
                    "msg": f"at most {settings.max_batch_rows} rows per batch, "
                    f"got {len(payload.applicants)}",
                    "type": "too_long",
                }
            ]
        )

    valid: list[tuple[int, dict[str, Any]]] = []
    rows: dict[int, BatchRowOut] = {}
    for index, raw in enumerate(payload.applicants):
        try:
            valid.append((index, _validated(applicant_model, raw)))
        except RequestValidationError as error:
            # Inline rather than fatal: failing all 1000 rows because row 407 has
            # a negative income is a worse answer than 999 scores and a message,
            # and a 422 about the body would not say which row was at fault.
            rows[index] = BatchRowOut(index=index, error=_first_message(error))

    scores = service.score_batch(
        [applicant for _, applicant in valid],
        with_reasons=options.explain,
        top_k=options.top_k,
    )
    for (index, _), score in zip(valid, scores, strict=True):
        rows[index] = BatchRowOut(index=index, prediction=_prediction(score, service))

    ordered = [rows[index] for index in sorted(rows)]
    return BatchOut(
        rows=ordered,
        scored=len(valid),
        failed=len(ordered) - len(valid),
        # Summed from the per-row figures, which already divide one vectorized
        # call across the batch, so this is the server-side cost of the request.
        latency_ms=sum(score.latency_ms for score in scores),
    )


@router.get(
    "/api/model",
    response_model=ModelIdentity,
    responses=ERROR_RESPONSES,
    tags=["model"],
    summary="Which model is active",
)
async def model_identity(service: ServiceDep) -> ModelIdentity:
    """The active run's identity, as it appears on every prediction."""
    return _identity(service)


@router.get(
    "/api/schema",
    response_model=SchemaOut,
    responses=ERROR_RESPONSES,
    tags=["model"],
    summary="The active model's input contract",
)
async def input_schema(service: ServiceDep) -> SchemaOut:
    """What ``/predict`` accepts, in a shape a form generator can consume."""
    return describe_spec(service.bundle.feature_spec, service.metadata.run_id)


# --- reports -------------------------------------------------------------------

#: Documented once for all seven report routes. ``?run_id=`` reads a past run
#: instead of the active one, which is what makes the dashboard's run-history
#: selector work without a separate endpoint per panel.
RunIdDep = Annotated[
    str | None,
    Query(description="A past run to read instead of the active one.", max_length=120),
]

#: Shared by every report route. The payloads are columnar tables rather than
#: declared models: their columns come from whatever the run wrote, so a pydantic
#: response model would be a fourth copy of a schema that changes per run.
_REPORT_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    200: {
        "content": {"application/json": {}},
        "description": "Columnar tables: one array per column, plus the run id.",
    },
    304: {"description": "The ETag matched; the client's copy is current."},
    404: {"model": ErrorOut, "description": "No such run, or this run has no such report."},
}


@router.get(
    "/api/metrics", responses=_REPORT_RESPONSES, tags=["reports"], summary="Headline metrics"
)
async def metrics(request: Request, run_id: RunIdDep = None) -> Response:
    """Everything ``metrics.json`` records: scores, split sizes, embargo counts."""
    return report_response(request, "metrics", run_id)


@router.get(
    "/api/calibration", responses=_REPORT_RESPONSES, tags=["reports"], summary="Calibration curves"
)
async def calibration(request: Request, run_id: RunIdDep = None) -> Response:
    """Both curves, keyed by partition, with per-bin counts.

    ``test`` is ``null`` rather than absent when the test window held too few
    positives to bin - a partial report, which the dashboard renders as a gap
    rather than as a zero.
    """
    return report_response(request, "calibration", run_id)


@router.get(
    "/api/threshold-costs",
    responses=_REPORT_RESPONSES,
    tags=["reports"],
    summary="Cost curve over candidate thresholds",
)
async def threshold_costs(request: Request, run_id: RunIdDep = None) -> Response:
    """The validation cost curve the selected threshold was chosen from.

    Validation only, and there is no test variant to ask for: a cost curve over
    test scores is the artifact that would let somebody pick a threshold on test
    by eye, which is the leak this project's split exists to prevent.
    """
    return report_response(request, "threshold-costs", run_id)


@router.get(
    "/api/vintages", responses=_REPORT_RESPONSES, tags=["reports"], summary="Metrics by vintage"
)
async def vintages(request: Request, run_id: RunIdDep = None) -> Response:
    """Per-origination-quarter default rate and metrics, after the embargo."""
    return report_response(request, "vintages", run_id)


@router.get("/api/drift", responses=_REPORT_RESPONSES, tags=["reports"], summary="PSI")
async def drift(request: Request, run_id: RunIdDep = None) -> Response:
    """Score PSI and per-feature PSI, with the 0.10 / 0.25 bands attached."""
    return report_response(request, "drift", run_id)


@router.get(
    "/api/shap-summary", responses=_REPORT_RESPONSES, tags=["reports"], summary="Global SHAP"
)
async def shap_summary(request: Request, run_id: RunIdDep = None) -> Response:
    """Mean absolute SHAP per feature: what drives the model overall."""
    return report_response(request, "shap-summary", run_id)


@router.get(
    "/api/comparison", responses=_REPORT_RESPONSES, tags=["reports"], summary="Model comparison"
)
async def comparison(request: Request) -> Response:
    """LR against XGBoost on an identical split. 404 until a comparison is run.

    The only report with no ``?run_id=``: a comparison describes several runs at
    once and is published at the report root rather than inside any one of them,
    so there is no run to select. The payload names the runs it compares.

    404 rather than an empty object, because "nothing has been compared" and "the
    two models scored the same" are different answers and a client cannot tell
    them apart from ``{}``.
    """
    return report_response(request, "comparison")


# --- health --------------------------------------------------------------------


@router.get("/healthz", response_model=HealthOut, tags=["health"], summary="Liveness")
async def healthz(request: Request) -> HealthOut:
    """Is the process answering.

    Deliberately 200 even with no model loaded, and deliberately separate from
    ``/readyz``. A liveness probe that fails when the model is missing makes an
    orchestrator restart a process that would have come up in exactly the same
    state, forever.
    """
    service: ScoringService | None = request.app.state.service
    return HealthOut(
        status="ok" if service is not None else "degraded",
        bundle_loaded=service is not None,
        run_id=None if service is None else service.metadata.run_id,
        reason=request.app.state.load_error,
    )


@router.get(
    "/readyz",
    response_model=HealthOut,
    responses=ERROR_RESPONSES,
    tags=["health"],
    summary="Readiness",
)
async def readyz(service: ServiceDep) -> HealthOut:
    """Can this process score a request.

    503 when it cannot, via the shared dependency, so a load balancer stops
    sending traffic to a replica whose bundle failed to load. This is the probe
    the container's ``HEALTHCHECK`` uses.
    """
    return HealthOut(status="ok", bundle_loaded=True, run_id=service.metadata.run_id)


def _validated(applicant_model: type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
    """One applicant, validated against the active spec, as canonical names.

    Re-raised as ``RequestValidationError`` so the body is validated by the same
    rules and reported in the same shape as any other 422, rather than escaping
    as a 500 from a pydantic error nobody caught.

    ``model_dump`` returns field names, not aliases, which is what turns a caller
    posting ``funded_amnt`` into the ``loan_amnt`` the pipeline declares.
    """
    try:
        applicant = applicant_model.model_validate(payload)
    except ValidationError as error:
        raise RequestValidationError(error.errors()) from error
    return applicant.model_dump()


def _first_message(error: RequestValidationError) -> str:
    """The first validation problem, named but not quoted.

    One message rather than all of them: a batch row's error field is read in a
    table, and the field name plus the reason is enough to fix the row. The value
    is never included - a batch is the request most likely to be logged whole.
    """
    for item in error.errors():
        location = [str(part) for part in item.get("loc", ()) if part != "body"]
        return f"{'.'.join(location) or 'body'}: {item.get('msg', 'invalid')}"
    return "invalid applicant"


def _identity(service: ScoringService) -> ModelIdentity:
    metadata = service.metadata
    return ModelIdentity(
        run_id=metadata.run_id,
        model_type=metadata.model_type,
        feature_tier=metadata.feature_tier,
        created_at=metadata.created_at,
        git_commit=metadata.git_commit,
        bundle_schema_version=metadata.bundle_schema_version,
    )


def _prediction(score: Score, service: ScoringService) -> PredictionOut:
    """A :class:`Score` as a response body, with the model that produced it."""
    return PredictionOut(
        default_probability=score.default_probability,
        decision=score.decision,  # type: ignore[arg-type]
        threshold=score.threshold,
        baseline_log_odds=score.baseline_log_odds,
        total_log_odds=score.total_log_odds,
        reasons=[
            ReasonCodeOut(
                feature=reason.feature,
                label=reason.label,
                value=reason.value,
                log_odds=reason.log_odds,
                direction=reason.direction,  # type: ignore[arg-type]
            )
            for reason in score.reasons
        ],
        model=_identity(service),
        latency_ms=score.latency_ms,
    )
