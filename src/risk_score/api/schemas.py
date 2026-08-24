"""Request and response bodies, generated from the bundle's own ``FeatureSpec``.

The applicant model is built at startup with :func:`pydantic.create_model` rather
than written out by hand, because a hand-written one is a third copy of the column
list - after the registry and the persisted spec - and the copy that drifts is
always the one furthest from the data. Generating it means a bundle trained without
``fico_range_low`` serves an API that does not advertise it, with no edit.

What that buys, concretely:

* ``/docs`` lists every accepted field, its type, and its description, taken from
  the column registry. The API documents itself from the same sentences the data
  dictionary is built from.
* A required input is a 422 naming the field, not a ``KeyError`` from inside a
  transformer.
* ``extra="forbid"``, so ``anual_inc`` is a validation error rather than a silently
  imputed median. This is the setting that matters most for a scoring API: a typo
  that still returns 200 returns a *different applicant's* risk.
* Source aliases are accepted, because the registry knows them. A client holding
  ``funded_amnt`` need not learn that this project calls it ``loan_amnt``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, create_model

from risk_score.artifacts import feature_tier
from risk_score.explain import DEFAULT_TOP_K
from risk_score.features import ParseKind, alias_priority, column_spec
from risk_score.transformers import FeatureSpec

#: The JSON type each parse kind accepts.
#:
#: Three kinds accept ``float | str``, and that union is the whole point of them.
#: ``term``, ``emp_length``, ``int_rate`` and ``revol_util`` arrive as
#: ``' 36 months'``, ``'10+ years'`` and ``'13.56%'`` in one real extract and as
#: plain numbers in the other, so a client holding either dialect can post what it
#: has. Narrowing them to ``float`` here would mean the API rejects the exact
#: strings the training pipeline was built to read, and would put a second parsing
#: policy in this file - :class:`~risk_score.transformers.CanonicalizeFrame`
#: already owns that, and owning it once is why both dialects work at all.
_TYPE_BY_PARSE_KIND: dict[ParseKind, Any] = {
    ParseKind.NUMERIC: float,
    ParseKind.PERCENT: float | str,
    ParseKind.TERM_MONTHS: float | str,
    ParseKind.EMP_LENGTH_YEARS: float | str,
    ParseKind.MONTH_DATE: str,
    ParseKind.CATEGORY: str,
    ParseKind.TEXT: str,
}

#: Bounds a *request* is refused for, as opposed to bounds the pipeline clips. The
#: pipeline is lenient by necessity - it ingests a 1.19 GB file nobody curated -
#: and an interactive caller is better served by being told the value is wrong.
#: Only obviously-impossible values are here; anything arguable is left to the
#: model, because an API that refuses unusual applicants cannot score them.
#:
#: Enforced as a validation constraint only where the field is numbers-only, since
#: ``ge`` cannot be applied to a ``float | str`` union without also rejecting
#: ``'10+ years'``. For those four the bound is advisory: reported by
#: ``/api/schema`` as a form hint, and left to the parser and the model otherwise.
_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "loan_amnt": (0.0, None),
    "annual_inc": (0.0, None),
    "term": (1.0, None),
    "emp_length": (0.0, 100.0),
    "dti": (-100.0, 1000.0),
    "revol_util": (0.0, 1000.0),
    "revol_bal": (0.0, None),
    "delinq_2yrs": (0.0, None),
    "inq_last_6mths": (0.0, None),
    "open_acc": (0.0, None),
    "total_acc": (0.0, None),
    "pub_rec": (0.0, None),
    "int_rate": (0.0, 100.0),
    "installment": (0.0, None),
}


#: How the dashboard should render each parse kind. Coarser than ``ParseKind``
#: because a form has four kinds of input, not seven: everything numeric is a
#: number box regardless of how the extract happened to spell it.
_KIND_BY_PARSE_KIND: dict[ParseKind, str] = {
    ParseKind.NUMERIC: "numeric",
    ParseKind.PERCENT: "numeric",
    ParseKind.TERM_MONTHS: "numeric",
    ParseKind.EMP_LENGTH_YEARS: "numeric",
    ParseKind.MONTH_DATE: "date",
    ParseKind.CATEGORY: "categorical",
    ParseKind.TEXT: "text",
}


def _field(name: str, *, required: bool) -> tuple[Any, Any]:
    """One ``(annotation, FieldInfo)`` pair for ``create_model``."""
    spec = column_spec(name)
    base = _TYPE_BY_PARSE_KIND[spec.parse]
    low, high = _BOUNDS.get(name, (None, None))
    constraints: dict[str, Any] = {}
    # Only for the numbers-only kinds; see the note on `_BOUNDS`.
    if base is float:
        if low is not None:
            constraints["ge"] = low
        if high is not None:
            constraints["le"] = high

    # Aliases beyond the canonical name, so a caller can post the extract's own
    # column names. `populate_by_name` on the model keeps the canonical name
    # working at the same time.
    aliases = tuple(key for key in alias_priority(name) if key != name)
    field = Field(
        default=... if required else None,
        description=spec.description or None,
        validation_alias=AliasChoices(name, *aliases) if aliases else None,
        **constraints,
    )
    return (base if required else base | None, field)


def build_applicant_model(spec: FeatureSpec) -> type[BaseModel]:
    """A pydantic model for one applicant, from the fitted spec.

    Field order follows ``spec.raw_inputs``, which is registry order, so ``/docs``
    lists the loan request before the bureau pull rather than alphabetically.
    """
    required = set(spec.required_raw_inputs)
    fields = {name: _field(name, required=name in required) for name in spec.raw_inputs}
    model: type[BaseModel] = create_model(
        "Applicant",
        __config__=ConfigDict(
            # The setting that turns a typo into a 422 instead of a plausible
            # score computed from an imputed median.
            extra="forbid",
            populate_by_name=True,
            json_schema_extra={
                "description": (
                    "One loan application, at origination. Fields are the inputs "
                    "the active model was fitted on; the extract's own column "
                    "names are accepted as aliases."
                )
            },
        ),
        **fields,  # type: ignore[call-overload]
    )
    return model


# --- request bodies ------------------------------------------------------------


class BatchRequest(BaseModel):
    """A batch of applicants.

    A wrapper object rather than a bare JSON array so the request has somewhere
    to grow - a top-level array cannot gain a field without breaking every
    client. The item type is patched into the OpenAPI document at startup from
    the loaded bundle's spec, since it is not known until then.
    """

    model_config = ConfigDict(extra="forbid")

    applicants: list[dict[str, Any]] = Field(
        min_length=1,
        description="One object per applicant, each shaped like the /predict body.",
    )


class PredictOptions(BaseModel):
    """Knobs shared by both predict routes."""

    model_config = ConfigDict(extra="forbid")

    explain: bool = Field(
        default=True,
        description="Include reason codes. Roughly triples the latency of a single call.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=50,
        description="How many reason codes to return, largest absolute contribution first.",
    )


# --- response bodies -----------------------------------------------------------


class ReasonCodeOut(BaseModel):
    """One reason code. Exact, in the model's own log-odds."""

    feature: str
    label: str = Field(description="Human-readable meaning, from the column registry.")
    value: Any = Field(description="The applicant's value after engineering, before scaling.")
    log_odds: float = Field(description="Signed contribution. The full set sums to the score.")
    direction: Literal["increases risk", "reduces risk", "no effect"]


class ModelIdentity(BaseModel):
    """Which model produced a score. On every prediction response on purpose.

    A score with no model identity cannot be reproduced, and "which model declined
    this application" is the first question asked about any decline. Cheap to
    include and impossible to reconstruct later.
    """

    run_id: str
    model_type: str
    feature_tier: str
    created_at: str
    git_commit: str
    bundle_schema_version: int


class PredictionOut(BaseModel):
    """One scored applicant."""

    default_probability: float = Field(
        description="Calibrated probability of default over the full term."
    )
    decision: Literal["approve", "decline"]
    threshold: float = Field(description="The bundle's own cut-off, selected on validation.")
    baseline_log_odds: float | None = Field(
        default=None, description="What the model says about an average applicant."
    )
    total_log_odds: float | None = Field(
        default=None, description="Baseline plus every contribution. Uncalibrated."
    )
    reasons: list[ReasonCodeOut] = Field(default_factory=list)
    model: ModelIdentity
    latency_ms: float = Field(
        description="Server-side scoring time: frame, preprocessing, model, calibrator, reasons."
    )


class BatchRowOut(BaseModel):
    """One row of a batch: a prediction or the reason it could not be scored.

    Errors are inline rather than fatal because a batch is a job. Failing all
    1000 rows because row 407 has a negative income is a worse answer than 999
    scores and one message, and the caller cannot tell which row was at fault
    from a 422 about the body.
    """

    index: int
    prediction: PredictionOut | None = None
    error: str | None = None


class BatchOut(BaseModel):
    rows: list[BatchRowOut]
    scored: int
    failed: int
    latency_ms: float = Field(description="Total server-side time for the batch.")


class SchemaFieldOut(BaseModel):
    """One input, as the dashboard needs it to build a form field."""

    name: str
    kind: Literal["numeric", "categorical", "date", "text"]
    required: bool
    description: str
    tier: str
    aliases: list[str]
    minimum: float | None = None
    maximum: float | None = None


class SchemaOut(BaseModel):
    """The active model's input contract, in the order it should be presented."""

    run_id: str
    feature_tier: str
    fields: list[SchemaFieldOut]
    engineered: list[str] = Field(
        description="Derived features. Computed from the inputs above; never sent by a client."
    )
    model_features: list[str] = Field(description="What the estimator actually sees.")


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    bundle_loaded: bool
    run_id: str | None = None
    reason: str | None = None


class RunListOut(BaseModel):
    """Every recorded run, newest first, and which one is being served.

    ``runs`` is a list of open dictionaries rather than a declared model, and that
    is deliberate: the entries come from ``registry.json``, whose ``metrics`` block
    holds whatever the run measured. Declaring it here would be a fourth copy of a
    metric list that changes when evaluation does, and the older entries in a real
    registry were written by older code - a strict model would make the history
    unreadable the first time a metric is added.
    """

    active_run_id: str | None
    runs: list[dict[str, Any]]


class DatasetOut(BaseModel):
    """A stored upload, named by its content."""

    dataset_id: str = Field(description="Pass this to POST /api/runs. The file's SHA-256 prefix.")
    size_bytes: int
    #: Whether these exact bytes were already stored. Reported rather than hidden
    #: because an operator who uploads twice by accident should be told the second
    #: one was a no-op, not left wondering which copy a run used.
    existing: bool


class RetrainIn(BaseModel):
    """What a retrain request may specify, and deliberately nothing else.

    A ``dataset_id`` rather than a path or a CSV body. A path would let a caller
    name any file the process can read; a body would put a 64 MiB upload inside a
    request that also has to start a fit. Uploading is a separate, separately
    switched-on route, and this one only refers to what it stored.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(description="From POST /api/datasets.")
    #: A ``Literal`` rather than a validated ``str`` so ``/docs`` renders a choice
    #: instead of a free text box. Spelled out rather than built from
    #: ``SUPPORTED_MODEL_TYPES``, because a Literal cannot be constructed from a
    #: runtime tuple - ``test_retrain_offers_every_supported_model`` is what keeps
    #: the two in step.
    model_type: Literal["logistic_regression", "xgboost"] = "logistic_regression"
    include_lender_priced: bool = Field(
        default=False,
        description=(
            "Include the lender's own pricing columns. Off by default: they encode "
            "the decision being predicted. See ADR 0005."
        ),
    )


class JobOut(BaseModel):
    """A retrain's status, as a poller sees it.

    ``detail`` carries the child's exception text, which is the whole reason this
    endpoint exists - and the reason the retrain routes need a key. It is a
    server-side error message, so it is only ever shown to a caller that was
    authorized to start the job in the first place.
    """

    job_id: str
    status: Literal["running", "succeeded", "failed", "timed_out"]
    dataset: str
    model_type: str
    submitted_at: str
    finished_at: str | None = None
    run_id: str | None = None
    detail: str | None = None
    exit_code: int | None = None


class ErrorOut(BaseModel):
    """Every error body the service produces, and deliberately not more.

    No path, no date range, no traceback: an error body is the one place a service
    volunteers information to an unauthenticated caller. ``request_id`` is how a
    report of "it returned 500" is joined to the traceback in the log, which is
    where the detail belongs.
    """

    detail: str
    request_id: str


def describe_spec(spec: FeatureSpec, run_id: str) -> SchemaOut:
    """The input contract as data, for a client that builds a form from it.

    The same information ``/docs`` carries, in a shape a form generator can use
    without parsing JSON Schema. It exists because the dashboard's
    score-an-applicant panel has to render fields for whichever bundle is active,
    and a hand-written form would be the copy of the column list that drifts.
    """
    required = set(spec.required_raw_inputs)
    fields = []
    for name in spec.raw_inputs:
        column = column_spec(name)
        low, high = _BOUNDS.get(name, (None, None))
        fields.append(
            SchemaFieldOut(
                name=name,
                kind=_KIND_BY_PARSE_KIND[column.parse],  # type: ignore[arg-type]
                required=name in required,
                description=column.description,
                tier=str(column.tier),
                aliases=[key for key in alias_priority(name) if key != name],
                minimum=low,
                maximum=high,
            )
        )
    return SchemaOut(
        run_id=run_id,
        # From `artifacts`, not spelled again here: the tier name appears in run
        # ids and on the model card, and two spellings of it would be two tiers
        # as far as anyone reading a dashboard is concerned.
        feature_tier=feature_tier(spec.include_lender_priced),
        fields=fields,
        engineered=list(spec.engineered),
        model_features=list(spec.model_features),
    )


#: Reused on every route so ``/docs`` shows the error shape rather than FastAPI's
#: default ``{"detail": ...}`` with no request id.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorOut, "description": "Malformed request."},
    422: {"model": ErrorOut, "description": "The body did not match the input contract."},
    503: {"model": ErrorOut, "description": "No model is loaded."},
}
