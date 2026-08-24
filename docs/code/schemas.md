# `src/risk_score/api/schemas.py`

## Purpose

Every request and response body the service accepts or produces - and the applicant model, which is **generated from the bundle's own `FeatureSpec`** rather than written out by hand.

A hand-written applicant model would be a third copy of the column list, after the registry in `features.py` and the spec persisted inside the bundle.
The copy that drifts is always the one furthest from the data.
Generating it means a bundle trained without `fico_range_low` serves an API that does not advertise it, with no edit anywhere.

What that buys, concretely:

- `/docs` lists every accepted field with its type and its description, taken from the column registry - so the API documents itself from the same sentences the data dictionary is built from.
- A missing required input is a 422 naming the field, not a `KeyError` from inside a transformer.
- `extra="forbid"`, so `anual_inc` is a validation error rather than a silently imputed median.
- Source aliases are accepted, because the registry already knows them. A client holding `funded_amnt` need not learn that this project calls it `loan_amnt`.

## Public API

| Name | What it is |
| --- | --- |
| `build_applicant_model` | The generated `Applicant` model for one fitted spec. Called once per bundle load. |
| `describe_spec` | The input contract as plain data, for a client that builds a form from it. |
| `BatchRequest`, `PredictOptions` | The two request bodies that are not generated. |
| `PredictionOut`, `ReasonCodeOut`, `ModelIdentity`, `BatchRowOut`, `BatchOut` | The scoring responses. |
| `SchemaOut`, `SchemaFieldOut`, `HealthOut`, `RunListOut` | The informational responses. |
| `DatasetOut`, `RetrainIn`, `JobOut` | The admin bodies. |
| `ErrorOut` | Every error body the service produces, and deliberately not more. |
| `ERROR_RESPONSES` | Reused on every route so `/docs` shows the real error shape. |

## Inputs and outputs

Reads the column registry (`column_spec`, `alias_priority`) and a `FeatureSpec`.
Produces pydantic model classes and instances.
Opens nothing and computes nothing.

Field order follows `spec.raw_inputs`, which is registry order, so `/docs` and the dashboard form list the loan request before the bureau pull rather than alphabetically.

## Invariants and failure modes

**`extra="forbid"` on the applicant model is the setting that matters most.**
A typo that still returns 200 returns a *different applicant's* risk - the misspelled field is dropped, the real one is imputed to a median, and the response looks entirely normal.
There is no way to detect that from the outside, which is why it has to be refused at the boundary.

**Three parse kinds accept `float | str`, and that union is the whole point of them.**
`term`, `emp_length`, `int_rate` and `revol_util` arrive as `' 36 months'`, `'10+ years'` and `'13.56%'` in one real extract and as plain numbers in the other.
Narrowing them to `float` here would make the API reject the exact strings the training pipeline was built to read, and would put a second parsing policy in this file.
`CanonicalizeFrame` already owns parsing, and owning it once is why both dialects work at all.

**Request bounds are not pipeline bounds.**
The pipeline is lenient by necessity - it ingests a 1.19 GB file nobody curated - while an interactive caller is better served by being told the value is wrong.
Only obviously-impossible values are in `_BOUNDS`; anything arguable is left to the model, because an API that refuses unusual applicants cannot score them.

**Bounds on a `float | str` field are advisory, not enforced.**
`ge` cannot be applied to that union without also rejecting `'10+ years'`.
For those four fields the bound is reported by `/api/schema` as a form hint and left to the parser and the model otherwise.
That asymmetry is deliberate and is the price of accepting both dialects.

**`ModelIdentity` is on every prediction response.**
A score with no model identity cannot be reproduced, and "which model declined this application" is the first question asked about any decline.
Cheap to include, impossible to reconstruct afterwards.

**Batch errors are inline, not fatal.**
Failing all 1000 rows because row 407 has a negative income is a worse answer than 999 scores and one message - and from a 422 about the body, the caller cannot tell which row was at fault.

**`BatchRequest` is a wrapper object, not a bare array.**
A top-level JSON array cannot gain a field without breaking every client.
The item type is patched into the OpenAPI document at startup, because it is not known until a bundle is loaded.

**`RetrainIn` takes a `dataset_id`, never a path and never CSV text.**
A path would let a caller name any file the process can read.
A body would put a 64 MiB upload inside the request that also has to start a fit.
Uploading is a separate, separately switched-on route, and this one only refers to what that route stored.

**`RunListOut.runs` is a list of open dictionaries, on purpose.**
The entries come from `registry.json`, whose `metrics` block holds whatever the run measured.
Declaring it here would be a fourth copy of a metric list that changes whenever evaluation does - and the older entries in a real registry were written by older code, so a strict model would make the history unreadable the first time a metric is added.

**`ErrorOut` is two fields and nothing else.**
No path, no date range, no traceback.
An error body is the one place a service volunteers information to an unauthenticated caller.
`request_id` is how a report of "it returned 500" is joined to the traceback in the log, which is where the detail belongs.

**`JobOut.detail` carries the child's exception text, and that is why the retrain routes need a key.**
It is a server-side error message, so it is only ever shown to a caller that was authorized to start the job in the first place.

**`_KIND_BY_PARSE_KIND` is coarser than `ParseKind` because a form has four kinds of input, not seven.**
Everything numeric is a number box regardless of how the extract happened to spell it.

**`feature_tier` comes from `artifacts.py`, not from a literal here.**
The tier name appears in run ids and on the model card, and two spellings of it would be two tiers as far as anybody reading a dashboard is concerned.

**`RetrainIn.model_type` is a `Literal`, spelled out.**
A `Literal` renders as a choice in `/docs` where a validated `str` renders as a free text box, and a `Literal` cannot be constructed from a runtime tuple.
`tests/test_routes_admin.py::test_retrain_offers_every_supported_model` is what keeps it in step with `SUPPORTED_MODEL_TYPES`.

## What must NOT live here

- **Parsing.** These models accept a dialect; `CanonicalizeFrame` interprets it. A `field_validator` that stripped `'%'` would be the second parsing policy this file exists to avoid.
- **The column list.** It comes from the registry via the spec. Nothing here names a column except `_BOUNDS`, which is about plausibility rather than existence.
- **Business rules.** No model here decides anything; `scoring.py` does.
- **The OpenAPI patch.** `app.py` applies it, because it is a property of the document rather than of the models.
- **A second copy of `ParseKind`.** The three dictionaries here map *from* it and are exhaustive over it, so adding a kind is a mypy error rather than a `KeyError` at startup.

## Related tests

- `tests/test_api.py::test_predict_refuses_an_unknown_field` is the `extra="forbid"` claim, and `test_predict_names_a_missing_required_field` is the 422-naming-the-field claim.
- `test_predict_accepts_both_extract_dialects` and `test_predict_accepts_a_source_alias` are the union and the alias policy.
- `test_schema_describes_every_accepted_input` asserts `/api/schema` covers `spec.raw_inputs` exactly, so a field the form cannot render is a failure rather than a blank input; `test_schema_kinds_are_form_kinds` pins the coarse vocabulary.
- `test_batch_reports_bad_rows_inline` and `test_batch_refuses_an_empty_list` cover the two halves of the batch contract.
- `test_predict_options_reject_an_unknown_query_param` covers `PredictOptions`.
- `test_openapi_describes_the_loaded_model` proves the generated model reaches `/docs`.
- `tests/test_routes_admin.py::test_the_job_body_has_exactly_the_documented_keys` pins `JobOut` against what the runner actually produces.

## Known limits

- **`_BOUNDS` is hand-maintained and keyed by canonical name.** A new numeric column gets no bounds until somebody adds a line, which is a lenient failure rather than a wrong one. Moving the bounds into `ColumnSpec` is the obvious upgrade and was not done because plausibility bounds for an *interactive form* are a serving concern rather than a property of the column.
- **Categorical fields advertise no enumeration.** `/api/schema` reports `kind: "categorical"` without the levels the model was fitted on, so a form renders a text box rather than a select. The levels are in the fitted `OneHotEncoder` and exposing them is a real improvement; it needs a decision about how to present `infrequent_if_exist` groupings.
- **The generated model has no cross-field validation.** Nothing checks that `earliest_cr_line` precedes `issue_d`, for instance. That belongs in the pipeline, which sees the same rule at training time.
- **`RunListOut` gives up type safety for forward compatibility.** A dashboard reading a metric that no longer exists gets `undefined` rather than a clear error. Deliberate, for the reason above.
- **Regenerating the model on every bundle load costs a `create_model` call.** Milliseconds, once per load, and the reason it is not cached is that a retrain must be able to change the contract.
