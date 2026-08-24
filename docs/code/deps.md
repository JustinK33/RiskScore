# `src/risk_score/api/deps.py`

## Purpose

The four things a route handler is allowed to ask the application for, and the one thing it has to prove before it mutates anything.

Every dependency here reads `app.state` and nothing else.
No handler reads the environment, and no handler opens `active_run.json`: the bundle is loaded once by `create_app` and the settings object is built once beside it, so a request cannot observe a different configuration from the one the startup log recorded.

They are dependencies rather than module globals for testability.
A test builds an app with its own settings and its own bundle, and two apps can exist in one process without fighting over a global - which is exactly what `tests/test_api.py` does when it checks the no-bundle behaviour against a temporary directory while the session's real service is still loaded.

## Public API

| Name | What it is |
| --- | --- |
| `get_settings` | The one `Settings` instance. |
| `get_service` | The loaded `ScoringService`, or a 503 explaining why there is not one. |
| `get_applicant_model` | The pydantic model generated from the active bundle's `FeatureSpec`. |
| `require_api_key` | The gate on every state-changing route. Returns nothing; raises or permits. |
| `SettingsDep`, `ServiceDep`, `ApplicantModelDep` | `Annotated` aliases, so a handler signature reads as a type rather than as a `Depends` call. |

## Inputs and outputs

Reads `request.app.state.settings`, `.service`, `.applicant_model`, and `.load_error`.
Reads the `X-API-Key` request header.
Returns objects the app already built; constructs nothing except exceptions.

## Invariants and failure modes

**No model loaded is a 503, not a 500.**
It is a deployment state rather than a bug, and it is the state a fresh clone starts in.
The detail names the command that fixes it, because "service unavailable" with no next step is the least useful message an API can send to somebody following the README.
The recorded `load_error` is included, so a corrupt bundle and an absent one are distinguishable from the response alone.

**Requesting the applicant model without a bundle is the same 503.**
`get_applicant_model` calls `get_service` first for exactly that reason.
Without it the handler would receive `None` and fail later with an `AttributeError` and a 500 - the same underlying condition reported as a bug.

**An unauthorized mutating route gets two different answers, and the difference matters.**
503 means no key is configured, so the route cannot be authorized at all; answering 401 there would invite a client to keep guessing at a door that is bolted shut.
401 means a key is configured and the presented one is wrong.

**The key comparison is constant-time**, in `Settings.check_api_key`.
See [settings.md](settings.md) for why `==` is not acceptable here.

**A header, not a query parameter.**
Query strings are logged by proxies, by access logs, and by browser history.

**The 401 carries `WWW-Authenticate`.**
It is the header that tells a generic HTTP client which scheme to retry with, and omitting it makes the failure look like a server problem to anything that follows the spec.

## What must NOT live here

- **Reading the environment.** `Settings` does that once, in `create_app`.
- **Loading a bundle.** `app.load_service` owns it, so a reload is one code path rather than one per dependency.
- **The feature-flag check.** Whether uploads or retraining are enabled at all is checked in `routes_admin.py`, *before* the key, so the guard order is visible next to the routes it guards. Both halves of that order are asserted by `test_the_flag_is_checked_before_the_key`.
- **Per-request scoring state.** A dependency that built something per request would be doing work on the hot path, which is the thing `scoring.py` is organized to avoid.

## Related tests

- `tests/test_api.py::test_require_api_key_distinguishes_unconfigured_from_wrong` is the 503-versus-401 rule.
- `tests/test_api.py::test_healthz_is_ok_and_readyz_is_503_with_no_bundle` and `test_reports_are_503_with_no_active_run` cover `get_service`'s refusal from both directions - liveness still answers, everything that needs a model does not.
- `tests/test_routes_admin.py::test_an_enabled_route_still_needs_the_key` and `test_an_enabled_route_with_no_key_configured_is_503` are the gate, parametrized over a missing header and a wrong one.
- `tests/test_routes_admin.py::test_the_flag_is_checked_before_the_key` pins the guard order this file deliberately does not own.

## Known limits

- **One key, no identity.** `require_api_key` answers "is this the configured secret", not "who is this". There is no per-caller audit trail, and there cannot be one without an issuer.
- **No rate limiting on the failed-key path.** Nothing here slows an attacker down between attempts; the constant-time compare removes the timing oracle, not the guessing.
- **No scopes.** Every mutating route takes the same key, so a caller permitted to upload is permitted to retrain. Splitting them would need two secrets and a reason to have two.
- **`get_settings` returns the frozen object by reference.** Nothing can mutate it, which is why that is safe, but it also means a handler cannot override a setting for one request - deliberately.
