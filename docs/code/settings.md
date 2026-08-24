# `src/risk_score/api/settings.py`

## Purpose

One typed, validated configuration object for the whole service, and the place where "do not expose this by accident" is enforced.

Environment variables rather than a config file, and validated rather than read with `os.environ.get`, for the same reason `config.py` exists on the training side: in a `getenv` codebase a typo in a variable name is a default silently taking over.
`RISKSCORE_HOST` with one letter wrong is not an error anywhere - the service just binds somewhere else, and nothing says so.

Every default here is the safe one.
The service binds loopback, serves no CORS, accepts no upload, permits no retrain, and requires a key for anything that mutates state.
Each of those has to be turned on deliberately, by name, in a variable this file validates.

## Public API

| Name | What it is |
| --- | --- |
| `Settings` | The frozen `BaseSettings` model. Built once in `create_app` and passed down. |
| `Settings.check_api_key` | Constant-time comparison against the configured key. Fails closed when none is configured. |
| `Settings.mutating_routes_enabled` | `"upload"`, `"retrain"`, `"both"` or `"none"`, for the identity endpoint and the startup log. |
| `Settings.bind_is_public`, `Settings.runs_dir` | The two derived values other modules read. |
| `DEFAULT_HOST`, `DEFAULT_PORT`, `DEFAULT_MAX_BODY_BYTES`, `DEFAULT_MAX_UPLOAD_BYTES`, `DEFAULT_MAX_BATCH_ROWS`, `DEFAULT_JOB_TIMEOUT_SECONDS` | The defaults, each justified where it is defined. |

## Inputs and outputs

Reads `RISKSCORE_*` environment variables and a `.env` file, in pydantic-settings' usual precedence.
Produces one immutable object.
Touches no filesystem: `reports_dir` and `dashboard_dir` are paths, and whether they exist is `app.py`'s and `reports.py`'s question, not this file's.

`frozen=True`, so nothing downstream can reconfigure the service by assigning to a field it happens to hold a reference to.

## Invariants and failure modes

**A misspelled `RISKSCORE_` variable fails the boot.**
`extra="forbid"`, and this is the one place that can reasonably enforce it: anything carrying the prefix was meant for this service, so an unrecognized one is a typo rather than an unrelated variable.

**Binding off loopback takes two decisions, not one typo.**
`host` outside `{127.0.0.1, ::1, localhost}` raises unless `RISKSCORE_ALLOW_PUBLIC_BIND=1`.
A risk model that scores real applicants is not a thing to expose accidentally, and "it defaulted to `0.0.0.0`" is exactly how that happens.

**Uploads or retraining on a public bind with no key refuses to start.**
That combination is an unauthenticated remote fit ending in a pickle write, which is remote code execution with extra steps.
The check is on the *combination* rather than on each flag alone, and that distinction is the whole design: `allow_retrain` with no key on loopback is a reasonable local convenience, while the same two settings on `0.0.0.0` are not.

**`check_api_key` uses `secrets.compare_digest`, not `==`.**
String comparison returns at the first differing byte, and the timing difference is enough to recover a key one byte at a time.
An unconfigured key never matches, so a route that requires one fails closed rather than open - which is why an enabled mutating route with no key configured answers 503 rather than accepting anything.

**Validation runs before the socket is opened.**
`riskscore serve` constructs `Settings` first, so a refused public bind never binds.
pydantic's `ValidationError` subclasses `ValueError`, so it lands in the CLI's existing user-error path and exits 3 with the validator's own message and no traceback.

**`RISKSCORE_ALLOWED_HOSTS=localhost,example.com` works.**
pydantic-settings parses a complex field as JSON, so the obvious comma-separated form would otherwise fail with a JSON decode error naming a column number inside an environment variable - not a message anybody can act on.
The `before` validator accepts both forms and leaves a leading `[` to the JSON parser.
The list itself guards DNS rebinding: without it, a page on an attacker's domain that resolves to `127.0.0.1` can reach a loopback service from the victim's own browser.

**`log_level` is checked against the five real levels.**
`logging` accepts an unknown string by silently doing nothing useful, and a service that logs nothing because of a typo is worse than one that refuses to start.

**`require_bundle` is off by default and on in the container.**
A fresh clone has no bundle and should boot far enough to say so.
A container that cannot score should fail its health check and be replaced, rather than pass liveness while returning 503 to every caller.

## What must NOT live here

- **Anything that reads the filesystem or the network.** A settings object that can fail on a missing directory turns a configuration error and an environment error into the same message.
- **Per-request state.** Reading the environment inside a handler makes behaviour depend on when the handler ran and makes it untestable without `monkeypatch.setenv`.
- **The training configuration.** `RunConfig` in `config.py` owns the model's knobs. A retrain child builds its own; nothing about a fit is configured from a serving variable.
- **Secrets in defaults.** `api_key` defaults to `None`, and the only way to set it is the environment.

## Related tests

`tests/test_api.py`, section 9, plus the guard-order tests in `tests/test_routes_admin.py`.

- `test_public_bind_is_refused_without_the_flag` and `test_public_bind_with_mutating_routes_needs_a_key` are the two exposure guards; `test_mutating_routes_on_loopback_need_no_key` is the case that must keep working, and the reason the check is on the combination.
- `test_require_api_key_distinguishes_unconfigured_from_wrong` pins 503-versus-401, and `test_api_key_check_is_false_when_unconfigured` is the fail-closed rule.
- `test_allowed_hosts_accepts_a_comma_separated_string` and `test_unknown_log_level_is_refused` cover the two validators.
- `tests/test_api.py::test_unknown_host_is_refused` proves the allow-list is actually wired into the middleware stack rather than merely stored.
- `tests/test_cli.py::test_serve_refuses_a_public_bind` asserts the refusal reaches the command line as an exit code and a readable message, with no traceback.

## Known limits

- **The API key is a single shared secret with no rotation and no per-caller identity.** It is an authorization gate on a local demo's mutating routes, not an auth system. Anything multi-tenant wants real tokens and an issuer, which is a different project.
- **`.env` is read from the working directory.** A service started from an unexpected directory silently gets different settings. The startup log states host, port, docs and which mutating routes are on, which is the mitigation.
- **`allowed_hosts` includes `testserver`** so `TestClient` works without every test setting it. That is one extra accepted `Host` value in production, and it is inert unless something resolves that name to the service.
- **No TLS settings.** Terminating TLS is a reverse proxy's job, and a service that reads a private key from an environment variable is a worse idea than one that does not.
- **The size caps are byte counts, not rates.** Nothing here limits request *frequency*; rate limiting belongs in front of the service.
