"""One typed, validated settings object for the whole service.

Environment variables rather than a config file, and validated rather than read
with ``os.environ.get``, for the same reason ``config.py`` exists on the training
side: a typo in a name is a default silently taking over. ``RISKSCORE_HOST`` with
one letter wrong is not an error anywhere in a ``getenv`` codebase - the service
just binds somewhere else.

Every default here is the safe one. The service binds loopback, serves no CORS,
accepts no upload, and requires a key for anything that mutates state; each of
those has to be turned on deliberately, by name, in an environment variable that
this file validates.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Loopback. A risk model that scores real applicants is not a thing to expose by
#: accident, and "it defaulted to 0.0.0.0" is how that happens.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: Any bind address that is not loopback. Refused unless
#: ``RISKSCORE_ALLOW_PUBLIC_BIND=1``, so reaching the network is two decisions
#: rather than one typo.
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

#: 1 MiB. A ``/predict`` body is a few hundred bytes and the largest legitimate
#: request is a 1000-row batch, which fits well inside this. Enforced by
#: middleware so an oversized body is refused before it is read into memory.
DEFAULT_MAX_BODY_BYTES = 1 << 20

#: 64 MiB for an uploaded extract, which is off by default anyway. Large enough
#: for a realistic sample, small enough that a full disk needs many attempts.
DEFAULT_MAX_UPLOAD_BYTES = 64 << 20

#: At most this many rows in one ``/predict/batch``. The limit exists because the
#: work is synchronous: 1000 rows is a few milliseconds of vectorized scoring,
#: while an unbounded batch is a request that holds a worker for a minute.
DEFAULT_MAX_BATCH_ROWS = 1000

#: How long a retrain may run before the job runner kills its process. Twenty
#: minutes is generous for the 1.19 GB extract and finite, which is the point: a
#: hung fit must not occupy the single retrain slot forever.
DEFAULT_JOB_TIMEOUT_SECONDS = 1200.0


class Settings(BaseSettings):
    """Service configuration, read from ``RISKSCORE_*`` environment variables.

    Instantiate once, in ``create_app``, and pass it down. Reading the
    environment from inside a request handler makes behaviour depend on when the
    handler ran, and makes it untestable without ``monkeypatch.setenv``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RISKSCORE_",
        env_file=".env",
        env_file_encoding="utf-8",
        # A misspelled RISKSCORE_ variable is a typo worth failing on, and this
        # is the one place that can tell: anything with the prefix was meant for
        # us.
        extra="forbid",
        frozen=True,
    )

    reports_dir: Path = Field(
        default=Path("reports"),
        description="Report tree the service reads. active_run.json inside it names the bundle.",
    )
    dashboard_dir: Path = Field(
        default=Path("dashboard"),
        description="Static dashboard to mount at /. Skipped when absent.",
    )
    datasets_dir: Path = Field(
        default=Path("data/uploads"),
        description="Where content-addressed uploads land. Only used when uploads are enabled.",
    )

    host: str = DEFAULT_HOST
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    allow_public_bind: bool = Field(
        default=False,
        description="Required to bind anything but loopback. Two decisions, not one typo.",
    )
    allowed_hosts: tuple[str, ...] = Field(
        default=("localhost", "127.0.0.1", "testserver"),
        description=(
            "Host header allow-list. Guards DNS rebinding: a page on an attacker's "
            "domain resolving to 127.0.0.1 can otherwise reach a loopback service "
            "from the victim's own browser."
        ),
    )

    api_key: str | None = Field(
        default=None,
        description="Required in X-API-Key on every mutating route. Absent means those routes 503.",
    )
    allow_upload: bool = Field(
        default=False,
        description=(
            "Enables POST /api/datasets. Remote-triggered execution over "
            "attacker-supplied data ending in a pickle write, so it stays off "
            "unless a local demo asks for it."
        ),
    )
    allow_retrain: bool = Field(
        default=False,
        description="Enables POST /api/runs. Off by default for the same reason.",
    )

    max_body_bytes: int = Field(default=DEFAULT_MAX_BODY_BYTES, gt=0)
    max_upload_bytes: int = Field(default=DEFAULT_MAX_UPLOAD_BYTES, gt=0)
    max_batch_rows: int = Field(default=DEFAULT_MAX_BATCH_ROWS, gt=0)
    job_timeout_seconds: float = Field(default=DEFAULT_JOB_TIMEOUT_SECONDS, gt=0)

    log_level: str = "INFO"
    log_json: bool = False
    docs_enabled: bool = Field(
        default=True,
        description="Serves /docs and /openapi.json. Worth turning off in a public deployment.",
    )
    require_bundle: bool = Field(
        default=False,
        description=(
            "Refuse to start without a loadable bundle. Off by default so a fresh "
            "clone can boot and be told to train; on in a container, where a "
            "process that cannot score should fail its health check and be replaced."
        ),
    )

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, value: Any) -> Any:
        """Accept ``a,b,c`` as well as a JSON list.

        pydantic-settings parses a complex field as JSON, so the obvious
        ``RISKSCORE_ALLOWED_HOSTS=localhost,example.com`` would otherwise fail
        with a JSON decode error naming a column number in an environment
        variable, which is not a message anybody can act on.
        """
        if isinstance(value, str) and not value.strip().startswith("["):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}.")
        return level

    @model_validator(mode="after")
    def _refuse_accidental_exposure(self) -> Settings:
        """The two ways a local demo becomes an open endpoint, both refused here.

        Binding off loopback needs ``allow_public_bind``, and once bound off
        loopback the mutating routes need a key. Checking the combination rather
        than each flag alone is the point: ``allow_retrain`` with no key on
        loopback is a reasonable local convenience, and the same two settings on
        ``0.0.0.0`` are remote code execution.
        """
        if self.host not in _LOOPBACK and not self.allow_public_bind:
            raise ValueError(
                f"host={self.host!r} is not loopback. Set RISKSCORE_ALLOW_PUBLIC_BIND=1 "
                "to bind it deliberately, and set RISKSCORE_API_KEY before you do."
            )
        mutating = self.allow_upload or self.allow_retrain
        if self.host not in _LOOPBACK and mutating and not self.api_key:
            raise ValueError(
                "Uploads or retraining are enabled on a non-loopback bind with no "
                "RISKSCORE_API_KEY. That is an unauthenticated remote fit writing a "
                "pickle; refusing to start."
            )
        return self

    @property
    def bind_is_public(self) -> bool:
        return self.host not in _LOOPBACK

    @property
    def runs_dir(self) -> Path:
        return self.reports_dir / "runs"

    def check_api_key(self, presented: str | None) -> bool:
        """Whether a presented key is the configured one.

        ``compare_digest``, not ``==``: string comparison returns on the first
        differing byte, and the timing difference is enough to recover a key one
        byte at a time. An unconfigured key never matches, so a route that
        requires one fails closed.
        """
        if not self.api_key or presented is None:
            return False
        return secrets.compare_digest(self.api_key, presented)

    def mutating_routes_enabled(self) -> Literal["upload", "retrain", "both", "none"]:
        """What is switched on, for the identity endpoint and the startup log."""
        if self.allow_upload and self.allow_retrain:
            return "both"
        if self.allow_upload:
            return "upload"
        return "retrain" if self.allow_retrain else "none"
