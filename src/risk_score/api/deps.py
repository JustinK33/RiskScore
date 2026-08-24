"""The four things a route handler is allowed to ask the application for.

Every dependency here reads :attr:`fastapi.FastAPI.state` and nothing else. No
handler reads the environment, and no handler opens ``active_run.json``: the
bundle is loaded once by :func:`~risk_score.api.app.create_app` and the settings
object is built once beside it, so a request cannot observe a different
configuration from the one the startup log recorded.

The reason these are dependencies rather than module globals is testability. A
test builds an app with its own settings and its own bundle, and two apps can
exist in one process without fighting over a global.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from risk_score.api.scoring import ScoringService
from risk_score.api.settings import Settings


def get_settings(request: Request) -> Settings:
    """The one settings instance, built in ``create_app``."""
    settings: Settings = request.app.state.settings
    return settings


def get_service(request: Request) -> ScoringService:
    """The loaded bundle, or a 503 explaining that there is not one.

    503 rather than 500: no model loaded is a deployment state, not a bug, and it
    is the state a fresh clone starts in. The detail names the command that fixes
    it, because "service unavailable" with no next step is the least useful
    message an API can send to somebody following the README.
    """
    service: ScoringService | None = request.app.state.service
    if service is None:
        reason: str = request.app.state.load_error or "no active run"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No model is loaded ({reason}). Train one with `riskscore train`.",
        )
    return service


def get_applicant_model(request: Request) -> type[BaseModel]:
    """The pydantic model generated from the active bundle's ``FeatureSpec``.

    Built alongside the service, so it always describes the model that will
    actually score the request. Requesting it without a bundle is the same 503 as
    requesting the service, which is why this goes through :func:`get_service`.
    """
    get_service(request)
    model: type[BaseModel] = request.app.state.applicant_model
    return model


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Gate for every route that changes state on the server.

    Two distinct failures, deliberately distinguished:

    * **503** - no key is configured, so the route cannot be authorized at all.
      Answering 401 here would invite a client to keep guessing at a door that is
      bolted shut.
    * **401** - a key is configured and this is not it.

    The comparison is constant-time, in :meth:`Settings.check_api_key`. A header
    rather than a query parameter because query strings are logged by proxies.
    """
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This route requires RISKSCORE_API_KEY to be configured.",
        )
    if not settings.check_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


SettingsDep = Annotated[Settings, Depends(get_settings)]
ServiceDep = Annotated[ScoringService, Depends(get_service)]
ApplicantModelDep = Annotated[type[BaseModel], Depends(get_applicant_model)]
