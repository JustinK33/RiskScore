"""The HTTP service: scoring, reports, run history, and retraining.

A subpackage rather than one module because the service has two halves with
different threat models. Everything in ``routes_public`` is read-only and safe to
expose; everything in ``routes_admin`` mutates state and requires an API key. A
file boundary is a weaker guarantee than a type, but it is one a reviewer can
check by reading the imports - which is more than a decorator somebody can forget
to apply offers.

Nothing here is imported by the training code. The dependency direction is
``api -> risk_score`` and never back, which is what lets ``[train]`` and
``[serve]`` be separate extras: the serving container installs no matplotlib, no
xgboost and no pyarrow, and a training box installs no uvicorn.
"""

from __future__ import annotations

from risk_score.api.app import create_app
from risk_score.api.settings import Settings

__all__ = ["Settings", "create_app"]
