"""Serving the report artifacts a run wrote, without reading them per request.

A published run directory holds a dozen small files - metrics, two calibration
tables, the threshold cost curve, the vintage breakdown, two PSI tables, the SHAP
summary. The dashboard needs most of them on every page load, and reading and
parsing them per request would mean a page refresh costs a dozen ``read_csv``
calls for bytes that have not changed since the run was published.

So this module does three things:

**Caches by file identity, not by clock.** The cache key is ``(mtime_ns, size)``
per file, so a republished run is picked up on the next request with no TTL to
tune and no stale window. A time-based cache has to choose between serving old
metrics and re-reading files that never change; keying on identity chooses
neither.

**Caches the serialized bytes, not the parsed object.** The response is the same
bytes every time, so ``json.dumps`` runs once per version of a file rather than
once per request. That is most of the cost of these endpoints - the tables are
small but ``json.dumps`` over a 99-row table is still far more work than a
dictionary lookup.

**Answers 304 when the client already has it.** Every payload carries an ETag
derived from the same fingerprint, so a dashboard poll that finds nothing changed
transfers headers and no body.

The other half of the design is what is *not* here. Rows are serialized
columnar - one array per column, via :func:`risk_score.reporting.columnar` -
because the threshold cost table is 99 rows of six numbers and a row-per-object
encoding repeats every key 99 times. It is also the shape a chart wants.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import pandas as pd
from fastapi import HTTPException, Request, Response, status

from risk_score.artifacts import read_active_run_id
from risk_score.reporting import columnar

_log = logging.getLogger(__name__)

#: A run id is a path segment, so it is validated before it is joined to
#: anything. Conservative on purpose: :func:`~risk_score.artifacts.build_run_id`
#: produces only these characters, and everything a traversal needs - ``/``,
#: ``\``, ``.`` - is outside the class. The resolved path is checked for
#: containment as well, because one guard on a filesystem path is one guard too
#: few.
RUN_ID_PATTERN: Final = re.compile(r"\A[A-Za-z0-9_-]{1,120}\Z")

#: How many distinct (run, report, version) payloads to keep. Twenty runs of a
#: dozen artifacts is the retention ceiling, so this holds a full history; the
#: bound exists so a caller enumerating run ids cannot grow the process.
#: ponytail: plain LRU on serialized bytes, no size accounting - the artifacts
#: are kilobytes. If a future report is megabytes, key eviction on bytes instead.
CACHE_ENTRIES: Final = 256

#: Immutable payloads get a year; the active run's get revalidated every time.
#: A run directory never changes once published, so ``?run_id=`` may be cached
#: hard - but "the active run" is a pointer that a retrain moves, and a stale
#: dashboard showing the previous model's metrics under the new model's name is
#: the one caching bug that matters here.
IMMUTABLE_CACHE_CONTROL: Final = "public, max-age=31536000, immutable"
REVALIDATE_CACHE_CONTROL: Final = "no-cache"


@dataclass(frozen=True, slots=True)
class Report:
    """One endpoint's payload: the files it reads, under the keys it reads them as.

    ``parts`` is ordered so the payload's key order is stable, which keeps the
    ETag stable across processes - ``json.dumps`` writes keys in insertion order,
    and a payload that reordered itself per worker would defeat the ETag entirely.
    """

    parts: tuple[tuple[str, str], ...]
    optional: frozenset[str] = frozenset()

    def filenames(self) -> tuple[str, ...]:
        return tuple(filename for _, filename in self.parts)


#: Every report the service will serve, and nothing else. An allowlist rather
#: than "any file in the run directory": the run directory also holds
#: ``model.joblib`` and ``run.log``, and neither is something an unauthenticated
#: caller should be able to name.
REPORTS: Final[dict[str, Report]] = {
    "metrics": Report(parts=(("metrics", "metrics.json"),)),
    "calibration": Report(
        parts=(
            ("validation", "calibration_validation.csv"),
            ("test", "calibration_test.csv"),
        ),
        # A run whose test window held too few positives to bin writes only the
        # validation curve. That is a partial report, not a broken one.
        optional=frozenset({"test"}),
    ),
    "threshold-costs": Report(parts=(("validation", "threshold_costs_validation.csv"),)),
    "vintages": Report(parts=(("vintages", "metrics_by_vintage.csv"),)),
    "drift": Report(
        parts=(
            ("score", "psi_score.csv"),
            ("features", "psi_features.csv"),
        )
    ),
    "shap-summary": Report(parts=(("features", "shap_summary.csv"),)),
    # Written only by `riskscore compare`. Absent on a plain train run, which is
    # a 404 rather than an empty object: "this run has no comparison" and "the
    # two models tied" are different answers.
    "comparison": Report(parts=(("comparison", "comparison.json"),)),
}


def resolve_run_dir(runs_dir: Path, run_id: str | None, reports_dir: Path) -> tuple[Path, bool]:
    """The directory for ``run_id``, or the active run's, plus whether it was named.

    The boolean is what decides the ``Cache-Control`` header: a named run is
    immutable, the active run is a moving pointer.

    Two independent guards on the path. The pattern rejects anything that could
    traverse, and the containment check catches whatever the pattern did not
    anticipate - a symlink inside ``runs/``, most plausibly. A 404 rather than a
    400 for a rejected id, because distinguishing "malformed" from "absent" tells
    a scanner which of its guesses were the right shape.
    """
    explicit = run_id is not None
    if run_id is None:
        active = read_active_run_id(reports_dir)
        if active is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No active run. Train one with `riskscore train`.",
            )
        run_id = active
    elif not RUN_ID_PATTERN.match(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such run.")

    run_dir = (runs_dir / run_id).resolve()
    if not run_dir.is_dir() or runs_dir.resolve() not in run_dir.parents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such run.")
    return run_dir, explicit


def fingerprint(run_dir: Path, report: Report) -> tuple[tuple[str, int, int], ...]:
    """Identity of every file this report reads, as ``(name, mtime_ns, size)``.

    Size as well as mtime because a filesystem with coarse timestamps can rewrite
    a file within one tick, and a report that changed length is the case where
    that would matter. Absent files are simply omitted, so an optional part
    appearing later changes the fingerprint and invalidates the cache.
    """
    entries = []
    for filename in report.filenames():
        path = run_dir / filename
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((filename, stat.st_mtime_ns, stat.st_size))
    return tuple(entries)


def _read_part(path: Path) -> Any:
    """One artifact file as a JSON-safe object, by extension.

    CSV becomes columnar; JSON is passed through. Nothing else is served, so
    there is no third branch to get wrong.
    """
    if path.suffix == ".json":
        loaded: Any = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    return columnar(pd.read_csv(path))


@lru_cache(maxsize=CACHE_ENTRIES)
def _serialize(
    run_dir: Path,
    run_id: str,
    name: str,
    _version: tuple[tuple[str, int, int], ...],
) -> bytes:
    """The payload bytes for one version of one report.

    ``_version`` is unused in the body and load-bearing in the signature: it is
    the file fingerprint, and including it in the cache key is the whole
    invalidation mechanism. Reading it inside the function would defeat the point.

    ``separators`` without spaces because this is machine-read; over the vintage
    and threshold tables the difference is a few percent of the body for zero
    benefit.
    """
    report = REPORTS[name]
    payload: dict[str, Any] = {"run_id": run_id}
    for key, filename in report.parts:
        path = run_dir / filename
        if not path.is_file():
            if key in report.optional:
                payload[key] = None
                continue
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"This run has no {name} report.",
            )
        payload[key] = _read_part(path)
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def clear_cache() -> None:
    """Drop every cached payload. Called after a retrain publishes a new run.

    Not strictly required - the fingerprint would invalidate the changed files on
    its own - but a new run means the previous one's entries will never be asked
    for again, and holding them is the difference between a bounded cache that
    stays warm and one that evicts the run everybody is looking at.
    """
    _serialize.cache_clear()


def report_response(request: Request, name: str, run_id: str | None = None) -> Response:
    """One report, with an ETag, a cache policy, and a 304 when nothing changed.

    A weak ETag rather than a strong one: gzip is applied downstream of this
    response, so two representations of the same payload exist and a strong
    validator would be asserting byte equality that does not hold.
    """
    settings = request.app.state.settings
    run_dir, explicit = resolve_run_dir(settings.runs_dir, run_id, settings.reports_dir)
    version = fingerprint(run_dir, REPORTS[name])
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"This run has no {name} report.",
        )

    etag = f'W/"{hashlib.sha256(repr((name, version)).encode()).hexdigest()[:32]}"'
    headers = {
        "ETag": etag,
        "Cache-Control": IMMUTABLE_CACHE_CONTROL if explicit else REVALIDATE_CACHE_CONTROL,
    }
    # `If-None-Match` may carry a list, and a proxy is entitled to have added
    # `W/` or stripped it, so the comparison is on the opaque part.
    presented = request.headers.get("if-none-match", "")
    if any(etag.strip('W/"') == candidate.strip(' W/"') for candidate in presented.split(",")):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)

    body = _serialize(run_dir, run_dir.name, name, version)
    return Response(content=body, media_type="application/json", headers=headers)
