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
from fastapi.responses import FileResponse

from risk_score.artifacts import (
    CALIBRATION_FIGURE,
    CALIBRATION_TEST_FILENAME,
    CALIBRATION_VALIDATION_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    MODEL_CARD_FILENAME,
    PSI_FEATURES_FILENAME,
    PSI_SCORE_FILENAME,
    RUN_LOG_FILENAME,
    SHAP_SUMMARY_FILENAME,
    THRESHOLD_COSTS_FILENAME,
    VINTAGE_METRICS_FILENAME,
    read_active_run_id,
)
from risk_score.reporting import COMPARISON_FILENAME, columnar

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
    #: Read from the report root rather than from a run directory. True only for
    #: the comparison, which is a statement about several runs and is published
    #: beside ``registry.json`` for the reasons :mod:`risk_score.reporting`
    #: documents. A root-scoped report has no ``?run_id=``.
    root_scoped: bool = False

    def filenames(self) -> tuple[str, ...]:
        return tuple(filename for _, filename in self.parts)


#: Every report the service will serve, and nothing else. An allowlist rather
#: than "any file in the run directory": the run directory also holds
#: ``model.joblib``, and a pickle is not something an unauthenticated caller
#: should be able to name.
#:
#: The filenames come from :mod:`risk_score.artifacts` rather than being spelled
#: here. They were spelled here, and the copy for ``comparison.json`` disagreed
#: with the writer's - which made that endpoint a permanent 404.
REPORTS: Final[dict[str, Report]] = {
    "metrics": Report(parts=(("metrics", METRICS_FILENAME),)),
    "calibration": Report(
        parts=(
            ("validation", CALIBRATION_VALIDATION_FILENAME),
            ("test", CALIBRATION_TEST_FILENAME),
        ),
        # A run whose test window held too few positives to bin writes only the
        # validation curve. That is a partial report, not a broken one.
        optional=frozenset({"test"}),
    ),
    "threshold-costs": Report(parts=(("validation", THRESHOLD_COSTS_FILENAME),)),
    "vintages": Report(parts=(("vintages", VINTAGE_METRICS_FILENAME),)),
    "drift": Report(
        parts=(
            ("score", PSI_SCORE_FILENAME),
            ("features", PSI_FEATURES_FILENAME),
        )
    ),
    "shap-summary": Report(parts=(("features", SHAP_SUMMARY_FILENAME),)),
    # Written only by `riskscore compare`, and at the report root: a comparison
    # describes several runs and is only complete once all of them are published,
    # so no single run directory owns it. Absent until a comparison has been run,
    # which is a 404 rather than an empty object - "nothing has been compared" and
    # "the two models tied" are different answers.
    "comparison": Report(parts=(("comparison", COMPARISON_FILENAME),), root_scoped=True),
}


#: Every file in a run directory that may be fetched by name, and nothing else.
#: An allowlist rather than a pattern: ``model.joblib`` is in the same directory,
#: and a pickle offered over HTTP is an invitation to unpickle something a
#: stranger chose. The run log is included deliberately - it is the run's own
#: record and contains no request data - and so is the figure, because the
#: dashboard renders it.
ARTIFACT_FILENAMES: Final = frozenset(
    {
        MANIFEST_FILENAME,
        METRICS_FILENAME,
        MODEL_CARD_FILENAME,
        CALIBRATION_VALIDATION_FILENAME,
        CALIBRATION_TEST_FILENAME,
        THRESHOLD_COSTS_FILENAME,
        VINTAGE_METRICS_FILENAME,
        PSI_SCORE_FILENAME,
        PSI_FEATURES_FILENAME,
        SHAP_SUMMARY_FILENAME,
        CALIBRATION_FIGURE,
        RUN_LOG_FILENAME,
    }
)

#: By suffix, because the allowlist is closed: there are five kinds of file in a
#: run directory and no sixth can appear without this dictionary being edited.
#: Charsets are explicit on the text types - a browser guessing an encoding for a
#: model card is how a card with a non-ASCII feature name renders as mojibake.
MEDIA_TYPES: Final = {
    ".json": "application/json",
    ".csv": "text/csv; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".png": "image/png",
    ".log": "text/plain; charset=utf-8",
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

    ``run_id`` is empty for a root-scoped report and the key is then omitted
    rather than filled in with the active run: a comparison names the runs it
    compares inside its own payload, and stamping the currently-served run onto a
    document about several would be a claim the file does not make.

    ``_version`` is unused in the body and load-bearing in the signature: it is
    the file fingerprint, and including it in the cache key is the whole
    invalidation mechanism. Reading it inside the function would defeat the point.

    ``separators`` without spaces because this is machine-read; over the vintage
    and threshold tables the difference is a few percent of the body for zero
    benefit.
    """
    report = REPORTS[name]
    payload: dict[str, Any] = {"run_id": run_id} if run_id else {}
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
    report = REPORTS[name]
    if report.root_scoped:
        # No run to resolve and nothing to name: the file sits beside
        # `registry.json` and the next comparison overwrites it, so it revalidates
        # like the active run rather than caching immutably.
        directory, label, explicit = Path(settings.reports_dir).resolve(), "", False
    else:
        directory, explicit = resolve_run_dir(settings.runs_dir, run_id, settings.reports_dir)
        label = directory.name
    version = fingerprint(directory, report)
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

    body = _serialize(directory, label, name, version)
    return Response(content=body, media_type="application/json", headers=headers)


def artifact_response(request: Request, run_id: str, name: str) -> FileResponse:
    """One named file out of one run directory, or a 404.

    Three guards, and the first one does almost all of the work: ``name`` must be
    a member of :data:`ARTIFACT_FILENAMES`, an exact-match allowlist, so no input
    that is not one of twelve known strings reaches the filesystem at all. The run
    id goes through the same pattern-plus-containment check as a report, and the
    resolved path is checked for containment again - because ``figures/`` means one
    entry legitimately contains a separator, and that is exactly the shape a
    traversal wants to borrow.

    Sent with :class:`~fastapi.responses.FileResponse`, which streams from the
    file rather than reading it into memory, and no ``filename=`` so a figure
    renders in the page instead of downloading.
    """
    settings = request.app.state.settings
    if name not in ARTIFACT_FILENAMES:
        # Same 404 as an unknown run, and deliberately not a 400 listing what is
        # allowed: the list is in the OpenAPI document for anybody entitled to it.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such artifact.")

    run_dir, _ = resolve_run_dir(settings.runs_dir, run_id, settings.reports_dir)
    path = (run_dir / name).resolve()
    if not path.is_file() or run_dir not in path.parents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such artifact.")

    return FileResponse(
        path,
        media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
        # A run directory is immutable once published, so its files may be cached
        # by the year. This is the only place that matters much: the calibration
        # figure is the largest thing the service serves.
        headers={"Cache-Control": IMMUTABLE_CACHE_CONTROL},
    )
