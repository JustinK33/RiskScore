"""A parquet cache for the canonicalized extract, keyed by the file's content.

Reading ``data/raw/1/loan.csv`` is the slowest step in a run and the only one
that is pure: 1.19 GB of text through the CSV parser, producing exactly the same
canonicalized frame every time. Twenty seconds of that per run is tolerable once
and absurd while iterating on a threshold or a cost matrix, which is what the
`compare` and `card` commands do.

So the canonical frame is written once as parquet - columnar, typed, and about
an order of magnitude faster to read back - under a key derived from the input's
**content**, not its path or its mtime. Content-addressing is what makes the
cache safe to trust: an extract that was re-exported in place gets a different
key rather than a stale hit, and two copies of the same file at different paths
share one entry.

Three properties keep a cache from becoming a source of wrong answers:

* **A miss is never fatal, and neither is a corrupt hit.** Every failure path
  falls back to reading the CSV. A cache that can break a run is worse than no
  cache, because the failure arrives on a machine where the CSV read worked
  yesterday.
* **The key includes everything that changes the output.** The file's digest, the
  alias overrides from the config, and :data:`CACHE_FORMAT_VERSION` - which is
  bumped by hand whenever canonicalization changes meaning, because a cache
  entry written by older code is a silent regression.
* **Writes are atomic.** Into a temporary name, then ``Path.replace``, so a
  process killed mid-write leaves no truncated parquet for the next run to
  believe.

The ``SchemaReport`` is cached alongside the frame, because it goes into the run
manifest and a cached run whose manifest said "renamed 0 columns" would be
quietly lying about how its own data was built.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import pandas as pd

from risk_score.data_loading import read_raw_loans
from risk_score.schema import SchemaReport

LOGGER = logging.getLogger(__name__)

#: Bumped by hand when canonicalization changes what the frame *means* - a new
#: date format, a changed alias priority, a different dtype. Entries written by
#: older code then miss instead of being trusted.
#:
#: Known limit, and it is a real one: nothing derives this from the code. An edit
#: to ``schema.py`` that forgets to bump it leaves stale entries readable. The
#: mitigations are that the version is one line to change and that
#: ``cache_dir=None`` turns the whole thing off.
CACHE_FORMAT_VERSION = 1

#: Under ``data/`` rather than ``reports/``: this is a derivative of the *input*,
#: and it is already gitignored there.
DEFAULT_CACHE_DIR = Path("data/cache")


def cache_key(*, fingerprint: str, column_aliases: Mapping[str, Any] | None = None) -> str:
    """The name of the cache entry for one (file, alias configuration).

    The alias overrides are part of the key because they change which source
    column becomes which canonical one - two runs of the same extract with
    different aliases produce genuinely different frames.
    """
    aliases = json.dumps(
        {
            str(key): sorted(map(str, value))
            for key, value in sorted((column_aliases or {}).items())
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(aliases.encode("utf-8")).hexdigest()[:8]
    return f"v{CACHE_FORMAT_VERSION}-{fingerprint}-{digest}"


def read_raw_loans_cached(
    path: str | Path,
    *,
    fingerprint: str,
    cache_dir: str | Path | None,
    column_aliases: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, SchemaReport]:
    """:func:`~risk_score.data_loading.read_raw_loans`, memoized on disk.

    ``cache_dir=None`` disables the cache entirely and calls straight through,
    which is the default everywhere except the CLI: a library function that
    writes files into the working directory by default is a surprise.

    ``fingerprint`` is passed in rather than computed here because the caller
    already needs it for the run manifest, and hashing 1.19 GB twice to write
    the same 16 characters into two places is exactly the kind of cost this
    module exists to remove.
    """
    source = Path(path)
    if cache_dir is None:
        return read_raw_loans(source, column_aliases=column_aliases)

    directory = Path(cache_dir)
    key = cache_key(fingerprint=fingerprint, column_aliases=column_aliases)
    frame_path = directory / f"{key}.parquet"
    report_path = directory / f"{key}.json"

    if frame_path.exists() and report_path.exists():
        try:
            frame = pd.read_parquet(frame_path)
            report = _report_from_dict(json.loads(report_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, ImportError) as error:
            # Broad on purpose, and never re-raised: a truncated parquet, a
            # pyarrow that is not installed, and a report written by an older
            # schema all mean the same thing here - read the CSV.
            LOGGER.warning("ignoring unreadable cache entry %s: %s", frame_path.name, error)
        else:
            LOGGER.info("read %d rows from cache %s", len(frame), frame_path.name)
            return frame, report

    frame, report = read_raw_loans(source, column_aliases=column_aliases)
    _write_entry(frame, report, frame_path=frame_path, report_path=report_path)
    return frame, report


def _write_entry(
    frame: pd.DataFrame,
    report: SchemaReport,
    *,
    frame_path: Path,
    report_path: Path,
) -> None:
    """Write both halves of an entry, or neither, and never raise.

    The frame is written first and the report last, so a crash between the two
    leaves an entry that misses on the next run rather than one that hits with
    half its metadata. Both go through a temporary name, because a reader that
    finds a half-written parquet has no way to tell.
    """
    suffix = uuid.uuid4().hex[:8]
    staged_frame = frame_path.with_name(f"{frame_path.name}.tmp-{suffix}")
    staged_report = report_path.with_name(f"{report_path.name}.tmp-{suffix}")
    try:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(staged_frame)
        staged_report.write_text(
            json.dumps(_report_to_dict(report), indent=2, sort_keys=True), encoding="utf-8"
        )
        staged_frame.replace(frame_path)
        staged_report.replace(report_path)
    except (OSError, ValueError, ImportError) as error:
        # A cache that can fail a run is worse than no cache. `pyarrow` lives in
        # the `[train]` extra, so a `[serve]`-only install lands here.
        LOGGER.warning("could not write cache entry %s: %s", frame_path.name, error)
        # `missing_ok` is not enough: if the failure was the `mkdir` - a cache
        # directory whose parent is a regular file, say - then unlinking a path
        # *through* that file raises `NotADirectoryError` and the cache would
        # fail the run after all, which is the one thing it must never do.
        for staged in (staged_frame, staged_report):
            with suppress(OSError):
                staged.unlink(missing_ok=True)


def _report_to_dict(report: SchemaReport) -> dict[str, Any]:
    """Serialized field by field rather than with ``asdict``, so that adding a
    field to :class:`SchemaReport` fails the round-trip test here instead of
    silently disappearing from a cached run's manifest."""
    return {
        "renamed": dict(report.renamed),
        "dropped_duplicate_sources": dict(report.dropped_duplicate_sources),
        "unknown_columns": list(report.unknown_columns),
        "missing_required": list(report.missing_required),
        "date_formats_used": dict(report.date_formats_used),
        "unparseable_dates": dict(report.unparseable_dates),
    }


def _report_from_dict(payload: Mapping[str, Any]) -> SchemaReport:
    """Rebuild a report, restoring the tuples JSON cannot represent.

    Indexing rather than ``.get``: a payload missing a key was written by
    different code, and the ``KeyError`` is caught by the caller and treated as
    a miss.
    """
    return SchemaReport(
        renamed=dict(payload["renamed"]),
        dropped_duplicate_sources=dict(payload["dropped_duplicate_sources"]),
        unknown_columns=tuple(payload["unknown_columns"]),
        missing_required=tuple(payload["missing_required"]),
        date_formats_used=dict(payload["date_formats_used"]),
        unparseable_dates={
            str(key): int(value) for key, value in payload["unparseable_dates"].items()
        },
    )
