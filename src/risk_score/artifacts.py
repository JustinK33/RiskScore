"""The scoring bundle: one artifact that can score an applicant by itself.

``pipeline.py`` used to ``joblib.dump`` a fitted estimator, and nothing ever
loaded it. That is not merely an unused file - it is an artifact that *cannot*
be used, because scoring needs four things and the file held one:

* the fitted pipeline (preprocessing plus estimator),
* the calibrator, without which the probability is not the number the cost
  matrix was applied to,
* the **decision threshold**, which lived in a JSON file beside the pickle and
  could therefore be paired with the wrong model by anyone loading them
  separately,
* the ``FeatureSpec``, without which a request payload cannot be validated
  against the columns the model was actually fitted on.

:class:`ScoringBundle` holds all four and is the only object this project
pickles. One pickle, one version number, one thing to load at boot.

Beside it sits ``manifest.json``, a plain-JSON mirror of the metadata. The point
of the duplication is that identity questions - which dataset, which commit,
which windows, how many rows - are answerable by ``cat`` and by any language,
without unpickling anything and therefore without importing this package or
trusting the file's contents. ``registry.json`` and ``active_run.json`` are
readable the same way.

**Writes are atomic at the granularity of a run.** Everything is written into
``runs/.staging-<uuid>/``, fsynced, and then moved into place with
``os.replace``, which is atomic on the same filesystem. A run directory
therefore never exists in a half-written state under its real name, so a
concurrent reader can never pair new metrics with an old calibration curve, and
a crashed run leaves a droppable ``.staging-`` directory rather than a plausible
looking one. ``prune_staging`` removes those.

The registry and the active-run pointer are shared mutable state, so they are
updated under a lock built from ``O_CREAT | O_EXCL`` - one syscall, no new
dependency, and correct across processes on a local filesystem. It is *not*
correct on NFS; that limit is documented rather than papered over, because the
alternative is a lock service this project has no use for.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from risk_score.transformers import FeatureSpec

#: Incremented whenever the *shape* of a pickled :class:`ScoringBundle` changes -
#: a renamed field, a removed one, a changed meaning. ``load_bundle`` refuses a
#: mismatch rather than unpickling into an attribute error three calls later, at
#: which point the traceback points at the API and not at the stale artifact.
#:
#: Note what this does not protect: pickle also encodes the *module path* of
#: every class inside the bundle. Moving ``risk_score.transformers`` breaks every
#: bundle ever written, silently at import time. That is the standing cost of
#: putting feature engineering inside the pipeline (ADR 0002), and the mitigation
#: is a rule rather than code: these module paths do not move.
BUNDLE_SCHEMA_VERSION = 1

#: File name of the one pickle in a run directory.
BUNDLE_FILENAME = "model.joblib"

#: Plain-JSON mirror of :class:`RunMetadata`, next to the pickle.
MANIFEST_FILENAME = "manifest.json"

#: Append-only index of every run, and the pointer the service reads at boot.
REGISTRY_FILENAME = "registry.json"
ACTIVE_RUN_FILENAME = "active_run.json"

#: Prefix for a run directory that is still being written. Leading dot so it
#: sorts and globs apart from real run ids, which start with a digit.
STAGING_PREFIX = ".staging-"

#: How many run directories `prune_runs` keeps by default. Each is a few hundred
#: kilobytes plus figures, so the cap is about keeping the tree readable rather
#: than about disk.
DEFAULT_RETENTION = 20

#: How long to wait for the registry lock before giving up. A registry update is
#: two small JSON writes, so ten seconds is already pathological and failing
#: loudly beats blocking a request thread.
LOCK_TIMEOUT_SECONDS = 10.0

#: How old a lock file must be before it is treated as abandoned and removed.
#: Strictly greater than the wait timeout, and that ordering is the whole design:
#: if the two were equal, a lock held legitimately for slightly too long would be
#: *stolen* rather than waited out, and the timeout error would be unreachable.
#: A stale timeout is nonetheless mandatory, because the process that would have
#: released the lock is the one that died - SIGKILL leaves no chance to clean up,
#: and without this the first crash wedges every later run.
LOCK_STALE_SECONDS = 60.0
LOCK_POLL_SECONDS = 0.05

#: The metrics lifted into the registry, so the run-history table renders from
#: one file instead of from N metrics files. One tuple rather than two lists,
#: because `register_run` and `rebuild_registry` disagreeing would mean a
#: rebuilt index quietly showing different columns than a freshly written one.
HEADLINE_METRICS = ("auc_roc", "brier_score", "expected_calibration_error")

#: Libraries whose version changes the numbers a bundle produces, so a run that
#: cannot be reproduced can at least be explained.
_TRACKED_LIBRARIES = ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "joblib")


# --- identity ------------------------------------------------------------------


def git_commit(*, cwd: str | Path | None = None) -> str:
    """The short commit the run was produced from, or ``"unknown"``.

    Never raises. A model built from a tarball, from a Docker layer, or from a
    directory that is not a repository is a legitimate model; refusing to train
    because ``git`` is absent would be absurd. The manifest says ``unknown`` and
    the reader draws the obvious conclusion.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


def library_versions() -> dict[str, str]:
    """Versions of the libraries whose behaviour the numbers depend on.

    Read through ``importlib.metadata`` rather than ``module.__version__``, so a
    library that is declared but not importable - ``xgboost`` without
    ``libomp``, the exact situation on this machine - still reports the version
    that *would* be used.
    """
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {"python": platform.python_version()}
    for name in _TRACKED_LIBRARIES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def dataset_fingerprint(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """First 16 hex characters of the file's SHA-256.

    Sixteen characters, not sixty-four: this identifies which extract a run
    read, and 64 bits of a cryptographic digest is far past the point where two
    CSVs on one laptop collide. The full digest in every manifest, registry
    entry, and model card is noise.

    Read in chunks because the real extract is 1.19 GB and hashing it by
    ``read_bytes()`` would cost more memory than training does.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def build_run_id(
    *,
    model_type: str,
    include_lender_priced: bool,
    created_at: datetime | None = None,
    commit: str | None = None,
) -> str:
    """``20260824T101530123Z-logistic_regression-origination_only-02e3a94``.

    Sortable first, because the most common question about a run directory is
    "which is the newest", and a lexical sort answers it with no parsing. Then
    the two facts that distinguish otherwise identical runs - which model, which
    feature tier - and the commit, so a surprising number can be traced to code.

    Basic ISO format: colons are legal in POSIX filenames and a catastrophe on
    Windows and in URLs, and this id appears in both.

    Milliseconds, not seconds, and that is not future-proofing: with the parquet
    cache warm a synthetic run finishes in well under a second, so two runs of
    the same model and tier landing in the same second is ordinary rather than
    pathological - and :func:`staged_run` refuses to overwrite an existing id.
    ``%f`` is microseconds and always six digits; the slice keeps three, which is
    finer than any code path that could produce two ids.
    """
    moment = (created_at or datetime.now(UTC)).astimezone(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%S") + f"{moment.microsecond // 1000:03d}Z"
    return f"{stamp}-{model_type}-{feature_tier(include_lender_priced)}-{commit or git_commit()}"


def feature_tier(include_lender_priced: bool) -> str:
    """The tier name that appears in run ids, manifests, and the model card.

    A boolean in a filename reads as ``True``/``False`` and tells a reader
    nothing about which is the safe one. See ADR 0005.
    """
    return "with_lender_priced" if include_lender_priced else "origination_only"


# --- metadata ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Everything needed to say what a bundle is, without loading the model.

    Typed rather than a bare dict, because the service reads ``run_id`` and
    ``model_type`` on the hot path and a typo there is a 500 at request time.
    The open-ended parts - the embargo counts, the split summary, the feature
    and leakage summaries - stay dicts, because they are *reported*, not
    branched on, and enumerating them here would mean editing this class every
    time a report gains a number.
    """

    run_id: str
    created_at: str
    model_type: str
    feature_tier: str
    git_commit: str
    dataset_path: str
    dataset_sha256: str
    dataset_bytes: int
    target_definition: str
    #: Row counts at each filter stage. A run that discarded 40% of its input
    #: must not look identical to one that discarded none.
    rows: dict[str, int] = field(default_factory=dict)
    split_windows: dict[str, str] = field(default_factory=dict)
    embargo: dict[str, Any] = field(default_factory=dict)
    cost_matrix: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)
    bundle_schema_version: int = BUNDLE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> RunMetadata:
        """Rebuild from a manifest, ignoring keys this version does not know.

        Ignoring rather than raising, so a manifest written by a *newer* version
        of this project can still be listed in ``riskscore runs``. Loading the
        pickle is where a version mismatch has to be fatal; reading identity out
        of JSON does not.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in mapping.items() if key in known})


# --- the bundle ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoringBundle:
    """A fitted model, its calibrator, its threshold, and its input contract.

    The threshold lives here rather than in ``metrics.json`` on purpose. It is
    part of the decision rule, it was selected against *these* calibrated
    scores, and any arrangement that allows loading the model without it allows
    serving a model with someone else's cut-off.
    """

    pipeline: Any
    calibrator: Any
    threshold: float
    feature_spec: FeatureSpec
    metadata: RunMetadata
    #: A small sample of *transformed* training rows for SHAP to explain
    #: against, so an explainer can be built at boot with no training data on
    #: disk. Populated in Phase 5; ``None`` means reason codes are unavailable,
    #: which the API reports rather than guessing.
    shap_background: Any = None

    def predict_probability(self, applicants: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Calibrated probability of default, one per row.

        Through the calibrator, always. The bare ``pipeline`` is kept for
        inspection and for the one uncalibrated number the report needs; it is
        not the scoring path, and having exactly one method here is what stops
        a caller from choosing.
        """
        probabilities = self.calibrator.predict_proba(applicants)[:, 1]
        return np.asarray(probabilities, dtype=np.float64)

    def decide(self, probabilities: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """``True`` where the applicant is approved.

        Approval is ``p < threshold``, strictly. The boundary case has to land
        somewhere and rejecting it matches how the cost table was built, so the
        reported cost is the cost of this rule and not of one point beside it.
        """
        return np.asarray(probabilities, dtype=np.float64) < self.threshold


def save_bundle(bundle: ScoringBundle, run_dir: str | Path) -> Path:
    """Write the pickle and its JSON mirror into ``run_dir``. Returns the path.

    Intended to be called inside :func:`staged_run`, which is what makes the
    write atomic. Calling it directly on a live run directory overwrites a
    bundle in place, and a reader can observe the intermediate state.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, directory / BUNDLE_FILENAME)
    # Insertion order, deliberately not sorted. `rows` is the filter pipeline in
    # the order it ran and `split_windows` is train, validation, test - so
    # alphabetising them puts `closed` before `raw` and `test` first, and anything
    # re-rendering the card from this file (`riskscore card`) prints the stages in
    # an order that reads as nonsense. The order is produced by code, so it is
    # stable across runs either way and diffs stay clean.
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(bundle.metadata.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return directory / BUNDLE_FILENAME


def load_bundle(run_dir: str | Path) -> ScoringBundle:
    """Read a bundle, refusing one written by an incompatible version.

    The version check is before any use of the object, so the error names the
    two versions and the directory instead of surfacing as a missing attribute
    somewhere inside a request handler.
    """
    path = Path(run_dir) / BUNDLE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No scoring bundle at {path}. A run directory must contain "
            f"{BUNDLE_FILENAME}; `riskscore runs` lists the ones that do."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, ScoringBundle):
        raise TypeError(
            f"{path} unpickled to {type(bundle).__name__}, not a ScoringBundle. "
            "Artifacts written before the bundle existed held a bare estimator "
            "and cannot be served: retrain."
        )
    found = bundle.metadata.bundle_schema_version
    if found != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"{path} was written with bundle schema version {found}; this build "
            f"reads version {BUNDLE_SCHEMA_VERSION}. Retrain, or check out the "
            "commit named in the manifest's `git_commit`."
        )
    return bundle


def read_manifest(run_dir: str | Path) -> RunMetadata:
    """Identity without unpickling - what the run history and the CLI read."""
    path = Path(run_dir) / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No manifest at {path}.")
    return RunMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --- atomic run directories ----------------------------------------------------


def _fsync_tree(directory: Path) -> None:
    """Flush every file in ``directory`` and the directory entries themselves.

    Without this, ``os.replace`` can publish a directory whose *name* is durable
    while its contents are still in the page cache, so a crash leaves a run that
    exists and is empty. Directory fsync needs ``O_RDONLY`` on the directory
    itself, which is POSIX-only - hence the guarded call.
    """
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in [*sorted(p for p in directory.rglob("*") if p.is_dir()), directory]:
        try:
            handle_fd = os.open(path, os.O_RDONLY)
        except OSError:  # pragma: no cover - Windows has no directory fd
            continue
        try:
            os.fsync(handle_fd)
        finally:
            os.close(handle_fd)


@contextmanager
def staged_run(root: str | Path, run_id: str) -> Iterator[Path]:
    """Yield a scratch directory that becomes ``root/runs/<run_id>`` on success.

    The whole run - bundle, manifest, metrics, curves, figures, log - is written
    into the scratch directory, fsynced, and then moved into place in one
    ``os.replace``. So the run directory is either absent or complete, and a
    dashboard polling it can never read new metrics against an old calibration
    curve.

    On any exception the scratch directory is removed and the exception
    propagates. A failed run leaves nothing to mistake for a result.
    """
    runs = Path(root) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    destination = runs / run_id
    if destination.exists():
        raise FileExistsError(
            f"Run {run_id} already exists at {destination}. Run ids embed a "
            "millisecond-resolution timestamp, so this means two runs of the "
            "same model and tier started within one millisecond."
        )
    # A uuid rather than the run id, so two concurrent runs cannot collide on
    # the scratch directory either.
    staging = runs / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
        _fsync_tree(staging)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prune_staging(root: str | Path, *, older_than_seconds: float = 3600.0) -> list[str]:
    """Remove abandoned scratch directories. Returns the names removed.

    ``staged_run`` cleans up after any exception, but not after ``SIGKILL`` - an
    OOM kill during a fit on the real extract, which is precisely the scenario
    this project's memory work is about. Age-gated so a concurrently running fit
    is never deleted out from under itself.
    """
    runs = Path(root) / "runs"
    if not runs.exists():
        return []
    cutoff = time.time() - older_than_seconds
    removed: list[str] = []
    for path in sorted(runs.glob(f"{STAGING_PREFIX}*")):
        if path.is_dir() and path.stat().st_mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    return removed


# --- the registry and the active pointer ---------------------------------------


@contextmanager
def _registry_lock(
    root: Path,
    *,
    timeout: float = LOCK_TIMEOUT_SECONDS,
    stale_after: float = LOCK_STALE_SECONDS,
) -> Iterator[None]:
    """Hold an exclusive lock on the registry files, or raise.

    ``O_CREAT | O_EXCL`` is a single atomic syscall on a local filesystem, which
    is the whole mechanism: no new dependency, no daemon, no partial-write
    window. The pid is written into the file purely so a human debugging a stuck
    lock knows who to look for.

    A lock file older than ``stale_after`` is treated as abandoned and removed,
    because the process that would have cleaned it up is the one that died.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "registry.lock"
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - lock_path.stat().st_mtime if lock_path.exists() else 0.0
            if age > stale_after:
                # Stale: the holder cannot still be doing two small JSON writes.
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Could not acquire {lock_path} within {timeout:g}s. Another "
                    "run is updating the registry; if none is, delete the file."
                ) from None
            time.sleep(LOCK_POLL_SECONDS)
            continue
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode())
        finally:
            os.close(descriptor)
        try:
            yield
        finally:
            lock_path.unlink(missing_ok=True)
        return


def write_json_atomic(path: Path, payload: Any) -> None:
    """Replace ``path`` in one step, never truncating the reader's view.

    A plain ``write_text`` on ``active_run.json`` has a window in which the file
    is zero bytes, and the service reads that file at boot. Temp file plus
    ``os.replace`` removes the window; the temp file is a sibling so the rename
    stays within one filesystem.

    Public because the same window exists for every root-level report the service
    serves - ``comparison.json`` in particular, which is rewritten in place by
    each ``riskscore compare`` while a dashboard may be polling it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    # Insertion order for the same reason as the manifest: `comparison.json` lists
    # its metrics in the order they are displayed, and the writer builds every
    # payload here in code, so the output is deterministic without sorting.
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_registry(root: str | Path) -> list[dict[str, Any]]:
    """Every recorded run, oldest first. An absent or corrupt registry is empty.

    Corruption is tolerated rather than raised on because the registry is an
    *index*: the run directories are the data, and refusing to list them
    because one JSON file got truncated would turn a cosmetic problem into an
    outage. :func:`rebuild_registry` reconstructs it from the manifests.
    """
    path = Path(root) / REGISTRY_FILENAME
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [entry for entry in loaded if isinstance(entry, dict)]


def register_run(
    root: str | Path,
    *,
    metadata: RunMetadata,
    metrics: Mapping[str, Any] | None = None,
    make_active: bool = True,
) -> dict[str, Any]:
    """Add a run to the registry and, by default, make it the active one.

    The entry carries the headline metrics as well as the identity, so the run
    history table renders from one file rather than from N manifests plus N
    metrics files - which at twenty runs is forty reads per page load.

    Both files are written under one lock, so a reader never sees an
    ``active_run.json`` pointing at a run the registry does not list.
    """
    root_path = Path(root)
    entry: dict[str, Any] = {
        "run_id": metadata.run_id,
        "created_at": metadata.created_at,
        "model_type": metadata.model_type,
        "feature_tier": metadata.feature_tier,
        "git_commit": metadata.git_commit,
        "dataset_sha256": metadata.dataset_sha256,
        "rows": dict(metadata.rows),
        "metrics": dict(metrics or {}),
    }
    with _registry_lock(root_path):
        existing = [
            item for item in read_registry(root_path) if item.get("run_id") != entry["run_id"]
        ]
        write_json_atomic(root_path / REGISTRY_FILENAME, [*existing, entry])
        if make_active:
            write_json_atomic(
                root_path / ACTIVE_RUN_FILENAME,
                {"run_id": metadata.run_id, "activated_at": now_iso()},
            )
    return entry


def set_active_run(root: str | Path, run_id: str) -> None:
    """Point the service at a different run - the rollback path.

    Refuses a run id the registry does not know, because the failure mode being
    prevented is a typo that leaves the service unable to boot at all.
    """
    root_path = Path(root)
    with _registry_lock(root_path):
        known = {entry.get("run_id") for entry in read_registry(root_path)}
        if run_id not in known:
            raise ValueError(
                f"Unknown run {run_id!r}. Known runs: {sorted(str(item) for item in known)}."
            )
        write_json_atomic(
            root_path / ACTIVE_RUN_FILENAME,
            {"run_id": run_id, "activated_at": now_iso()},
        )


def read_active_run_id(root: str | Path) -> str | None:
    """The run the service should load, or ``None`` if there is not one yet."""
    path = Path(root) / ACTIVE_RUN_FILENAME
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    run_id = loaded.get("run_id") if isinstance(loaded, dict) else None
    return str(run_id) if run_id else None


def load_active_bundle(root: str | Path) -> ScoringBundle:
    """Load whatever ``active_run.json`` points at. The service's boot path.

    The two failure modes get different errors on purpose: no pointer means
    nothing has been trained, and a pointer to a missing directory means a run
    was deleted out from under it. The fix differs.
    """
    root_path = Path(root)
    run_id = read_active_run_id(root_path)
    if run_id is None:
        raise FileNotFoundError(
            f"No active run in {root_path / ACTIVE_RUN_FILENAME}. Train one first: "
            "`riskscore train`."
        )
    directory = root_path / "runs" / run_id
    if not directory.exists():
        raise FileNotFoundError(
            f"Active run {run_id} is registered but its directory {directory} is "
            "gone. Point at another with `riskscore runs --activate <run_id>`."
        )
    return load_bundle(directory)


def rebuild_registry(root: str | Path) -> list[dict[str, Any]]:
    """Reconstruct the registry from the manifests on disk.

    The run directories are the source of truth and the registry is a cache of
    them, so this is the repair for a truncated index, for runs copied in from
    elsewhere, and for a tree whose registry predates a field.
    """
    root_path = Path(root)
    runs = root_path / "runs"
    entries: list[dict[str, Any]] = []
    for directory in sorted(p for p in runs.glob("*") if p.is_dir()):
        if directory.name.startswith(STAGING_PREFIX):
            continue
        try:
            metadata = read_manifest(directory)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        metrics_path = directory / "metrics.json"
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            try:
                loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics = {key: loaded[key] for key in HEADLINE_METRICS if key in loaded}
            except json.JSONDecodeError:
                metrics = {}
        entries.append(
            {
                "run_id": metadata.run_id,
                "created_at": metadata.created_at,
                "model_type": metadata.model_type,
                "feature_tier": metadata.feature_tier,
                "git_commit": metadata.git_commit,
                "dataset_sha256": metadata.dataset_sha256,
                "rows": dict(metadata.rows),
                "metrics": metrics,
            }
        )
    entries.sort(key=lambda entry: str(entry.get("created_at", "")))
    with _registry_lock(root_path):
        write_json_atomic(root_path / REGISTRY_FILENAME, entries)
    return entries


def prune_runs(
    root: str | Path,
    *,
    keep: int = DEFAULT_RETENTION,
    protected: Sequence[str] = (),
) -> list[str]:
    """Delete all but the newest ``keep`` runs. Returns the ids removed.

    The active run is never deleted regardless of age, because the alternative
    is retention policy taking the service down. Deletion is best-effort per
    directory: a locked file on one run must not stop the others being cleaned.
    """
    if keep < 1:
        raise ValueError(f"`keep` must be at least 1; got {keep}.")
    root_path = Path(root)
    entries = read_registry(root_path) or rebuild_registry(root_path)
    entries.sort(key=lambda entry: str(entry.get("created_at", "")))
    never_delete = {*protected}
    active = read_active_run_id(root_path)
    if active:
        never_delete.add(active)

    doomed = [
        str(entry["run_id"])
        for entry in entries[: max(0, len(entries) - keep)]
        if entry.get("run_id") and str(entry["run_id"]) not in never_delete
    ]
    for run_id in doomed:
        shutil.rmtree(root_path / "runs" / run_id, ignore_errors=True)
    if doomed:
        survivors = [entry for entry in entries if str(entry.get("run_id")) not in set(doomed)]
        with _registry_lock(root_path):
            write_json_atomic(root_path / REGISTRY_FILENAME, survivors)
    return doomed


def now_iso(when: datetime | None = None) -> str:
    """UTC, millisecond resolution, ``Z`` suffix. One spelling of a timestamp.

    Public because the pipeline stamps ``created_at`` with it, and the registry
    sorts runs by that string: two spellings of the same instant - extended here,
    basic in the run id - would sort into two separate groups.

    Millisecond resolution for the same reason :func:`build_run_id` uses it - two
    runs can share a second - and because the registry breaks ties by this
    string, so a coarser stamp would make "which run is newest" ambiguous
    exactly when two runs were published back to back.
    """
    moment = (when or datetime.now(UTC)).astimezone(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"
