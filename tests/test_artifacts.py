"""Tests for the scoring bundle, atomic run directories, and the registry.

The previous artifact was a bare ``joblib.dump`` of an estimator that nothing
loaded, so there was nothing here to test. What is tested now is mostly *not*
the happy path: a half-written run must be unreachable, a stale lock must not
wedge the next run, and retention must not delete the run the service is
serving. Those are the failures that cost a day rather than a minute.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from risk_score.artifacts import (
    ACTIVE_RUN_FILENAME,
    BUNDLE_FILENAME,
    BUNDLE_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    REGISTRY_FILENAME,
    STAGING_PREFIX,
    RunMetadata,
    ScoringBundle,
    build_run_id,
    dataset_fingerprint,
    feature_tier,
    git_commit,
    library_versions,
    load_active_bundle,
    load_bundle,
    prune_runs,
    prune_staging,
    read_active_run_id,
    read_manifest,
    read_registry,
    rebuild_registry,
    register_run,
    save_bundle,
    set_active_run,
    staged_run,
)
from risk_score.transformers import build_feature_spec

RAW_COLUMNS = ("loan_amnt", "annual_inc", "dti", "term", "issue_d", "home_ownership")


class StubModel:
    """A picklable stand-in for a fitted pipeline.

    Defined at module level because that is what makes it picklable at all -
    the same constraint the real bundle lives under, which is why ADR 0002's
    "these module paths do not move" rule exists.
    """

    def __init__(self, probability: float = 0.25) -> None:
        self.probability = probability

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray[Any, Any]:
        column = np.full(len(frame), self.probability)
        return np.column_stack([1.0 - column, column])


def make_metadata(**overrides: Any) -> RunMetadata:
    defaults: dict[str, Any] = {
        "run_id": "20260101T000000Z-logistic_regression-origination_only-abc1234",
        "created_at": "2026-01-01T00:00:00Z",
        "model_type": "logistic_regression",
        "feature_tier": "origination_only",
        "git_commit": "abc1234",
        "dataset_path": "data/raw/1/loan.csv",
        "dataset_sha256": "0123456789abcdef",
        "dataset_bytes": 1234,
        "target_definition": "Charged Off or Default -> 1, Fully Paid -> 0",
        "rows": {"train": 10, "validation": 5, "test": 5},
    }
    return RunMetadata(**{**defaults, **overrides})


def make_bundle(metadata: RunMetadata | None = None, **overrides: Any) -> ScoringBundle:
    return ScoringBundle(
        pipeline=StubModel(0.40),
        calibrator=StubModel(0.25),
        threshold=0.30,
        feature_spec=build_feature_spec(RAW_COLUMNS),
        metadata=metadata or make_metadata(**overrides),
    )


# --- identity ------------------------------------------------------------------


def test_the_run_id_sorts_chronologically_and_names_the_run() -> None:
    """Lexical order is chronological order, so "which is newest" needs no
    parsing. Basic ISO format because a colon in a filename is fine on POSIX and
    fatal on Windows and in a URL, and this id appears in both."""
    from datetime import UTC, datetime

    run_id = build_run_id(
        model_type="xgboost",
        include_lender_priced=True,
        created_at=datetime(2026, 8, 24, 10, 15, 30, tzinfo=UTC),
        commit="02e3a94",
    )

    assert run_id == "20260824T101530Z-xgboost-with_lender_priced-02e3a94"
    assert ":" not in run_id


def test_the_tier_is_named_in_the_id_rather_than_spelled_true_or_false() -> None:
    """`...-True-abc1234` does not tell a reader which one is the safe default."""
    assert feature_tier(False) == "origination_only"
    assert feature_tier(True) == "with_lender_priced"


def test_the_dataset_fingerprint_is_a_stable_sixteen_character_digest(tmp_path: Path) -> None:
    """Value-exact against hashlib, because a fingerprint that silently changes
    definition makes every earlier manifest incomparable."""
    import hashlib

    path = tmp_path / "loans.csv"
    path.write_bytes(b"loan_amnt,loan_status\n1000,Fully Paid\n")

    expected = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    assert dataset_fingerprint(path) == expected
    # Chunking must not change the digest: the real extract is 1.19 GB and is
    # hashed a megabyte at a time.
    assert dataset_fingerprint(path, chunk_bytes=4) == expected


def test_the_commit_is_unknown_rather_than_an_exception_outside_a_repository(
    tmp_path: Path,
) -> None:
    """A model built from a tarball or a Docker layer is a legitimate model.
    Refusing to train because `git` is absent would be absurd."""
    assert git_commit(cwd=tmp_path) in {"unknown", git_commit()}
    assert git_commit(cwd=tmp_path / "does-not-exist") == "unknown"


def test_the_library_versions_include_python_and_the_numeric_stack() -> None:
    versions = library_versions()

    assert versions["python"].startswith("3.")
    assert {"numpy", "pandas", "scikit-learn"} <= set(versions)


# --- the bundle ----------------------------------------------------------------


def test_the_bundle_round_trips_with_its_threshold_and_spec(tmp_path: Path) -> None:
    """The threshold travels *inside* the pickle. Held in a separate JSON file it
    can be paired with the wrong model by anyone loading them separately."""
    save_bundle(make_bundle(), tmp_path)

    loaded = load_bundle(tmp_path)

    assert loaded.threshold == 0.30
    assert loaded.feature_spec == build_feature_spec(RAW_COLUMNS)
    assert loaded.metadata.run_id.endswith("abc1234")
    assert loaded.shap_background is None


def test_the_manifest_is_readable_without_unpickling_anything(tmp_path: Path) -> None:
    """The point of the JSON mirror: identity questions are answerable by `cat`,
    in any language, without importing this package or trusting the pickle."""
    save_bundle(make_bundle(), tmp_path)

    payload = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert payload["model_type"] == "logistic_regression"
    assert payload["dataset_sha256"] == "0123456789abcdef"
    assert payload["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert read_manifest(tmp_path) == make_metadata()


def test_scoring_goes_through_the_calibrator_not_the_bare_pipeline(tmp_path: Path) -> None:
    """One scoring method, so a caller cannot pick the uncalibrated one. The stub
    pipeline returns 0.40 and the stub calibrator 0.25; the bundle must say 0.25."""
    bundle = make_bundle()
    applicants = pd.DataFrame({"loan_amnt": [1000.0, 2000.0]})

    probabilities = bundle.predict_probability(applicants)

    assert probabilities.tolist() == [0.25, 0.25]
    # Approval is `p < threshold`, strictly, matching how the cost table was built.
    assert bundle.decide(np.array([0.29, 0.30, 0.31])).tolist() == [True, False, False]


def test_a_bundle_from_an_incompatible_schema_version_is_refused(tmp_path: Path) -> None:
    """Refused before use, so the error names the two versions instead of
    surfacing as a missing attribute inside a request handler."""
    save_bundle(make_bundle(bundle_schema_version=BUNDLE_SCHEMA_VERSION + 1), tmp_path)

    with pytest.raises(ValueError, match=f"version {BUNDLE_SCHEMA_VERSION + 1}"):
        load_bundle(tmp_path)


def test_a_pickle_holding_a_bare_estimator_is_refused_by_type(tmp_path: Path) -> None:
    """Exactly what `reports/models/logistic_regression.joblib` held before this
    existed: an estimator with no calibrator, no threshold, and no spec."""
    joblib.dump(StubModel(), tmp_path / BUNDLE_FILENAME)

    with pytest.raises(TypeError, match="not a ScoringBundle"):
        load_bundle(tmp_path)


def test_a_missing_bundle_names_the_path_and_the_way_out(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="riskscore runs"):
        load_bundle(tmp_path)


def test_the_metadata_ignores_fields_a_newer_version_added() -> None:
    """A manifest from a newer build must still be *listable*. The pickle is
    where a version mismatch has to be fatal; reading identity out of JSON is not."""
    payload = {**make_metadata().to_dict(), "future_field": ["something"]}

    assert RunMetadata.from_dict(payload) == make_metadata()


# --- atomic run directories ----------------------------------------------------


def test_a_run_becomes_visible_only_when_it_is_complete(tmp_path: Path) -> None:
    """The run directory is either absent or complete, so a dashboard polling it
    can never pair new metrics with an old calibration curve."""
    destination = tmp_path / "runs" / "run-a"

    with staged_run(tmp_path, "run-a") as staging:
        save_bundle(make_bundle(), staging)
        (staging / "metrics.json").write_text("{}", encoding="utf-8")
        # Mid-write: the real name does not exist yet.
        assert not destination.exists()
        assert staging.name.startswith(STAGING_PREFIX)

    assert (destination / BUNDLE_FILENAME).exists()
    assert (destination / "metrics.json").exists()


def test_a_failed_run_leaves_nothing_that_could_be_mistaken_for_a_result(
    tmp_path: Path,
) -> None:
    with (
        pytest.raises(RuntimeError, match="fit exploded"),
        staged_run(tmp_path, "run-b") as staging,
    ):
        (staging / "metrics.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("fit exploded")

    assert not (tmp_path / "runs" / "run-b").exists()
    assert list((tmp_path / "runs").glob(f"{STAGING_PREFIX}*")) == []


def test_publishing_over_an_existing_run_is_refused(tmp_path: Path) -> None:
    """Run ids embed a second-resolution timestamp, so a collision means two runs
    of the same model and tier started within one second - which is a bug, not a
    reason to overwrite a published artifact."""
    (tmp_path / "runs" / "run-c").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="run-c"), staged_run(tmp_path, "run-c"):
        pass


def test_abandoned_staging_directories_are_pruned_only_once_they_are_old(
    tmp_path: Path,
) -> None:
    """`staged_run` cleans up after an exception but not after SIGKILL - an OOM
    kill during a fit on the real extract, which is the scenario this project's
    memory work is about. Age-gated so a running fit is never deleted."""
    runs = tmp_path / "runs"
    fresh = runs / f"{STAGING_PREFIX}fresh"
    stale = runs / f"{STAGING_PREFIX}stale"
    fresh.mkdir(parents=True)
    stale.mkdir()
    old = time.time() - 7200
    os.utime(stale, (old, old))

    removed = prune_staging(tmp_path, older_than_seconds=3600)

    assert removed == [stale.name]
    assert fresh.exists()


# --- the registry --------------------------------------------------------------


def publish(root: Path, run_id: str, created_at: str, **overrides: Any) -> RunMetadata:
    """Write a complete run and register it. The helper the retention tests use."""
    metadata = make_metadata(run_id=run_id, created_at=created_at, **overrides)
    with staged_run(root, run_id) as staging:
        save_bundle(make_bundle(metadata), staging)
    register_run(root, metadata=metadata, metrics={"auc_roc": 0.64})
    return metadata


def test_registering_a_run_records_it_and_makes_it_active(tmp_path: Path) -> None:
    metadata = publish(tmp_path, "run-1", "2026-01-01T00:00:00Z")

    entries = read_registry(tmp_path)

    assert [entry["run_id"] for entry in entries] == ["run-1"]
    assert entries[0]["metrics"] == {"auc_roc": 0.64}
    assert entries[0]["rows"] == {"train": 10, "validation": 5, "test": 5}
    assert read_active_run_id(tmp_path) == "run-1"
    assert load_active_bundle(tmp_path).metadata == metadata


def test_registering_the_same_run_twice_does_not_duplicate_the_entry(tmp_path: Path) -> None:
    """A retried registration after a crash between the two writes must converge,
    not grow the index."""
    metadata = make_metadata(run_id="run-1")
    register_run(tmp_path, metadata=metadata)
    register_run(tmp_path, metadata=metadata, metrics={"auc_roc": 0.7})

    entries = read_registry(tmp_path)

    assert len(entries) == 1
    assert entries[0]["metrics"] == {"auc_roc": 0.7}


def test_a_run_can_be_registered_without_becoming_active(tmp_path: Path) -> None:
    """`riskscore compare` fits two models; only one of them should be served."""
    publish(tmp_path, "run-1", "2026-01-01T00:00:00Z")
    register_run(tmp_path, metadata=make_metadata(run_id="run-2"), make_active=False)

    assert read_active_run_id(tmp_path) == "run-1"
    assert {entry["run_id"] for entry in read_registry(tmp_path)} == {"run-1", "run-2"}


def test_activating_an_unknown_run_is_refused_with_the_known_ones(tmp_path: Path) -> None:
    """The failure being prevented is a typo that leaves the service unable to
    boot at all."""
    publish(tmp_path, "run-1", "2026-01-01T00:00:00Z")

    with pytest.raises(ValueError, match=r"Known runs: \['run-1'\]"):
        set_active_run(tmp_path, "run-2")


def test_rolling_back_to_an_earlier_run_is_one_call(tmp_path: Path) -> None:
    publish(tmp_path, "run-1", "2026-01-01T00:00:00Z")
    publish(tmp_path, "run-2", "2026-01-02T00:00:00Z")
    assert read_active_run_id(tmp_path) == "run-2"

    set_active_run(tmp_path, "run-1")

    assert read_active_run_id(tmp_path) == "run-1"
    assert load_active_bundle(tmp_path).metadata.run_id == "run-1"


def test_no_active_run_and_a_dangling_pointer_are_different_errors(tmp_path: Path) -> None:
    """One means nothing has been trained; the other means a run was deleted out
    from under the pointer. The fix differs, so the message must."""
    with pytest.raises(FileNotFoundError, match="riskscore train"):
        load_active_bundle(tmp_path)

    publish(tmp_path, "run-1", "2026-01-01T00:00:00Z")
    import shutil

    shutil.rmtree(tmp_path / "runs" / "run-1")

    with pytest.raises(FileNotFoundError, match="is registered but its directory"):
        load_active_bundle(tmp_path)


def test_a_corrupt_registry_reads_as_empty_rather_than_raising(tmp_path: Path) -> None:
    """The run directories are the data and the registry is an index of them.
    Refusing to list runs because one JSON file got truncated turns a cosmetic
    problem into an outage."""
    (tmp_path / REGISTRY_FILENAME).write_text('[{"run_id": "run-1"', encoding="utf-8")

    assert read_registry(tmp_path) == []


def test_the_registry_can_be_rebuilt_from_the_manifests_on_disk(tmp_path: Path) -> None:
    """The repair path for a truncated index, and for runs copied in from
    elsewhere."""
    publish(tmp_path, "run-2", "2026-01-02T00:00:00Z")
    publish(tmp_path, "run-1", "2026-01-01T00:00:00Z")
    (tmp_path / "runs" / "run-1" / "metrics.json").write_text(
        json.dumps({"auc_roc": 0.61, "brier_score": 0.12, "rows_test": 500}), encoding="utf-8"
    )
    (tmp_path / REGISTRY_FILENAME).unlink()

    entries = rebuild_registry(tmp_path)

    # Chronological by `created_at`, not by directory listing order.
    assert [entry["run_id"] for entry in entries] == ["run-1", "run-2"]
    # Only the headline metrics are lifted into the index; the rest stay in the
    # metrics file, which is what the index exists to avoid reading N times.
    assert entries[0]["metrics"] == {"auc_roc": 0.61, "brier_score": 0.12}
    assert read_registry(tmp_path) == entries


def test_rebuilding_skips_staging_directories_and_manifestless_ones(tmp_path: Path) -> None:
    publish(tmp_path, "run-1", "2026-01-01T00:00:00Z")
    (tmp_path / "runs" / f"{STAGING_PREFIX}half").mkdir()
    (tmp_path / "runs" / "hand-made").mkdir()

    assert [entry["run_id"] for entry in rebuild_registry(tmp_path)] == ["run-1"]


# --- retention -----------------------------------------------------------------


def test_retention_keeps_the_newest_runs_and_deletes_the_rest(tmp_path: Path) -> None:
    for day in range(1, 5):
        publish(tmp_path, f"run-{day}", f"2026-01-0{day}T00:00:00Z")

    removed = prune_runs(tmp_path, keep=2)

    assert removed == ["run-1", "run-2"]
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["run-3", "run-4"]
    assert [entry["run_id"] for entry in read_registry(tmp_path)] == ["run-3", "run-4"]


def test_retention_never_deletes_the_active_run(tmp_path: Path) -> None:
    """Otherwise retention policy takes the service down, which is a worse
    outcome than a stale directory."""
    for day in range(1, 4):
        publish(tmp_path, f"run-{day}", f"2026-01-0{day}T00:00:00Z")
    set_active_run(tmp_path, "run-1")

    removed = prune_runs(tmp_path, keep=1)

    assert removed == ["run-2"]
    assert (tmp_path / "runs" / "run-1").exists()
    assert load_active_bundle(tmp_path).metadata.run_id == "run-1"


def test_retention_honours_an_explicit_protected_list(tmp_path: Path) -> None:
    for day in range(1, 4):
        publish(tmp_path, f"run-{day}", f"2026-01-0{day}T00:00:00Z")

    removed = prune_runs(tmp_path, keep=1, protected=("run-1",))

    assert removed == ["run-2"]


def test_keeping_zero_runs_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        prune_runs(tmp_path, keep=0)


# --- the lock ------------------------------------------------------------------


def test_a_stale_lock_is_broken_rather_than_wedging_every_later_run(tmp_path: Path) -> None:
    """A registry update is two small JSON writes. Anything holding the lock for a
    minute has died - most likely by SIGKILL, which leaves no chance to clean up -
    and without a stale timeout the first crash wedges the project."""
    lock = tmp_path / "registry.lock"
    tmp_path.mkdir(parents=True, exist_ok=True)
    lock.write_text("999999\n", encoding="utf-8")
    old = time.time() - 600
    os.utime(lock, (old, old))

    register_run(tmp_path, metadata=make_metadata(run_id="run-1"))

    assert [entry["run_id"] for entry in read_registry(tmp_path)] == ["run-1"]
    # Released, not merely broken: the next writer must not have to wait it out.
    assert not lock.exists()


def test_a_lock_held_by_someone_else_is_waited_out_and_then_reported(tmp_path: Path) -> None:
    """The wait timeout is strictly shorter than the stale timeout, and that
    ordering is the point: a lock held legitimately for slightly too long must be
    *waited out*, not stolen. Were the two equal this error would be unreachable."""
    from risk_score import artifacts

    lock = tmp_path / "registry.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("1\n", encoding="utf-8")

    with (
        pytest.raises(TimeoutError, match="delete the file"),
        artifacts._registry_lock(tmp_path, timeout=0.2, stale_after=60.0),
    ):
        pass

    # Still held: a timed-out waiter must not remove someone else's lock.
    assert lock.exists()


def test_the_active_pointer_is_replaced_never_truncated(tmp_path: Path) -> None:
    """The service reads this file at boot, and a plain `write_text` has a window
    in which it is zero bytes."""
    publish(tmp_path, "run-1", "2026-01-01T00:00:00Z")
    publish(tmp_path, "run-2", "2026-01-02T00:00:00Z")

    payload = json.loads((tmp_path / ACTIVE_RUN_FILENAME).read_text(encoding="utf-8"))

    assert payload["run_id"] == "run-2"
    assert payload["activated_at"].endswith("Z")
    # No temp files left behind by the replace.
    assert [p.name for p in tmp_path.glob("*.tmp-*")] == []
