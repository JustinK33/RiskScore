"""Tests for the content-addressed parquet cache.

A cache is only worth having if a stale or corrupt entry cannot change a run's
answer, so most of what is asserted here is the *fallback* behaviour: a hit must
equal a cold read exactly, and every way an entry can be broken must end in the
CSV being read again rather than in an exception.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from risk_score.artifacts import dataset_fingerprint
from risk_score.cache import (
    CACHE_FORMAT_VERSION,
    cache_key,
    read_raw_loans_cached,
)
from risk_score.data_loading import read_raw_loans


def read(path: Path, cache_dir: Path | None, **kwargs: Any) -> tuple[pd.DataFrame, Any]:
    return read_raw_loans_cached(
        path, fingerprint=dataset_fingerprint(path), cache_dir=cache_dir, **kwargs
    )


def test_a_warm_read_returns_exactly_what_the_cold_read_returned(
    raw_csv: Path, tmp_path: Path
) -> None:
    """Value-exact, including dtypes and the index: a cached frame that differs
    from a freshly parsed one by an `object` column would train a different
    model, which is the one failure a cache must not have."""
    cache = tmp_path / "cache"
    expected, expected_report = read_raw_loans(raw_csv)

    cold, cold_report = read(raw_csv, cache)
    warm, warm_report = read(raw_csv, cache)

    assert_frame_equal(cold, expected)
    assert_frame_equal(warm, expected)
    assert warm.dtypes.to_dict() == expected.dtypes.to_dict()
    assert cold_report == expected_report
    assert warm_report == expected_report


def test_the_second_read_does_not_touch_the_csv_at_all(raw_csv: Path, tmp_path: Path) -> None:
    """The point of the exercise, proven by removing the input: the 1.19 GB parse
    is the slowest step in a run and this is what skips it."""
    cache = tmp_path / "cache"
    fingerprint = dataset_fingerprint(raw_csv)
    expected, _ = read_raw_loans_cached(raw_csv, fingerprint=fingerprint, cache_dir=cache)

    raw_csv.unlink()
    warm, _ = read_raw_loans_cached(raw_csv, fingerprint=fingerprint, cache_dir=cache)

    assert_frame_equal(warm, expected)


def test_disabling_the_cache_writes_nothing(raw_csv: Path, tmp_path: Path) -> None:
    """`cache_dir=None` is the default for the library, because a function that
    writes files into the working directory unasked is a surprise."""
    cache = tmp_path / "cache"

    read(raw_csv, None)

    assert not cache.exists()


def test_a_changed_extract_gets_its_own_entry(raw_csv: Path, tmp_path: Path) -> None:
    """Keyed by content, not by path or mtime, so an extract re-exported in place
    cannot produce a stale hit."""
    cache = tmp_path / "cache"
    read(raw_csv, cache)
    before = {path.name for path in cache.glob("*.parquet")}

    frame = pd.read_csv(raw_csv)
    frame.iloc[:-1].to_csv(raw_csv, index=False)
    shortened, _ = read(raw_csv, cache)

    after = {path.name for path in cache.glob("*.parquet")}
    assert len(after) == 2
    assert before < after
    assert len(shortened) == len(frame) - 1


def test_different_aliases_are_a_different_entry() -> None:
    """The overrides decide which source column becomes which canonical one, so
    two runs of one file with different aliases are different frames."""
    plain = cache_key(fingerprint="abc123", column_aliases=None)
    aliased = cache_key(fingerprint="abc123", column_aliases={"loan_amnt": ("amount",)})

    assert plain != aliased
    # The version is in the key rather than in a sidecar file, so entries written
    # by older canonicalization miss instead of being trusted.
    assert plain.startswith(f"v{CACHE_FORMAT_VERSION}-abc123-")
    # Order and iterable type must not change the key, or a config rewritten with
    # the same meaning would invalidate every entry.
    assert cache_key(fingerprint="abc123", column_aliases={"a": ["x", "y"]}) == cache_key(
        fingerprint="abc123", column_aliases={"a": ("y", "x")}
    )


def test_a_truncated_parquet_is_ignored_rather_than_raised(
    raw_csv: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The scenario is a SIGKILL during a write, and the correct outcome is a
    slower run rather than a failed one."""
    cache = tmp_path / "cache"
    expected, _ = read(raw_csv, cache)
    entry = next(cache.glob("*.parquet"))
    entry.write_bytes(b"PAR1 and then nothing")

    with caplog.at_level("WARNING"):
        recovered, _ = read(raw_csv, cache)

    assert_frame_equal(recovered, expected)
    assert "ignoring unreadable cache entry" in caplog.text
    # And the entry is rewritten, so the corruption does not cost every later run.
    assert entry.read_bytes() != b"PAR1 and then nothing"


def test_a_report_written_by_older_code_is_a_miss_not_a_crash(
    raw_csv: Path, tmp_path: Path
) -> None:
    """A cached report missing a field would otherwise reach the run manifest
    half-built, which is worse than reading the CSV again."""
    cache = tmp_path / "cache"
    expected, expected_report = read(raw_csv, cache)
    report_path = next(cache.glob("*.json"))
    report_path.write_text(json.dumps({"renamed": {}}), encoding="utf-8")

    frame, report = read(raw_csv, cache)

    assert_frame_equal(frame, expected)
    assert report == expected_report


def test_a_frame_without_its_report_is_a_miss(raw_csv: Path, tmp_path: Path) -> None:
    """Both halves or neither: the report goes into the run manifest, so a hit
    that supplied only the frame would make the manifest describe a load that
    did not happen."""
    cache = tmp_path / "cache"
    expected, expected_report = read(raw_csv, cache)
    next(cache.glob("*.json")).unlink()

    frame, report = read(raw_csv, cache)

    assert_frame_equal(frame, expected)
    assert report == expected_report


def test_a_write_into_an_unwritable_directory_does_not_fail_the_read(
    raw_csv: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A full disk or a read-only volume must cost the cache, not the run."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    with caplog.at_level("WARNING"):
        frame, _ = read(raw_csv, blocked / "cache")

    expected, _ = read_raw_loans(raw_csv)
    assert_frame_equal(frame, expected)
    assert "could not write cache entry" in caplog.text


def test_no_temporary_files_are_left_behind(raw_csv: Path, tmp_path: Path) -> None:
    """A reader that found a half-written parquet would have no way to tell, so
    the write goes through a temporary name and then a rename."""
    cache = tmp_path / "cache"
    read(raw_csv, cache)

    assert [path.name for path in cache.glob("*.tmp-*")] == []
