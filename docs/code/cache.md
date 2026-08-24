# `src/risk_score/cache.py`

## Purpose

Stop paying for the same CSV parse over and over.

Reading `data/raw/1/loan.csv` is the slowest step in a run and the only one that is pure: 1.19 GB of text through the pandas parser, producing exactly the same canonicalized frame every time.
Once, that is tolerable.
While iterating on a cost matrix, comparing two models on an identical split, or regenerating a model card, it is most of the wall clock, and all of it is spent recomputing a value that did not change.

So the canonicalized frame is written once as parquet - columnar, typed, roughly an order of magnitude faster to read back - under a key derived from the input's **content**.
Not its path, and not its mtime.
Content-addressing is what makes the cache safe to trust: an extract re-exported in place gets a different key rather than a stale hit, and two copies of the same file at different paths share one entry.

Measured on the synthetic 2,000-row extract, a warm run completes in about 1.3 seconds against roughly 2.5 cold.
On the real extract the ratio is the point rather than the seconds.

## Public API

| Name | What it is |
| --- | --- |
| `read_raw_loans_cached` | `read_raw_loans`, memoized on disk. `cache_dir=None` calls straight through. |
| `cache_key` | The entry name for one (file digest, alias configuration) pair. |
| `CACHE_FORMAT_VERSION` | Bumped by hand when canonicalization changes meaning, which invalidates every older entry. |
| `DEFAULT_CACHE_DIR` | `data/cache`, already gitignored. |

## Inputs and outputs

Reads a CSV, writes two files per entry into the cache directory:

```
data/cache/
  v1-<digest16>-<aliashash8>.parquet   the canonicalized frame
  v1-<digest16>-<aliashash8>.json      the SchemaReport that describes how it was built
```

`fingerprint` is a **parameter**, not something this module computes.
The caller already needs the digest for the run manifest, and hashing 1.19 GB twice to write the same sixteen characters into two places is exactly the cost this module exists to remove.

The `SchemaReport` is cached beside the frame because it goes into the run manifest.
A cached run whose manifest said "renamed 0 columns, 0 unparseable dates" would be quietly lying about how its own data was built, and the manifest's entire job is to be trustworthy about that.

Nothing else in the project imports this module except `pipeline.py`, and the pipeline's `cache_dir` defaults to `None`.

## Invariants and failure modes

**A miss is never fatal, and neither is a corrupt hit.**
Every failure path falls back to reading the CSV.
A truncated parquet, a `pyarrow` that is not installed, a report written by older code, a full disk, a read-only volume: all of them log at `WARNING` and read the CSV.
A cache that can break a run is worse than no cache, because the failure arrives on a machine where the CSV read worked yesterday.

**The cleanup after a failed write cannot fail either.**
`unlink(missing_ok=True)` is not enough on its own: when the failure was the `mkdir` - a cache directory whose parent is a regular file, say - unlinking a path *through* that file raises `NotADirectoryError`, and the cache would fail the run after all.
The cleanup suppresses `OSError` for that reason, and there is a test that creates exactly that filesystem shape.

**The key includes everything that changes the output.**
The file digest, the alias overrides from the run config, and `CACHE_FORMAT_VERSION`.
Alias overrides are in the key because they decide which source column becomes which canonical one, so two runs of one file with different aliases are genuinely different frames.
The key is invariant to alias *ordering* and to iterable type, so a config rewritten with the same meaning does not invalidate every entry.

**Both halves or neither.**
A hit requires the parquet *and* the JSON.
The frame is written first and the report last, so a crash between the two produces an entry that misses rather than one that hits with half its metadata.

**Writes are atomic.**
Into a temporary name with a uuid suffix, then `Path.replace`.
A process killed mid-write leaves no truncated parquet for the next run to believe, and two concurrent runs on the same extract cannot interleave into one file.

**A hit is value-exact.**
The warm frame equals the cold frame including dtypes and index, asserted with `assert_frame_equal` rather than by comparing shapes.
A cached frame that differed by one `object` column would train a different model, which is the one failure a cache must not have.

**Off by default in the library, on by default in the CLI.**
Opposite defaults on purpose: a command line writing under `data/cache` is expected, and a library function doing it unasked is a surprise.

## What must NOT live here

- **Caching anything downstream of the raw read.** A fitted model, a metrics payload, or a split is not pure with respect to its inputs in the way a parse is, and the moment a cache holds derived results a stale entry becomes a wrong *answer* rather than a slow read.
- **Deriving the fingerprint.** The caller owns it, so it is computed once per run.
- **Deciding *when* to cache.** The pipeline and the CLI decide; this module does what it is told.
- **Cache eviction by size or age.** The entries are content-addressed and idempotent, so `rm -rf data/cache` is the eviction policy and it is always safe.

## Related tests

`tests/test_cache.py`, ten tests, most of them about the fallback rather than the hit: every way an entry can be broken must end in the CSV being read again rather than in an exception.

- `test_a_warm_read_returns_exactly_what_the_cold_read_returned` is the correctness claim, dtypes included.
- `test_the_second_read_does_not_touch_the_csv_at_all` deletes the input between the two reads.
- `test_a_truncated_parquet_is_ignored_rather_than_raised` writes `b"PAR1 and then nothing"` over an entry and asserts both the fallback and that the entry is *rewritten*, so one corruption does not tax every later run.
- `test_a_report_written_by_older_code_is_a_miss_not_a_crash` and `test_a_frame_without_its_report_is_a_miss` cover the both-halves rule.
- `test_a_write_into_an_unwritable_directory_does_not_fail_the_read` is the one that found the `NotADirectoryError` in the cleanup path.
- `test_different_aliases_are_a_different_entry` pins the key's contents and its invariance to ordering.
- `tests/test_pipeline.py::test_a_second_run_reuses_the_cached_extract_instead_of_reparsing_it` is the end-to-end version: two runs, and the second run's own published log says it read the frame from the cache.

## Known limits

- **`CACHE_FORMAT_VERSION` is bumped by hand.** Nothing derives it from the code, so an edit to `schema.py` that forgets to bump it leaves stale entries readable, and a stale entry here is a silently different frame. The mitigations are that it is one line to change and that `cache_dir=None` turns the whole thing off. Deriving it from a hash of the canonicalization source is possible and was rejected: it would invalidate every entry on a comment change.
- **The cache does not save the hash.** The run still reads the whole file to fingerprint it, so a warm run is bounded below by one streaming pass over the input rather than by zero. Keying on `(size, mtime, path)` would avoid that and reintroduce exactly the stale-hit class of bug the digest exists to prevent.
- **No cross-machine sharing.** Entries are local files. A shared object store would need a fetch, an integrity check, and a permission model, none of which this project needs.
- **`pyarrow` lives in the `[train]` extra.** A `[serve]`-only install logs one warning per read and works, which is correct - a serving process has no reason to read a CSV at all.
- **Unbounded growth.** One entry per distinct extract, and nothing removes old ones. `rm -rf data/cache` is safe at any time, and that is the documented policy.
