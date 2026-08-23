# Conventions

Rules that apply to every file in this repository.
They exist so a reviewer never has to ask "is this deliberate?".

## Comments

The codebase started with zero inline comments and complete docstrings, which sounds tidy and is the wrong split.
A docstring tells you the contract.
It cannot tell you why a magic number is 0.005, why one step must precede another, or which bug a guard clause prevents - and those are exactly the things that get broken by a well-meaning refactor.

**Docstring = the contract. Inline `#` = the why.**

A comment is mandatory at:

- **Every guard clause.** What invalid state is being rejected, and what breaks downstream if it gets through.
- **Every magic number,** with its provenance. `min_frequency=0.005` is meaningless; "0.5% of rows, which caps `addr_state` at the ~30 states with enough volume to estimate a coefficient" is reviewable.
- **Every non-obvious pandas or numpy idiom.** If a reader has to reconstruct why `.reindex` rather than `[]`, or why a `Series` is built before assignment, say it.
- **Every leakage-relevant choice.** Which partition a fitted object saw, and why it could not have seen another.
- **Every order-dependent step.** If moving two lines changes the result, that must be stated where the lines are, not in a design doc.
- **Every fixed bug,** tagged with its audit ID: `# Fix B12: ...`.

A comment is banned when it restates the code (`# increment the counter`) and when it is a `TODO` with no owner and no condition.
Twelve bare `TODO:` blocks in docstrings were removed during this work: each was either resolved or deleted, because a TODO nobody owns is a comment that reads as a plan and functions as noise.

### Audit-ID tags

Bugs found in the audit that prompted this work are numbered `B01`-`B32` for correctness issues, `S01`-`S06` for security issues, and `P01`-`P05` for performance issues.

Each ID appears in exactly **two** places:

1. A `# Fix B12: ...` comment at the code that fixes it.
2. A test named `test_b12_*` that fails if the fix is reverted.

So `rg B12` shows you the fix and its proof, and nothing else.
An ID with only a comment and no test means the fix is unproven.

### Section banners

Functions long enough to have distinct phases get numbered banners:

```python
# --- 2. fit the calibrator on validation only ---
```

This is a stronger signal than a blank line when the *order* is load-bearing, which in this codebase it usually is.

## Errors

Errors are read by someone at a terminal who does not have the code open.
Every raise includes the observed state, not just the expectation:

```python
# Good - tells you what to change.
raise ValueError(
    f"Time-based split produced an empty validation set. "
    f"Available date range is {observed_min} to {observed_max}."
)

# Bad - true, unactionable.
raise ValueError("Invalid split dates.")
```

`TRY003` is disabled in ruff for this reason.

Errors are raised by name: never `KeyError("missing columns")` when the caller needs to know *which*.

Never swallow an exception to keep a run alive unless the recovery is deliberate and counted.
Where rows are dropped - unparseable dates, immature outcomes, split gaps - the count goes into the run manifest.
A pipeline that silently discards 40% of its input looks identical to one that discards none.

## Naming

- Canonical column names are the Lending Club raw names (`loan_amnt`, `annual_inc`, `revol_util`). Aliases map *to* them, never the reverse.
- Engineered columns carry a suffix describing the operation: `dti_clean`, `credit_utilization`, `loan_to_income_ratio`, `credit_history_months`.
- A fitted object's variable name says which partition fitted it: `threshold_from_validation`, not `threshold`.
- Module-level constants are `UPPER_SNAKE` and immutable (`tuple` / `frozenset`, never `list` / `set`), so a caller cannot mutate the default for every other caller.

## Types

`mypy --strict` on `src` and `tests`, both.
Tests are where signature drift shows up first, so exempting them defeats the point.

`Any` is allowed in exactly two places: sklearn estimators, which have no stubs, and YAML config values before validation.
Everywhere else, if a type is hard to write, that is usually the design telling you something.

## Formatting

`ruff format` is authoritative; `ruff check` runs the lint rules in `pyproject.toml`.
Line length is 100.
Both run in CI, and `ruff format --check` fails the build - formatting arguments are not worth a single review comment.

## Tests

- **Assert values, not shapes.** `assert 0 <= result <= 1` is why an inverted-KS bug survived 25 passing tests: the statistic was 0.93 alongside an AUC of 0.07, and every assertion passed. Metric helpers are compared to a scipy or sklearn oracle, or to a hand-computed constant.
- **One test per audit ID,** named for it.
- **No copy-pasted fixtures.** Data comes from `risk_score.sample_data`, which is also what `riskscore make-sample-data` uses, so the demo path and the test path cannot diverge.
- **`pytest.importorskip` for optional dependencies,** never a bare `try/except ImportError` that silently passes.
- Anything that needs the real 1.19 GB extract or a full fit is marked `slow` and excluded from the default run.

## Markdown

One sentence per line.
It makes diffs show which sentence changed instead of reflowing a paragraph.
