# `src/risk_score/api/bench.py`

## Purpose

Check the latency budget rather than assert it.

`scoring.py`'s docstring declares a target - p50 <= 12 ms and p99 <= 25 ms for a score, p50 <= 25 ms and p99 <= 50 ms with reason codes.
A declared budget nobody measures is a comment.
This module is `riskscore bench`: it scores one applicant a few thousand times and reports the percentiles, so the claim in the docstring is a number somebody can reproduce on their own machine.

It measures **in-process**, calling `ScoringService.score` directly with no socket involved.
That deliberately excludes uvicorn, JSON parsing, pydantic validation and the kernel, which together are a millisecond or two of roughly constant overhead and which vary with the client, the loop implementation, and the machine's network stack.
What is left is the part this project controls and the part a regression would show up in.
The end-to-end figure is a different and also useful question; a load generator against a running server is the right tool for it, and `tests/test_api.py` covers the HTTP contract itself.

## Public API

| Name | What it is |
| --- | --- |
| `measure` | Score one applicant `calls` times and summarize the durations as a `Latency`. |
| `bench_applicant` | One generated applicant covering a bundle's declared raw inputs. |
| `Latency` | Frozen result: label, call count, p50, p90, p99, max. `__str__` is the CLI line, `as_dict` is `--json`. |
| `DEFAULT_CALLS`, `DEFAULT_WARMUP` | 2000 and 25. Mirrored in `cli.py` as `BENCH_CALLS`/`BENCH_WARMUP`; see **Known limits**. |
| `BENCH_SEED` | Fixed, so two benchmark runs score the same applicant. |

## Inputs and outputs

Takes a `ScoringService` and a mapping.
Returns a `Latency` per configuration.
Reads no files, writes no files, and touches no network.

`bench_applicant` is the reason `riskscore bench` needs nothing but a published run.
It generates its row with `make_synthetic_loans(n_rows=1, seed=BENCH_SEED)` and resolves aliases through `normalize_column_names`, because the generator emits the extract's own column names while `FeatureSpec.raw_inputs` are canonical.
The extract a model was trained on is frequently not on the machine serving it, and a benchmark that requires the 1.19 GB download is a benchmark nobody runs.

`Score.latency_ms` is the timing source, not a stopwatch wrapped around the call.
So what `bench` prints is exactly what a `/predict` response reports - one definition of latency rather than two that can disagree by however much the wrapper costs.

## Invariants and failure modes

**Percentiles are nearest-rank, with no interpolation.**
`rank = ceil(fraction * n)`, then that element of the sorted sample, clamped to the ends.
Every figure printed is therefore the duration of a call that actually happened.
An interpolated p99 is an average of two calls: a number nothing measured, and one that quietly smooths over a bimodal tail - which is the shape a latency distribution with an occasional GC pause actually has.

**Warmup calls are excluded, and they are load-bearing.**
The first score through a fresh process pays for lazy scipy imports, the first allocation of numpy scratch buffers, and pandas' own import-time caches.
On a cold process that call is tens of times the steady-state cost, so including it would put a startup artifact in the p99 of the first few hundred runs.
`warmup` is clamped at zero rather than validated, because a negative warmup has an obvious intended meaning.

**`calls < 1` raises rather than dividing by zero.**
The alternative is percentiles of an empty list, which is an `IndexError` from inside `_percentile` with no useful message.

**The same applicant every time, on purpose.**
The cost of a score is dominated by the number of passes through the preprocessor and not by the values in the row, so varying the input would add noise without covering anything new.
It also means a difference between two `bench` runs is a change in the code rather than in the data.

**`riskscore bench` only measures a configuration the bundle can actually serve.**
The CLI adds the "with reason codes" measurement only when `service.can_explain`.
Measuring a with-reasons configuration that silently produced none would report the cheaper number under the more expensive label, which is the one way this output could lie about the thing it exists to report.

## What must NOT live here

- **The budget itself.** It is declared in `scoring.py`, next to the code responsible for meeting it. A second copy here would be a second thing to update, and the two would disagree.
- **HTTP.** The moment this module starts a server it is measuring uvicorn, and the number stops being attributable.
- **Anything that writes.** `bench` is read-only against a run directory, so it is safe to point at a production report tree.
- **A load model.** Concurrency, arrival distributions, and connection reuse are a load generator's job. This measures the cost of one call.

## Related tests

`tests/test_bench.py`, eight tests, in two halves that need different things.

- The arithmetic is tested against a stub whose call durations are *declared*. `test_the_percentiles_are_the_nearest_measured_call` runs 100 shuffled durations of 1.0 through 100.0 ms and asserts exactly `(50.0, 90.0, 99.0)`, so interpolation (which reports 50.5 for the median) and `round` instead of `ceil` (which reports 99.0 for the p90's neighbour) are both visible. Measuring a real service and asserting "the p99 is a number" would pass whether the rank arithmetic were right or off by one, and off by one is the entire failure mode of a hand-rolled percentile.
- `test_warmup_calls_are_excluded_from_the_measurement` declares ten 900 ms calls then five 5 ms ones and asserts a reported maximum of 5 ms with fifteen calls made.
- `test_the_generated_applicant_covers_the_declared_inputs` asserts set *equality* against `spec.raw_inputs`, not a subset: a column the generator stopped emitting would otherwise drop out of the measurement silently, and the imputed path is faster than the parsed one.
- `test_the_generated_applicant_actually_scores` is what makes the whole benchmark mean something - the row it times is a row the service can score, reason codes included.
- `test_scoring_stays_inside_the_regression_budget` is the tripwire: a real bundle, both configurations, `p99 < 250 ms`. Marked `slow` and excluded from the default run, because a timing assertion in a suite everybody runs on a laptop with a build in the background is a test that cries wolf - and a flaky guard gets deleted, which leaves no guard at all.
- `tests/test_cli.py::test_bench_json_reports_ordered_percentiles` pins `p50 <= p90 <= p99 <= max` and that the with-reasons configuration is the slower one.

## Known limits

- **`DEFAULT_CALLS` and `DEFAULT_WARMUP` are spelled twice.** `cli.py` cannot import this module at the top level: `risk_score/api/__init__.py` eagerly imports `app`, which imports fastapi, which is the `[serve]` extra - so a module-level import would make `riskscore train` fail on a training box. Resolving the defaults inside the handler was rejected because `--help` would stop stating the number. `tests/test_cli.py::test_bench_defaults_match_the_library` fails if the two copies drift.
- **Single-threaded, one applicant, no concurrency.** This says nothing about throughput or about what happens under contention. Both are real questions and neither is this one.
- **The number is not portable.** Percentiles from a laptop are not percentiles from a container with a CPU quota, and BLAS thread counts alone move them. The regression guard is loose by an order of magnitude for exactly that reason.
- **Batch scoring is not measured.** `/predict/batch` is a different cost curve - one transform for many rows - and it would need its own row-count sweep to say anything useful.
