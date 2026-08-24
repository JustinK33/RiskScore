"""Measuring what ``/predict`` actually costs, in-process and with no HTTP.

The latency budget is declared in :mod:`risk_score.api.scoring`; this is the
thing that checks it. Two decisions about how it measures are worth stating,
because a benchmark that flatters itself is worse than none.

**In-process, not over a socket.** ``ScoringService.score`` is called directly.
That deliberately excludes uvicorn, JSON parsing, pydantic validation and the
kernel, which together are a millisecond or two of roughly constant overhead -
and which vary with the client, the loop implementation and the machine's network
stack. What is left is the part this project controls and the part a regression
would show up in. An end-to-end figure is a useful thing to have and is not this;
:mod:`tests.test_api` asserts the HTTP contract, and a load generator against a
running server is the right tool for the other question.

**Nearest-rank percentiles, no interpolation.** Every figure printed is the
duration of a request that actually happened. An interpolated p99 is an average
of two calls, which is a number nothing measured and which quietly hides the
shape of a bimodal tail.

The warmup calls are not decoration. The first score through a fresh process pays
for lazy scipy imports, the first allocation of numpy scratch buffers, and
pandas' own import-time caches; on a cold process that call is tens of times the
steady-state cost, and including it would put a startup artifact in the p99 of
the first few hundred runs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from risk_score.api.scoring import ScoringService
from risk_score.explain import DEFAULT_TOP_K
from risk_score.sample_data import make_synthetic_loans
from risk_score.schema import normalize_column_names
from risk_score.transformers import FeatureSpec

#: Enough samples that a p99 is 20 calls rather than 1, and few enough that the
#: whole benchmark is seconds. At 2000 calls the p99 is the 1980th slowest, so a
#: single unlucky GC pause moves it by one rank instead of defining it.
DEFAULT_CALLS = 2000

#: See the module docstring. Twenty-five is well past where the curve flattens on
#: every machine this has been run on; the cost of overshooting is milliseconds.
DEFAULT_WARMUP = 25

#: Fixed, so two runs of ``riskscore bench`` score the same applicant. A
#: benchmark whose input changes between runs cannot be compared with itself.
BENCH_SEED = 20130101


@dataclass(frozen=True, slots=True)
class Latency:
    """One measured configuration, in milliseconds.

    ``max_ms`` is reported alongside the percentiles because it is the only figure
    that shows a single catastrophic call - a p99 over 2000 samples averages 20 of
    them away by construction.
    """

    label: str
    calls: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float

    def __str__(self) -> str:
        return (
            f"{self.label:<24} p50 {self.p50_ms:6.2f} ms   p90 {self.p90_ms:6.2f} ms   "
            f"p99 {self.p99_ms:6.2f} ms   max {self.max_ms:6.2f} ms   n={self.calls}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure(
    service: ScoringService,
    applicant: Mapping[str, Any],
    *,
    label: str,
    calls: int = DEFAULT_CALLS,
    warmup: int = DEFAULT_WARMUP,
    with_reasons: bool = True,
    top_k: int = DEFAULT_TOP_K,
) -> Latency:
    """Score the same applicant ``calls`` times and summarize the durations.

    The same applicant every time, which is the right choice for this
    measurement: the cost of a score is dominated by the number of passes through
    the preprocessor and not by the values in the row, so varying the input would
    add noise without covering anything. It also means a difference between two
    runs is a change in the code rather than in the data.

    ``Score.latency_ms`` is the timing source rather than a stopwatch around the
    call, so what is reported is exactly what the ``/predict`` response reports -
    one definition of latency, not two that can disagree.
    """
    if calls < 1:
        raise ValueError(f"calls must be at least 1, got {calls}.")
    for _ in range(max(warmup, 0)):
        service.score(applicant, with_reasons=with_reasons, top_k=top_k)

    samples = sorted(
        service.score(applicant, with_reasons=with_reasons, top_k=top_k).latency_ms
        for _ in range(calls)
    )
    return Latency(
        label=label,
        calls=calls,
        p50_ms=_percentile(samples, 0.50),
        p90_ms=_percentile(samples, 0.90),
        p99_ms=_percentile(samples, 0.99),
        max_ms=samples[-1],
    )


def bench_applicant(spec: FeatureSpec, *, seed: int = BENCH_SEED) -> dict[str, Any]:
    """One realistic applicant for the bundle's declared inputs.

    Generated rather than read from a file, so ``riskscore bench`` needs nothing
    but a published run - the extract a model was trained on is frequently not on
    the machine serving it, and a benchmark that requires the 1.19 GB download is
    a benchmark nobody runs.

    Aliases are resolved through :func:`~risk_score.schema.normalize_column_names`
    because the generator emits the extract's own column names and
    ``raw_inputs`` are canonical. Columns the bundle wants but this frame lacks
    are simply absent, which is a case scoring already handles - it is what an
    applicant with an unknown credit history looks like.
    """
    frame, _report = normalize_column_names(make_synthetic_loans(n_rows=1, seed=seed))
    row = frame.iloc[0]
    return {name: row[name] for name in spec.raw_inputs if name in frame.columns}


def _percentile(sorted_ms: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted sample. See the module docstring."""
    rank = math.ceil(fraction * len(sorted_ms))
    return sorted_ms[min(max(rank - 1, 0), len(sorted_ms) - 1)]
