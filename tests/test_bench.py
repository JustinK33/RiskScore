"""The latency benchmark: its arithmetic, and one guard against a regression.

Two halves, and they need different things.

The percentile arithmetic is tested against a stub whose call durations are
*declared*, because that is the only way to assert that a p99 is the call it says
it is. Measuring a real service and asserting "the p99 is a number" would pass
whether the rank arithmetic were right or off by one, and off by one is the whole
failure mode of a hand-rolled percentile.

The regression guard is the opposite: a real bundle, real scoring, and a budget so
generous it can only fail on a change that made scoring an order of magnitude
slower. It is marked ``slow`` and excluded from the default run, because a
timing assertion in a suite everybody runs on a laptop with a build in the
background is a test that cries wolf - and a flaky guard gets deleted, which
leaves no guard at all. The declared budget lives in
:mod:`risk_score.api.scoring`; this is deliberately not it.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any, cast

import pytest

from risk_score.api.bench import (
    BENCH_SEED,
    Latency,
    bench_applicant,
    measure,
)
from risk_score.api.scoring import Score, ScoringService
from risk_score.pipeline import RunResult

#: Ten times the p99 this project measures for the more expensive configuration,
#: which is the point: this catches "scoring got 10x slower", not "the runner was
#: busy". A tight budget here would be a second, worse copy of the one declared in
#: scoring.py, and the first one to fail on a loaded CI box.
REGRESSION_BUDGET_MS = 250.0


class _DeclaredDurations:
    """A :class:`ScoringService` stand-in that reports latencies from a list.

    Only ``score`` is implemented, because that is all :func:`measure` calls -
    hence the ``cast`` at each use site rather than a subclass, which would drag in
    a bundle, an explainer and a pickle to test division.
    """

    def __init__(self, durations: Iterable[float]) -> None:
        self.calls = 0
        self._durations = list(durations)

    def score(
        self,
        applicant: Mapping[str, Any],
        *,
        with_reasons: bool = True,
        top_k: int = 5,
    ) -> Score:
        latency = self._durations[self.calls]
        self.calls += 1
        return Score(
            default_probability=0.1,
            decision="approve",
            threshold=0.5,
            baseline_log_odds=None,
            total_log_odds=None,
            reasons=(),
            latency_ms=latency,
        )


def _measure(durations: Iterable[float], **kwargs: Any) -> Latency:
    service = cast(ScoringService, _DeclaredDurations(durations))
    return measure(service, {}, label="stub", warmup=0, **kwargs)


# --- the percentiles ------------------------------------------------------------


def test_the_percentiles_are_the_nearest_measured_call() -> None:
    """Nearest rank, so every figure reported is a call that happened.

    A hundred calls taking 1.0 through 100.0 ms makes the correct answers exact
    and an off-by-one visible: interpolation would report 50.5 for the median, and
    a ``round`` instead of a ``ceil`` would report 99.0 for the p99.
    """
    durations = [float(value) for value in range(1, 101)]
    random.Random(0).shuffle(durations)

    result = _measure(durations, calls=100)

    assert (result.p50_ms, result.p90_ms, result.p99_ms) == (50.0, 90.0, 99.0)
    assert result.max_ms == 100.0
    assert result.calls == 100


def test_one_call_is_its_own_every_percentile() -> None:
    """The degenerate case, which is where an index arithmetic bug lands first."""
    result = _measure([4.0], calls=1)

    assert (result.p50_ms, result.p90_ms, result.p99_ms, result.max_ms) == (4.0,) * 4


def test_warmup_calls_are_excluded_from_the_measurement() -> None:
    """The whole reason warmup exists: a cold first call must not set the p99.

    Ten expensive calls then five cheap ones. If the warmups were counted, the
    reported maximum would be 900 ms rather than 5 ms - which is what an unwarmed
    benchmark of a fresh process actually reports.
    """
    service = _DeclaredDurations([900.0] * 10 + [5.0] * 5)

    result = measure(cast(ScoringService, service), {}, label="stub", calls=5, warmup=10)

    assert result.max_ms == 5.0
    assert service.calls == 15


def test_a_benchmark_with_no_calls_is_refused() -> None:
    """Rather than dividing by zero or reporting percentiles of an empty list."""
    with pytest.raises(ValueError, match="at least 1"):
        _measure([], calls=0)


# --- the generated applicant ----------------------------------------------------


def test_the_generated_applicant_covers_the_declared_inputs(trained_run: RunResult) -> None:
    """Every raw input the bundle declares, because a benchmark of a half-filled
    applicant measures the imputer rather than the model.

    Asserted as equality and not as a subset: a column the generator stopped
    emitting would otherwise silently drop out of the measurement, and the
    imputed path is faster than the parsed one.
    """
    spec = trained_run.bundle.feature_spec

    applicant = bench_applicant(spec)

    assert set(applicant) == set(spec.raw_inputs)


def test_the_generated_applicant_is_the_same_row_every_time(trained_run: RunResult) -> None:
    """Fixed seed, so two runs of ``riskscore bench`` are comparable with each other."""
    spec = trained_run.bundle.feature_spec

    first = bench_applicant(spec)
    second = bench_applicant(spec, seed=BENCH_SEED)

    assert str(first) == str(second)


def test_the_generated_applicant_actually_scores(trained_run: RunResult) -> None:
    """The assertion that makes the benchmark mean anything: the row it times is a
    row the service can score, reason codes included."""
    service = ScoringService(trained_run.bundle)

    score = service.score(bench_applicant(trained_run.bundle.feature_spec))

    assert 0.0 <= score.default_probability <= 1.0
    assert score.decision in {"approve", "decline"}
    assert score.reasons, "the session bundle explains, so a benchmarked score should too"


# --- the regression guard -------------------------------------------------------


@pytest.mark.slow
def test_scoring_stays_inside_the_regression_budget(trained_run: RunResult) -> None:
    """A real bundle, both configurations, against a deliberately loose ceiling.

    Fewer calls than ``riskscore bench`` uses: this is a tripwire, and 200 calls
    is enough to notice a tenfold regression while keeping the test under a couple
    of seconds.
    """
    service = ScoringService(trained_run.bundle)
    applicant = bench_applicant(trained_run.bundle.feature_spec)

    for with_reasons in (False, True):
        result = measure(
            service,
            applicant,
            label=f"reasons={with_reasons}",
            calls=200,
            with_reasons=with_reasons,
        )

        assert result.p99_ms < REGRESSION_BUDGET_MS, str(result)
