"""Scoring one applicant, with no HTTP and no disk in sight.

The whole latency budget lives in this file, so it is worth stating what the
budget is, where it goes, and what it is measured on. The declared budget for a
warm process scoring one applicant against a logistic-regression bundle is
**p50 <= 12 ms / p99 <= 25 ms without reason codes** and **p50 <= 25 ms / p99
<= 50 ms with them**. ``riskscore bench`` prints the measured figures; on the
machine this was developed on they are 7.6/8.1 ms and 14.9/15.7 ms, so the
budget carries roughly 50% headroom for a slower CI runner.

Everything expensive has to have happened before the request arrives:

* the pickle is loaded once, when the app is built, not per request;
* the SHAP explainer is constructed once against the persisted background, so a
  request does no k-means and touches no training data;
* the one-row frame is built from a dict with **declared** dtypes, so pandas does
  no type inference on a single row - which is both slow and, for one row,
  frequently wrong (a single ``None`` in a numeric column makes it ``object``);
* nothing here opens a file.

Where the time actually goes, measured rather than assumed: about 5.5 ms in the
two pandas transformers, 1.5 ms in the ``ColumnTransformer``, and under 0.1 ms in
the model itself. Preprocessing *is* the request. That is why the optimizations
that mattered were pandas-pass-count reductions in
:mod:`risk_score.feature_engineering` and :mod:`risk_score.transformers` rather
than anything model-shaped, and why they sped up training by the same mechanism.

**Reason codes cost a second pass through the preprocessor**, which is the whole
7 ms difference between the two budgets above. The calibrator wraps the entire
pipeline, so a calibrated probability can only be had from the raw frame, while
the explainer needs the transformed matrix - and the two passes cannot be shared
without either reaching into ``CalibratedClassifierCV`` internals or changing
what the bundle calibrates. Calibrating the bare estimator instead of the
pipeline would make one pass serve both and is the upgrade path if this budget
ever binds; it is a bundle-format change, so it is not made lightly.

Batches do not pay this per row. 200 applicants cost about 8.8 ms in total -
0.04 ms each - because the cost is per *pass*, not per row.

Separated from the routes because the interesting failures are not HTTP failures.
"A required field was absent" and "the calibrated probability crossed the
threshold" are testable without a client, and a test that needs a client to
assert them is a test nobody writes enough of.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from risk_score.artifacts import RunMetadata, ScoringBundle
from risk_score.explain import DEFAULT_TOP_K, Contribution, Explainer
from risk_score.features import ParseKind, column_spec

_log = logging.getLogger(__name__)

#: One numpy dtype per declared parse kind, used to build the request frame
#: without inference.
#:
#: ``object`` for every kind whose source dialect may be a string - which is all
#: of them except plain numerics - because
#: :class:`~risk_score.transformers.CanonicalizeFrame` is what parses them, and
#: handing it a pre-coerced column would be a second parsing policy. Declaring
#: ``float64`` for ``term`` would reject ``' 36 months'`` here rather than in the
#: parser written to read it; the parsers already branch on dtype, so a column of
#: numbers in an ``object`` column costs one ``astype`` and nothing else.
_DTYPE_BY_PARSE_KIND: dict[ParseKind, str] = {
    ParseKind.NUMERIC: "float64",
    ParseKind.PERCENT: "object",
    ParseKind.TERM_MONTHS: "object",
    ParseKind.EMP_LENGTH_YEARS: "object",
    ParseKind.MONTH_DATE: "object",
    ParseKind.CATEGORY: "object",
    ParseKind.TEXT: "object",
}


@dataclass(frozen=True, slots=True)
class ReasonCode:
    """One contribution, flattened for a response body."""

    feature: str
    label: str
    value: Any
    log_odds: float
    direction: str


@dataclass(frozen=True, slots=True)
class Score:
    """What scoring one applicant produced.

    ``latency_ms`` is measured inside :meth:`ScoringService.score`, so it covers
    frame construction, preprocessing, the model, the calibrator and the
    explainer, and excludes HTTP framing. That is the number an operator can act
    on: the part this project controls.
    """

    default_probability: float
    decision: str
    threshold: float
    baseline_log_odds: float | None
    total_log_odds: float | None
    reasons: tuple[ReasonCode, ...]
    latency_ms: float

    @property
    def approved(self) -> bool:
        return self.decision == "approve"


class ScoringService:
    """A loaded bundle, its explainer, and the two things you can ask them.

    Constructed once per process. Building the explainer here rather than lazily
    on the first request is deliberate: a lazy build makes the first request after
    every deploy the slow one, which is exactly the request a smoke test measures.
    """

    def __init__(self, bundle: ScoringBundle, *, with_explainer: bool = True) -> None:
        self.bundle = bundle
        # Built now, and a failure to build recorded rather than raised. A linear
        # bundle with no persisted SHAP background cannot explain anything, and
        # finding that out per request would turn a missing artifact into a 500 on
        # the hot path; refusing to start would take a service that can still
        # score offline over a feature that is not the point of it.
        self.explainer: Explainer | None = None
        self.explainer_error: str | None = None
        if with_explainer:
            try:
                self.explainer = Explainer(bundle)
            except (ValueError, KeyError, AttributeError) as error:
                self.explainer_error = str(error)
                _log.warning("reason codes unavailable for %s: %s", bundle.metadata.run_id, error)
        self._dtypes = {
            name: _DTYPE_BY_PARSE_KIND[column_spec(name).parse]
            for name in bundle.feature_spec.raw_inputs
        }

    @property
    def metadata(self) -> RunMetadata:
        return self.bundle.metadata

    @property
    def can_explain(self) -> bool:
        return self.explainer is not None

    def build_frame(self, applicants: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """Rows of declared inputs, correctly typed, in the declared order.

        Declared dtypes rather than ``pd.DataFrame(rows)``, which infers per
        column and gets one row wrong in a way that matters: a numeric column
        whose only value is ``None`` infers as ``object``, and ``StandardScaler``
        then raises about a string it was never given.

        Absent keys become NA and are handled downstream exactly as an absent
        column in the training extract was - imputed, with the missing-indicator
        set. Unknown keys are dropped here rather than passed through, because
        ``CanonicalizeFrame`` would drop them anyway and dropping them early keeps
        the frame narrow.
        """
        columns = {
            name: pd.Series(
                [row.get(name) for row in applicants],
                dtype=dtype,
                name=name,
            )
            for name, dtype in self._dtypes.items()
        }
        return pd.DataFrame(columns, columns=list(self._dtypes))

    def score(
        self,
        applicant: Mapping[str, Any],
        *,
        with_reasons: bool = True,
        top_k: int = DEFAULT_TOP_K,
    ) -> Score:
        """One applicant, one probability, one decision, and why.

        ``perf_counter`` rather than ``time.time``: the latter is subject to NTP
        adjustment and can measure a negative duration, which is a confusing
        thing to find in a latency histogram.
        """
        started = time.perf_counter()
        frame = self.build_frame([applicant])
        probability = float(self.bundle.predict_probability(frame)[0])
        approved = bool(self.bundle.decide(np.asarray([probability]))[0])
        decision = "approve" if approved else "decline"

        baseline: float | None = None
        total: float | None = None
        reasons: tuple[ReasonCode, ...] = ()
        if with_reasons and self.explainer is not None:
            explanation = self.explainer.explain(frame)[0]
            baseline = explanation.baseline_log_odds
            total = explanation.total_log_odds
            reasons = tuple(_reason(item) for item in explanation.top(top_k))

        return Score(
            default_probability=probability,
            decision=decision,
            threshold=self.bundle.threshold,
            baseline_log_odds=baseline,
            total_log_odds=total,
            reasons=reasons,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def score_batch(
        self,
        applicants: Sequence[Mapping[str, Any]],
        *,
        with_reasons: bool = False,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[Score]:
        """Many applicants in one pass through the pipeline.

        Vectorized on purpose: scoring 1000 rows as 1000 one-row frames is
        roughly two orders of magnitude slower than one 1000-row frame, because
        the per-call cost is sklearn's transform overhead rather than the
        arithmetic.

        ``with_reasons`` defaults to **False** here and ``True`` on
        :meth:`score`, which is not an inconsistency: a batch is a scoring job and
        a single call is a decision somebody has to justify.
        """
        if not applicants:
            return []
        started = time.perf_counter()
        frame = self.build_frame(applicants)
        probabilities = np.asarray(self.bundle.predict_probability(frame), dtype=float)
        decisions = np.asarray(self.bundle.decide(probabilities), dtype=bool)

        explanations = (
            self.explainer.explain(frame)
            if with_reasons and self.explainer is not None
            else [None] * len(frame)
        )
        # One elapsed time divided across the batch. Timing each row separately
        # would report the cost of a vectorized call as if it were per row, and
        # the per-row figure is the one a caller compares against /predict.
        each_ms = (time.perf_counter() - started) * 1000.0 / len(frame)

        return [
            Score(
                default_probability=float(probability),
                decision="approve" if bool(approved) else "decline",
                threshold=self.bundle.threshold,
                baseline_log_odds=None if explanation is None else explanation.baseline_log_odds,
                total_log_odds=None if explanation is None else explanation.total_log_odds,
                reasons=()
                if explanation is None
                else tuple(_reason(item) for item in explanation.top(top_k)),
                latency_ms=each_ms,
            )
            for probability, approved, explanation in zip(
                probabilities, decisions, explanations, strict=True
            )
        ]


def _reason(contribution: Contribution) -> ReasonCode:
    """A contribution, with numpy scalars unwrapped so it can be serialized."""
    value = contribution.value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        value = None
    if isinstance(value, pd.Timestamp):
        value = value.isoformat()
    return ReasonCode(
        feature=contribution.feature,
        label=contribution.label,
        value=value,
        log_odds=contribution.log_odds,
        direction=contribution.direction,
    )
