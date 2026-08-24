"""Why this applicant got this score: exact per-feature contributions.

A credit decision that cannot be explained is, in several jurisdictions, a
decision that cannot legally be made - adverse action notices under ECOA/FCRA
have to state *reasons*, not a probability. So reason codes are not a portfolio
flourish here; they are the difference between a score and a decision.

What this module computes
------------------------
SHAP values: the contribution of each feature to *this* row's log-odds,
measured against a background population, such that

    log_odds(applicant) = baseline + sum(contributions)

That identity is the whole reason to use SHAP rather than, say, coefficient
magnitudes or a permutation ranking. It is additive and exact, so a reason code
is a number a reviewer can check, and :func:`Explanation.total_log_odds` is
asserted against the model's own ``decision_function`` in the tests.

Why there is no ``shap`` dependency
-----------------------------------
Because for the two model families this project fits, the exact SHAP values are
already available without it.

**Linear models.** For a linear model with independent features, the
interventional SHAP value of feature *i* is exactly ``coef_i * (x_i - E[x_i])``.
That is one multiply, and it is what ``shap.LinearExplainer`` computes - a
one-line derivation, not an approximation of one.

**Gradient-boosted trees.** XGBoost implements TreeSHAP *inside the booster*:
``booster.predict(dmatrix, pred_contribs=True)`` returns the exact contributions
plus a bias column. ``shap.TreeExplainer`` calls into that same code.

So the ``shap`` package would add a numba/llvmlite toolchain to the serving
image, and a wheel-availability risk on new Python versions, to reach code paths
this module reaches in a dozen lines. Both paths are pinned by tests that
reconstruct the model's own margin from the contributions, which is a stronger
check than "it agrees with the library that produced it".

Explanations are in the **uncalibrated log-odds** of the fitted pipeline.
Calibration is a monotone map applied afterwards, so it changes the probability
but not the ranking or the sign of any contribution - and the log-odds scale is
the only one on which contributions are additive at all.

One-hot families collapse
-------------------------
The model sees ``purpose_debt_consolidation``; the applicant has a ``purpose``.
Reporting 40 near-zero one-hot columns as 40 reasons is useless, so
contributions are summed back to the *source* feature - which is also the field
name the request payload uses, so a reason code names something the caller sent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline

from risk_score.artifacts import ScoringBundle
from risk_score.features import COLUMN_REGISTRY, ENGINEERED_BY_NAME
from risk_score.modeling import engineering_prefix

LOGGER = logging.getLogger(__name__)

#: How many transformed training rows a bundle carries so an explainer can be
#: built without the training data. 200 is well past enough to estimate the
#: column means the linear path needs - the standard error of a mean over
#: standardized features is 1/sqrt(200), about 0.07, against coefficients whose
#: reason-code ordering is not decided at that resolution - and it keeps the
#: array around a quarter of a megabyte.
BACKGROUND_ROWS = 200

#: Reason codes returned per applicant unless the caller asks for more. Five is
#: what an adverse action notice has room for, and beyond the top few the
#: contributions of a model this size are noise.
DEFAULT_TOP_K = 5

#: The name a missing-value indicator column takes, from
#: ``SimpleImputer(add_indicator=True)``. Stripped so the indicator's
#: contribution is attributed to the feature it is about.
_INDICATOR_PREFIX = "missingindicator_"


# --- naming --------------------------------------------------------------------


def feature_label(name: str) -> str:
    """A human sentence for a feature, from the registry that declares it.

    Read from ``description`` rather than from a label table in this module,
    because a second table is a second thing to keep in sync and the registry is
    already the place a new feature has to be declared.
    """
    if name in COLUMN_REGISTRY:
        return COLUMN_REGISTRY[name].description or name
    if name in ENGINEERED_BY_NAME:
        return ENGINEERED_BY_NAME[name].description or name
    return name


def transformed_feature_names(pipeline: Pipeline) -> tuple[str, ...]:
    """The design matrix's column names, in matrix order.

    Available because ``build_preprocessor`` sets
    ``verbose_feature_names_out=False``, so a one-hot column is
    ``purpose_car`` rather than ``categorical__purpose_car`` - readable in a
    reason code, and parseable back to its source below.
    """
    return tuple(str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out())


def _source_of(name: str, declared: frozenset[str]) -> str:
    """Which model feature a design-matrix column came from."""
    if name in declared:
        return name
    bare = name.removeprefix(_INDICATOR_PREFIX)
    if bare in declared:
        return bare
    # One-hot columns are `<feature>_<level>`, and both parts can contain
    # underscores (`home_ownership_MORTGAGE`, `purpose_debt_consolidation`). The
    # *longest* declared prefix is the right answer: if ever both `purpose` and
    # `purpose_group` were features, `purpose_group_x` belongs to the longer one.
    candidates = [feature for feature in declared if bare.startswith(f"{feature}_")]
    if candidates:
        return max(candidates, key=len)
    # Reported under its own name rather than dropped. An unattributable column
    # means the preprocessor grew a step this function does not know about, and
    # a reason code with an odd name is a far better failure than a silently
    # incomplete explanation that no longer sums to the score.
    LOGGER.warning("cannot attribute design-matrix column %r to a declared feature", name)
    return bare


def feature_sources(pipeline: Pipeline) -> tuple[str, ...]:
    """The source feature for each design-matrix column, in matrix order."""
    declared = frozenset(pipeline.named_steps["engineer"].spec.model_features)
    return tuple(_source_of(name, declared) for name in transformed_feature_names(pipeline))


# --- the background ------------------------------------------------------------


def _preprocessing_prefix(pipeline: Pipeline) -> Pipeline:
    """Canonicalize -> engineer -> preprocess. Everything but the estimator."""
    return Pipeline(steps=pipeline.steps[:-1])


def _densify(matrix: Any) -> np.ndarray[Any, Any]:
    """A dense float64 view of a design matrix.

    The preprocessor emits CSR when the one-hot block dominates. Every path here
    is at most a few hundred columns by at most a few hundred rows, so densifying
    costs kilobytes and buys ordinary numpy arithmetic.
    """
    dense = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
    return np.asarray(dense, dtype=np.float64)


def build_background(
    pipeline: Pipeline,
    applicants: pd.DataFrame,
    *,
    max_rows: int = BACKGROUND_ROWS,
    random_state: int = 42,
) -> np.ndarray[Any, Any]:
    """Transformed training rows for the explainer to measure against.

    A **uniform random sample**, not a k-means summary. The linear path needs the
    background's column means to be the training population's column means, and
    the unweighted mean of k cluster centroids is not that - it over-weights
    sparse regions in exactly the tails where a reason code matters.

    ``float32`` because this is sampled data going into a mean, so the second
    half of a float64 mantissa is storing sampling noise, and the array is
    pickled into every bundle.
    """
    if max_rows < 1:
        raise ValueError(f"`max_rows` must be at least 1; got {max_rows}.")
    matrix = _densify(_preprocessing_prefix(pipeline).transform(applicants))
    if len(matrix) > max_rows:
        rng = np.random.default_rng(random_state)
        # Sorted so the sample keeps the frame's (chronological) row order and the
        # array is byte-identical across runs on the same data.
        matrix = matrix[np.sort(rng.choice(len(matrix), size=max_rows, replace=False))]
    return matrix.astype(np.float32)


# --- explanations --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Contribution:
    """One feature's push on one applicant's log-odds of default."""

    feature: str
    label: str
    #: The applicant's value for this feature *after* engineering and before
    #: scaling, so the reason code cites 65000 rather than -0.42.
    value: Any
    log_odds: float

    @property
    def direction(self) -> str:
        """``"increases risk"`` / ``"reduces risk"`` / ``"no effect"``."""
        if self.log_odds > 0:
            return "increases risk"
        return "reduces risk" if self.log_odds < 0 else "no effect"

    def sentence(self) -> str:
        """One line for an adverse action notice or a dashboard row."""
        return f"{self.feature}={self.value} {self.direction} ({self.log_odds:+.3f} log-odds)"


@dataclass(frozen=True, slots=True)
class Explanation:
    """Every source feature's contribution for one applicant, largest first.

    Complete rather than truncated: :meth:`top` is what a caller displays, but
    the full set is what makes :attr:`total_log_odds` reconstruct the model's own
    margin, which is the property that says the explanation is not approximate.
    """

    baseline_log_odds: float
    contributions: tuple[Contribution, ...]

    @property
    def total_log_odds(self) -> float:
        """The model's margin for this row, rebuilt from the parts."""
        return self.baseline_log_odds + sum(item.log_odds for item in self.contributions)

    def top(self, k: int = DEFAULT_TOP_K) -> tuple[Contribution, ...]:
        """The ``k`` largest contributions by magnitude, either direction."""
        if k < 1:
            raise ValueError(f"`k` must be at least 1; got {k}.")
        return self.contributions[:k]

    def adverse_reasons(self, k: int = DEFAULT_TOP_K) -> tuple[Contribution, ...]:
        """Only the contributions that pushed *towards* default.

        What a declined applicant is owed. A rejection explained by the two
        things that helped them is not an explanation.
        """
        return tuple(item for item in self.contributions if item.log_odds > 0)[:k]


class Explainer:
    """A pre-fit explainer: built once from a bundle, then called per applicant.

    Everything expensive - reading the coefficients, averaging the background,
    resolving the design-matrix column names back to source features - happens
    here, in ``__init__``, so the service builds one at boot and ``/predict`` does
    arithmetic on an array it already has.

    Constructed from the bundle alone. No training data, no artifact directory,
    and no second source of truth about which features the model uses.
    """

    def __init__(self, bundle: ScoringBundle) -> None:
        self.pipeline = bundle.pipeline
        self.spec = bundle.feature_spec
        self._engineer = engineering_prefix(bundle.pipeline)
        self._preprocess = bundle.pipeline.named_steps["preprocess"]
        self._sources = feature_sources(bundle.pipeline)
        # Grouping indices computed once: a reason code is a sum over the columns
        # of a one-hot family, and doing that with a per-request groupby would put
        # pandas on the hot path for arithmetic numpy does in microseconds.
        self.features: tuple[str, ...] = tuple(dict.fromkeys(self._sources))
        self._groups = tuple(
            np.flatnonzero(np.asarray(self._sources) == feature) for feature in self.features
        )

        estimator = bundle.pipeline.steps[-1][1]
        self.model_kind = _classify_estimator(estimator)
        self._estimator = estimator
        if self.model_kind == "linear":
            background = bundle.shap_background
            if background is None:
                raise ValueError(
                    "A linear model's contributions are measured against the "
                    "training population's means, and this bundle carries no "
                    "`shap_background`. Retrain: reason codes cannot be "
                    "reconstructed from the model alone."
                )
            mean = np.asarray(background, dtype=np.float64).mean(axis=0)
            coefficients = np.ravel(np.asarray(estimator.coef_, dtype=np.float64))
            if mean.shape != coefficients.shape:
                raise ValueError(
                    f"Bundle background has {mean.size} columns but the model has "
                    f"{coefficients.size} coefficients. The background was built "
                    "from a different preprocessor than the one that was fitted."
                )
            self._coefficients = coefficients
            self._mean = mean
            # The log-odds of the background population itself: what the model
            # would say about an applicant who is average in every feature.
            self._baseline = float(np.ravel(estimator.intercept_)[0] + coefficients @ mean)

    # --- the contributions themselves ------------------------------------------

    def _column_contributions(self, matrix: Any) -> tuple[np.ndarray[Any, Any], float]:
        """Per design-matrix column, plus the baseline. The only model-specific code."""
        if self.model_kind == "linear":
            # Exact interventional SHAP for a linear model: coef * (x - E[x]).
            return (_densify(matrix) - self._mean) * self._coefficients, self._baseline
        return _tree_contributions(self._estimator, matrix)

    def explain(self, applicants: pd.DataFrame) -> list[Explanation]:
        """One :class:`Explanation` per row, in the frame's order.

        Takes a **raw** frame - the same shape ``/predict`` receives - because the
        canonicalization and engineering steps are inside the pipeline, so an
        explanation is produced from exactly the columns that produced the score.

        Always complete. Truncating here would leave a baseline that no longer
        sums with the parts, and :meth:`Explanation.top` is a free slice.
        """
        if not isinstance(applicants, pd.DataFrame):
            raise TypeError(
                f"Expected a DataFrame of raw applicant rows, got {type(applicants).__name__}."
            )
        # Engineered once and used twice: transformed for the arithmetic, and read
        # for the values the reason codes cite.
        engineered = self._engineer.transform(applicants)
        columns, baseline = self._column_contributions(self._preprocess.transform(engineered))
        grouped = np.column_stack([columns[:, index].sum(axis=1) for index in self._groups])

        explanations: list[Explanation] = []
        for position in range(len(engineered)):
            row = grouped[position]
            # Sorted by magnitude here rather than at display time, so `top(k)`
            # and `adverse_reasons(k)` are both slices of one ordering.
            order = np.argsort(-np.abs(row))
            contributions = tuple(
                Contribution(
                    feature=self.features[index],
                    label=feature_label(self.features[index]),
                    value=_display_value(engineered, self.features[index], position),
                    log_odds=float(row[index]),
                )
                for index in order
            )
            explanations.append(
                Explanation(baseline_log_odds=float(baseline), contributions=contributions)
            )
        return explanations

    def global_summary(self, applicants: pd.DataFrame) -> pd.DataFrame:
        """Mean absolute contribution per feature over a population, ranked.

        The global importance artifact. Mean *absolute* value, because a feature
        that pushes half the population up and half down is important and its
        signed mean is zero; the signed mean is reported beside it so the
        direction is still visible.
        """
        engineered = self._engineer.transform(applicants)
        columns, _baseline = self._column_contributions(self._preprocess.transform(engineered))
        grouped = np.column_stack([columns[:, index].sum(axis=1) for index in self._groups])
        summary = pd.DataFrame(
            {
                "feature": list(self.features),
                "label": [feature_label(name) for name in self.features],
                "mean_abs_log_odds": np.abs(grouped).mean(axis=0),
                "mean_log_odds": grouped.mean(axis=0),
                "columns": [len(index) for index in self._groups],
            }
        )
        summary = summary.sort_values("mean_abs_log_odds", ascending=False, ignore_index=True)
        return summary.assign(rank=np.arange(1, len(summary) + 1))


def _classify_estimator(estimator: Any) -> str:
    """``"linear"`` or ``"tree"``, or refuse.

    Duck-typed on the attribute each path needs rather than on the class, so an
    ``SGDClassifier`` or an ``XGBRFClassifier`` works without an entry here, and
    a model neither path can explain fails at construction rather than returning
    plausible nonsense.
    """
    if hasattr(estimator, "get_booster"):
        return "tree"
    if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
        return "linear"
    raise TypeError(
        f"Cannot explain a {type(estimator).__name__}: it has neither `coef_` "
        "(exact linear SHAP) nor `get_booster` (XGBoost's own TreeSHAP)."
    )


def _tree_contributions(estimator: Any, matrix: Any) -> tuple[np.ndarray[Any, Any], float]:
    """Exact TreeSHAP from the booster, in margin space.

    ``pred_contribs=True`` returns one column per feature plus a trailing bias
    column, which is the baseline. This is the same C++ implementation
    ``shap.TreeExplainer`` dispatches to for XGBoost models.
    """
    import xgboost

    booster = estimator.get_booster()
    contributions = np.asarray(
        booster.predict(xgboost.DMatrix(matrix), pred_contribs=True), dtype=np.float64
    )
    # The bias column is constant across rows by construction, so taking row 0 is
    # not an approximation.
    return contributions[:, :-1], float(contributions[0, -1])


def _display_value(engineered: pd.DataFrame, feature: str, position: int) -> Any:
    """The applicant's own value for a feature, JSON-safe.

    Numpy scalars and ``pd.NA`` do not survive ``json.dumps``, and a reason code
    exists to be shown to somebody, so the conversion happens here rather than in
    every consumer.
    """
    if feature not in engineered.columns:
        return None
    value = engineered.iloc[position][feature]
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value
