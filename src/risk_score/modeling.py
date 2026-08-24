"""Time partitioning and the model pipeline, both built from declarations.

Two things changed here, and they are the same change seen from two ends.

**The split deleted the date column (and there was no validation set).** The old
``time_based_train_test_json`` dropped ``issue_d`` from both partitions, which
sounds like tidy hygiene and was actually load-bearing damage: it is the input
``credit_history_months`` is derived from. Keeping the model from seeing the
calendar is the :class:`~risk_score.transformers.FeatureSpec`'s job - the date is
simply not a declared feature - not the split's. So the split now hands the date
through and lets the declared contract decide what the model sees.

**The preprocessor inferred which transformer each column needed.** See
``transformers.py`` for the full account; the consequence here is that
:func:`build_preprocessor` takes a ``FeatureSpec`` rather than a DataFrame. It
cannot see the data at all, so it cannot be misled by it.

Ordering inside the assembled pipeline is the whole design::

    CanonicalizeFrame -> EngineerFeatures -> ColumnTransformer -> estimator
    |------------- stateless -------------| |----- fitted on train only -----|

Everything to the left of the bar learns nothing, so it may run on any partition
without carrying information between them. Everything to the right is fitted, and
is fitted exactly once, on train.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from risk_score.transformers import CanonicalizeFrame, EngineerFeatures, FeatureSpec

#: Model types this module can fit. Named here rather than as a dict of
#: functions so a caller can validate the argument before doing any work.
SUPPORTED_MODEL_TYPES: tuple[str, ...] = ("logistic_regression", "xgboost")

#: Boosting rounds without a validation-metric improvement before fitting stops.
#: 30 at a 0.05 learning rate is roughly a 1.5-unit stretch of no progress -
#: long enough to ride out the noise of a single unlucky round, short enough
#: that a 400-round budget is not spent memorizing the training vintages.
DEFAULT_EARLY_STOPPING_ROUNDS = 30

#: A category must cover at least this share of training rows to get its own
#: one-hot column; the rest are pooled into an "infrequent" level. 0.5% of rows
#: caps ``addr_state`` at roughly the 30 states with enough volume to estimate a
#: coefficient from, and pools the remainder instead of fitting noise per state.
DEFAULT_MIN_CATEGORY_FREQUENCY = 0.005

#: Logistic regression defaults. ``class_weight`` is deliberately ``None``.
#: The previous default was ``"balanced"``, which multiplies the minority class
#: weight by roughly 1/base-rate and therefore inflates every predicted
#: probability - and this project's headline outputs are a Brier score and a
#: calibration curve. The old artifact's calibration plot was not measuring a
#: miscalibrated model, it was measuring a deliberately reweighted one (audit
#: B05). Rebalancing changes the intercept, not the ranking, so AUC and KS are
#: unaffected; the threshold search is what handles class imbalance here.
DEFAULT_LOGISTIC_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": None,
    "random_state": 42,
}

#: XGBoost defaults. ``scale_pos_weight`` is absent for the same reason
#: ``class_weight`` is ``None`` above, and because the LR-versus-XGBoost
#: comparison is only meaningful if both models are weighted identically.
DEFAULT_XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    # 'hist' is the default from XGBoost 2.0 on, stated because it is the reason
    # a 1.8M-row fit is minutes rather than hours.
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A named, inclusive date range.

    Inclusive at both ends, and the end is expanded to the *end of the period it
    names*: ``"2014-09"`` means through 30 September, not 1 September. Writing a
    window as ``2013-01`` to ``2014-09`` is how anyone actually thinks about loan
    vintages, and a half-open interval silently dropped a month of them.
    """

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def parse(cls, name: str, bounds: TimeWindow | Sequence[str | pd.Timestamp]) -> Self:
        """Build a window from ``(start, end)``, expanding partial dates."""
        if isinstance(bounds, TimeWindow):
            return cls(name=name, start=bounds.start, end=bounds.end)
        if len(bounds) != 2:
            raise ValueError(
                f"Window `{name}` needs exactly (start, end); got {len(bounds)} values."
            )
        start, end = _start_of_period(bounds[0]), _end_of_period(bounds[1])
        if start > end:
            raise ValueError(f"Window `{name}` starts at {start} but ends at {end}.")
        return cls(name=name, start=start, end=end)

    def mask(self, dates: pd.Series) -> pd.Series:
        """Rows falling inside this window. ``NaT`` is never inside anything."""
        return dates.notna() & (dates >= self.start) & (dates <= self.end)

    def label(self) -> str:
        """``2013-01-01..2014-09-30``, for manifests and error messages."""
        return f"{self.start.date().isoformat()}..{self.end.date().isoformat()}"


def _start_of_period(value: str | pd.Timestamp) -> pd.Timestamp:
    """First instant of the period a partial date names."""
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Period(str(value)).start_time


def _end_of_period(value: str | pd.Timestamp) -> pd.Timestamp:
    """Last instant of the period a partial date names.

    ``pd.Period`` does the work: ``'2014'`` becomes 31 December, ``'2014-09'``
    becomes 30 September, and a full date becomes the end of that day. Hand-rolled
    month arithmetic here is where off-by-one-month split bugs come from.
    """
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Period(str(value)).end_time


@dataclass(frozen=True, slots=True)
class ValidationPartition:
    """The validation rows, wrapped so nothing else can be handed to a fitter.

    The calibrator is fitted, and it is fitted on validation only - see
    ``docs/decisions/0003-train-validation-test-split.md``. A function taking
    ``x`` and ``y`` cannot tell which partition it got; a function taking this
    can only be given the wrong one on purpose.

    :class:`risk_score.evaluation.ValidationScores` is the same idea for
    already-computed scores. Two types rather than one because the calibrator
    needs the features - it re-scores them through the frozen model - and the
    threshold search needs only the scores.
    """

    x: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class TimeSplit:
    """Three partitions, plus a full account of every row that reached none of them.

    The counts are not decoration. A run that quietly discarded a third of its
    rows to a window gap produces metrics indistinguishable from one that
    discarded none, and this was previously invisible: rows between the train
    cutoff and the test start were dropped in silence (audit B09).
    """

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    windows: tuple[TimeWindow, TimeWindow, TimeWindow]
    rows_in: int
    rows_unparseable_date: int
    rows_outside_windows: int

    @property
    def rows_out(self) -> int:
        """Rows that landed in some partition."""
        return len(self.x_train) + len(self.x_validation) + len(self.x_test)

    @property
    def validation(self) -> ValidationPartition:
        """The validation rows in the wrapper that fitted decisions require."""
        return ValidationPartition(x=self.x_validation, y=self.y_validation)

    def summary(self) -> str:
        """One-line summary for logs and the run manifest."""
        windows = " ".join(f"{window.name}={window.label()}" for window in self.windows)
        return (
            f"split {windows} "
            f"rows train={len(self.x_train)} validation={len(self.x_validation)} "
            f"test={len(self.x_test)} "
            f"dropped_gap={self.rows_outside_windows} "
            f"dropped_bad_date={self.rows_unparseable_date}"
        )


def _describe_range(dates: pd.Series) -> str:
    """Observed date coverage, safe on an all-``NaT`` column.

    The old error handler called ``.min().date()`` unconditionally, and
    ``NaT.date()`` raises. So a run whose dates all failed to parse - by far the
    likeliest reason for an empty partition - died inside the message that was
    supposed to explain it, and the real cause never reached the terminal
    (audit B08).
    """
    observed = dates.dropna()
    if observed.empty:
        return "no parseable dates at all"
    return f"{observed.min().date().isoformat()} to {observed.max().date().isoformat()}"


def split_by_time(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    train: TimeWindow | Sequence[str | pd.Timestamp],
    validation: TimeWindow | Sequence[str | pd.Timestamp],
    test: TimeWindow | Sequence[str | pd.Timestamp],
    date_column: str = "issue_d",
) -> TimeSplit:
    """Partition by origination date into train, validation, and test.

    Three partitions, because fitting a threshold or a calibrator on the same
    rows the headline metrics are reported from makes those metrics a description
    of the fitting procedure rather than of future performance (audit B04).
    Validation is where every decision derived from scores gets made; test is
    scored once and reported.

    The date column is **kept** in all three frames. It is an input to
    ``credit_history_months``, and what stops the model from fitting the calendar
    is that ``issue_d`` is not a declared feature - see ``transformers.py``.
    """
    if date_column not in features.columns:
        raise KeyError(
            f"Date column `{date_column}` is missing from features. "
            f"Present: {sorted(features.columns)[:20]}."
        )
    if len(features) != len(target):
        raise ValueError(
            f"Features and target must be the same length; got {len(features)} and {len(target)}."
        )
    if not features.index.equals(target.index):
        # Positional alignment would silently pair the wrong label with the wrong
        # applicant, and every metric would still look plausible.
        raise ValueError("Features and target must share an index.")

    dates = features[date_column]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise TypeError(
            f"`{date_column}` must already be datetime, got {dates.dtype}. "
            f"Run risk_score.schema.normalize_credit_schema first; re-parsing here "
            f"would apply a second, different date-format policy (audit B28)."
        )

    windows = (
        TimeWindow.parse("train", train),
        TimeWindow.parse("validation", validation),
        TimeWindow.parse("test", test),
    )
    _reject_overlapping_windows(windows)
    if dates.dt.tz is not None:
        windows = tuple(  # type: ignore[assignment]
            TimeWindow(
                name=window.name,
                start=window.start.tz_localize(dates.dt.tz),
                end=window.end.tz_localize(dates.dt.tz),
            )
            for window in windows
        )

    # Masks computed straight off the column. The old version assigned a
    # `_split_date` helper column and dropped it afterwards, which silently
    # destroyed any real column of that name (audit B32); not creating one at all
    # is both shorter and impossible to get wrong.
    masks = {window.name: window.mask(dates) for window in windows}

    partitions: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for window in windows:
        selected = dates.loc[masks[window.name]]
        # Sorted by date so XGBoost's eval_set and any vintage breakdown see the
        # rows in the order they were originated. Sorting the masked dates rather
        # than argsorting the whole column keeps NaT out of the ordering entirely.
        order = selected.sort_values(kind="stable").index
        partitions[window.name] = (features.loc[order], target.loc[order])

    empty = [name for name, (frame, _) in partitions.items() if frame.empty]
    if empty:
        raise ValueError(
            f"Time-based split produced an empty partition: {empty}. "
            f"Requested {', '.join(f'{w.name} {w.label()}' for w in windows)}, "
            f"but the data covers {_describe_range(dates)}. "
            f"Row counts: {', '.join(f'{n}={len(f)}' for n, (f, _) in partitions.items())}."
        )

    in_any = masks["train"] | masks["validation"] | masks["test"]
    return TimeSplit(
        x_train=partitions["train"][0],
        x_validation=partitions["validation"][0],
        x_test=partitions["test"][0],
        y_train=partitions["train"][1],
        y_validation=partitions["validation"][1],
        y_test=partitions["test"][1],
        windows=windows,
        rows_in=len(features),
        rows_unparseable_date=int(dates.isna().sum()),
        rows_outside_windows=int((dates.notna() & ~in_any).sum()),
    )


def _reject_overlapping_windows(windows: Sequence[TimeWindow]) -> None:
    """Windows must be disjoint and chronological.

    Overlap is the one split mistake that never surfaces as an error: the model
    simply scores rows it was fitted on, every metric improves, and the run looks
    like a success.
    """
    for earlier, later in itertools.pairwise(windows):
        if later.start <= earlier.end:
            raise ValueError(
                f"Window `{later.name}` starts at {later.start.date()} but "
                f"`{earlier.name}` runs through {earlier.end.date()}. Overlapping "
                f"windows leak training rows into evaluation without any error."
            )


def build_preprocessor(
    spec: FeatureSpec,
    *,
    min_category_frequency: float = DEFAULT_MIN_CATEGORY_FREQUENCY,
) -> ColumnTransformer:
    """Assemble the fitted preprocessing stage from the *declared* column lists.

    Takes a spec, not a DataFrame, which is the entire point: this function
    cannot see the data, so the data cannot mislead it. The previous version
    called ``select_dtypes`` and therefore sent every string pandas had failed to
    parse - dates, percent strings - to ``OneHotEncoder`` (audit B01, B22, B23).
    """
    numeric = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    # A missing value in credit data is informative: a borrower with
                    # no `revol_util` has no revolving account, which is not the
                    # same as one sitting at the median.
                    add_indicator=True,
                    # Without this, a column that is entirely missing in train is
                    # silently dropped and the design matrix is narrower than the
                    # spec declares - which only shows up at serving time, as a
                    # width mismatch with no column name in the message.
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(
        # Unseen categories join the infrequent bucket rather than raising or
        # becoming an all-zero row: a new `addr_state` in a serving request is
        # ordinary, and `handle_unknown="ignore"` would encode it as "no state".
        handle_unknown="infrequent_if_exist",
        min_frequency=min_category_frequency,
        # The other half of the 35 GB fix: the old encoder emitted dense float
        # columns. Both LogisticRegression and XGBoost accept CSR, so nothing
        # downstream needs a dense copy of a mostly-zero block.
        sparse_output=True,
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if spec.numeric_features:
        transformers.append(("numeric", numeric, list(spec.numeric_features)))
    if spec.categorical_features:
        transformers.append(("categorical", categorical, list(spec.categorical_features)))
    if not transformers:
        raise ValueError(f"FeatureSpec declares no columns to transform: {spec.summary()}.")

    return ColumnTransformer(
        transformers=transformers,
        # No `remainder="passthrough"`: EngineerFeatures already reduced the frame
        # to exactly the declared features, and passthrough would quietly
        # readmit anything a future change forgot to drop.
        remainder="drop",
        # `numeric__loan_amnt` is unreadable in a reason code. Names collide only
        # if a numeric column is literally named like a one-hot level, and
        # ColumnTransformer raises rather than colliding silently.
        verbose_feature_names_out=False,
        # sklearn's default, stated because it decides the memory layout. The
        # stacked result is CSR only when the *combined* density falls below this,
        # so a wide one-hot block gives sparse output while a narrow one gives
        # dense - and dense is genuinely cheaper there. Declared routing is what
        # made both regimes acceptable; with 2400 inferred one-hot columns the
        # dense branch was the 35 GB failure.
        sparse_threshold=0.3,
    )


def build_model_pipeline(
    spec: FeatureSpec,
    estimator: Any,
    *,
    min_category_frequency: float = DEFAULT_MIN_CATEGORY_FREQUENCY,
) -> Pipeline:
    """The complete estimator: raw frame in, default probability out.

    One object, so training and ``POST /predict`` cannot diverge - there is no
    second code path to keep in sync, because there is no second code path.
    """
    return Pipeline(
        steps=[
            ("canonicalize", CanonicalizeFrame(spec)),
            ("engineer", EngineerFeatures(spec)),
            ("preprocess", build_preprocessor(spec, min_category_frequency=min_category_frequency)),
            ("classifier", estimator),
        ]
    )


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    spec: FeatureSpec,
    config: dict[str, Any] | None = None,
    min_category_frequency: float = DEFAULT_MIN_CATEGORY_FREQUENCY,
) -> Pipeline:
    """Fit the regularized logistic regression baseline on the training rows only."""
    params = {**DEFAULT_LOGISTIC_PARAMS, **(config or {})}
    pipeline = build_model_pipeline(
        spec, LogisticRegression(**params), min_category_frequency=min_category_frequency
    )
    return pipeline.fit(x_train, y_train)


def fit_with_validation_monitoring(
    spec: FeatureSpec,
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    *,
    min_category_frequency: float = DEFAULT_MIN_CATEGORY_FREQUENCY,
) -> Pipeline:
    """Fit an estimator that watches a validation set, and return one Pipeline.

    Early stopping needs an *already transformed* validation matrix, and
    ``Pipeline.fit`` has nowhere to put one: passing raw validation rows through
    ``classifier__eval_set`` would hand the estimator a DataFrame of strings. So
    the preprocessing prefix is fitted and applied explicitly here, the estimator
    is fitted against both matrices, and the pipeline is *reassembled* from the
    already-fitted steps.

    The reassembly is sound because ``Pipeline`` does not clone the steps it is
    given - the objects in the returned pipeline are the ones that were just
    fitted, and a test asserts the assembled pipeline's predictions equal this
    function's own two-step predictions.

    Nothing about the validation partition reaches a ``fit`` call other than the
    estimator's own early-stopping monitor: the medians, the scaling moments, and
    the category lists all come from ``transform``, which cannot learn.
    """
    steps = build_model_pipeline(
        spec, estimator, min_category_frequency=min_category_frequency
    ).steps
    preprocessing = Pipeline(steps=steps[:-1])

    matrix_train = preprocessing.fit_transform(x_train, y_train)
    # `transform`, not `fit_transform`. This is the line the whole no-leakage
    # claim rests on, which is why it is one line away from the one above it.
    matrix_validation = preprocessing.transform(x_validation)

    estimator.fit(
        matrix_train,
        y_train,
        eval_set=[(matrix_validation, y_validation)],
        verbose=False,
    )
    return Pipeline(steps=[*preprocessing.steps, steps[-1]])


def train_xgboost_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    spec: FeatureSpec,
    config: dict[str, Any] | None = None,
    min_category_frequency: float = DEFAULT_MIN_CATEGORY_FREQUENCY,
    x_validation: pd.DataFrame | None = None,
    y_validation: pd.Series | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> Pipeline:
    """Fit the gradient-boosted model, stopping early on validation if given.

    Without a validation partition the model spends its whole ``n_estimators``
    budget, which on 1.8M rows is both slower and worse. With one, boosting stops
    when the validation metric stops improving and ``predict_proba`` uses the
    best iteration rather than the last.
    """
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - depends on the local libomp
        # Not just ImportError. An installed-but-unloadable wheel raises
        # XGBoostError from inside the import - a forty-line dlopen dump whose
        # actual instruction, `brew install libomp`, is buried in the middle of
        # it. Catching only ImportError let that reach the terminal unedited.
        raise ImportError(
            f"XGBoost is installed but could not be loaded: {type(exc).__name__}. "
            f"On macOS the wheel needs the OpenMP runtime: brew install libomp. "
            f"Otherwise install the train extra: pip install -e '.[train]'."
        ) from exc

    params = {**DEFAULT_XGBOOST_PARAMS, **(config or {})}
    monitored = x_validation is not None and y_validation is not None
    if monitored:
        # Only set with an eval_set present: XGBoost raises if asked to stop
        # early with nothing to measure.
        params.setdefault("early_stopping_rounds", early_stopping_rounds)
    estimator = XGBClassifier(**params)

    if not monitored:
        pipeline = build_model_pipeline(
            spec, estimator, min_category_frequency=min_category_frequency
        )
        return pipeline.fit(x_train, y_train)

    assert x_validation is not None and y_validation is not None  # narrowed by `monitored`
    return fit_with_validation_monitoring(
        spec,
        estimator,
        x_train,
        y_train,
        x_validation,
        y_validation,
        min_category_frequency=min_category_frequency,
    )


def train_model(
    model_type: str,
    split: TimeSplit,
    *,
    spec: FeatureSpec,
    config: dict[str, Any] | None = None,
    min_category_frequency: float = DEFAULT_MIN_CATEGORY_FREQUENCY,
) -> Pipeline:
    """Fit one model type on a split, giving each what it can legitimately use.

    Dispatching here rather than through a uniform trainer signature is
    deliberate. A shared ``(x_train, y_train, x_validation, y_validation)``
    signature would make the logistic baseline accept a validation partition it
    then ignores, which reads like an oversight and invites someone to "fix" it
    by fitting on it. Only the model that has a use for validation is handed it.
    """
    if model_type == "logistic_regression":
        return train_logistic_regression(
            split.x_train,
            split.y_train,
            spec=spec,
            config=config,
            min_category_frequency=min_category_frequency,
        )
    if model_type == "xgboost":
        return train_xgboost_model(
            split.x_train,
            split.y_train,
            spec=spec,
            config=config,
            min_category_frequency=min_category_frequency,
            x_validation=split.x_validation,
            y_validation=split.y_validation,
        )
    raise ValueError(
        f"Supported model types are {list(SUPPORTED_MODEL_TYPES)}; got {model_type!r}."
    )
