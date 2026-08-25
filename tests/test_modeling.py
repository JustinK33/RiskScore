"""Tests for the time partition and the assembled model pipeline.

The split tests use hand-built frames with two columns, because
:func:`split_by_time` reads exactly one of them. The preprocessor and pipeline
tests use the synthetic raw extract, because their whole claim is about how a
*raw* frame is routed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from risk_score.modeling import (
    DEFAULT_LOGISTIC_PARAMS,
    TimeSplit,
    TimeWindow,
    build_model_pipeline,
    build_preprocessor,
    categorical_choices,
    engineering_prefix,
    fit_with_validation_monitoring,
    split_by_time,
    train_logistic_regression,
    train_model,
    train_xgboost_model,
)
from risk_score.schema import normalize_credit_schema
from risk_score.transformers import FeatureSpec, build_feature_spec
from tests.conftest import requires_xgboost

TRAIN = ("2013-01", "2014-12")
VALIDATION = ("2015-01", "2015-06")
TEST = ("2015-07", "2015-12")


def _dense(matrix: Any) -> npt.NDArray[np.float64]:
    """The transformed matrix as an array, whichever layout it came back in.

    ``ColumnTransformer`` stacks to CSR only when the combined density is low
    enough, and on four rows it is not - so a test that asserts values has to
    handle both, and a test that asserts *layout* is asserting the size of its
    own fixture.
    """
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def dated(months: list[str | None], values: list[int] | None = None) -> pd.DataFrame:
    """A two-column frame: the date the split reads, and a value to trace."""
    return pd.DataFrame(
        {
            "issue_d": pd.to_datetime(months),
            "loan_amnt": values if values is not None else list(range(len(months))),
        }
    )


# --- TimeWindow ----------------------------------------------------------------


def test_a_window_end_covers_the_whole_month_it_names() -> None:
    """`2014-09` must mean through 30 September, not 1 September.

    Asserted by containment rather than by comparing ``end`` to a literal
    timestamp: the last instant of a period is a pandas resolution detail
    (microseconds today, nanoseconds before), and the contract is the boundary.
    """
    window = TimeWindow.parse("train", ("2013-01", "2014-09"))
    dates = pd.to_datetime(
        [
            "2012-12-31 00:00:00",
            "2013-01-01 00:00:00",
            "2014-09-01 00:00:00",
            "2014-09-30 23:00:00",
            "2014-10-01 00:00:00",
        ]
    )

    assert window.start == pd.Timestamp("2013-01-01")
    assert window.mask(pd.Series(dates)).tolist() == [False, True, True, True, False]
    assert window.label() == "2013-01-01..2014-09-30"


def test_a_window_end_covers_the_whole_year_it_names() -> None:
    window = TimeWindow.parse("train", ("2013", "2014"))
    dates = pd.Series(pd.to_datetime(["2014-12-31 18:00:00", "2015-01-01 00:00:00"]))
    assert window.mask(dates).tolist() == [True, False]
    assert window.label() == "2013-01-01..2014-12-31"


def test_a_full_date_is_inclusive_of_the_day_it_names() -> None:
    """A window ending 2015-12-31 must contain a loan issued on 2015-12-31."""
    window = TimeWindow.parse("test", ("2015-04-01", "2015-12-31"))
    dates = pd.Series(
        pd.to_datetime(["2015-12-31 00:00:00", "2015-12-31 23:00:00", "2016-01-01 00:00:00"])
    )
    assert window.mask(dates).tolist() == [True, True, False]


def test_a_window_ending_before_it_starts_is_rejected() -> None:
    with pytest.raises(ValueError, match="starts at"):
        TimeWindow.parse("train", ("2015-01", "2014-01"))


def test_a_window_needs_exactly_two_bounds() -> None:
    with pytest.raises(ValueError, match="exactly \\(start, end\\)"):
        TimeWindow.parse("train", ("2013-01", "2014-01", "2015-01"))


def test_parsing_an_already_parsed_window_is_the_identity() -> None:
    """`TimeWindow.parse` accepts its own output, unchanged.

    It matters because the config layer calls it on whatever the caller supplied,
    and a `TimeWindow` reaching it a second time must not have its end expanded
    again: `_end_of_period` on a Timestamp is a pass-through precisely so that
    2014-09-30 does not become 2014-09-30 23:59:59.999999999 on the second trip.
    """
    once = TimeWindow.parse("train", ("2013-01", "2014-09"))
    twice = TimeWindow.parse("train", once)

    assert (twice.start, twice.end) == (once.start, once.end)
    assert twice.name == "train"

    # The same pass-through, reached through the tuple form with real Timestamps.
    stamped = TimeWindow.parse("train", (pd.Timestamp("2013-01-15"), pd.Timestamp("2014-09-20")))
    assert stamped.label() == "2013-01-15..2014-09-20"


def test_a_window_never_contains_a_missing_date() -> None:
    dates = pd.Series(pd.to_datetime(["2014-06-01", None]))
    assert TimeWindow.parse("train", ("2013-01", "2014-12")).mask(dates).tolist() == [True, False]


# --- split_by_time -------------------------------------------------------------


def test_rows_land_in_the_window_that_contains_their_issue_date() -> None:
    features = dated(
        ["2013-05-01", "2014-12-31", "2015-01-01", "2015-06-30", "2015-07-01", "2015-12-01"],
        [10, 20, 30, 40, 50, 60],
    )
    target = pd.Series([0, 1, 0, 1, 0, 1])

    split = split_by_time(features, target, train=TRAIN, validation=VALIDATION, test=TEST)

    assert split.x_train["loan_amnt"].tolist() == [10, 20]
    assert split.x_validation["loan_amnt"].tolist() == [30, 40]
    assert split.x_test["loan_amnt"].tolist() == [50, 60]
    assert split.y_train.tolist() == [0, 1]
    assert split.y_test.tolist() == [0, 1]


def test_partitions_come_back_in_date_order_whatever_the_input_order() -> None:
    features = dated(
        ["2014-12-01", "2013-01-01", "2014-06-01", "2015-02-01", "2015-08-01"],
        [3, 1, 2, 4, 5],
    )
    target = pd.Series([0, 0, 1, 0, 1])

    split = split_by_time(features, target, train=TRAIN, validation=VALIDATION, test=TEST)

    assert split.x_train["loan_amnt"].tolist() == [1, 2, 3]
    assert split.y_train.tolist() == [0, 1, 0]


def test_the_date_column_survives_the_split() -> None:
    """It is the input to credit_history_months; the spec is what hides it."""
    split = split_by_time(
        dated(["2013-05-01", "2015-02-01", "2015-08-01"]),
        pd.Series([0, 1, 0]),
        train=TRAIN,
        validation=VALIDATION,
        test=TEST,
    )
    assert "issue_d" in split.x_train.columns
    assert split.x_train["issue_d"].tolist() == [pd.Timestamp("2013-05-01")]


def test_b09_rows_falling_between_windows_are_counted_not_silently_dropped() -> None:
    """A gap row is a legitimate exclusion; an unreported one is a lie about n."""
    features = dated(
        # The middle row is inside no window: train ends 2014-12, validation
        # starts 2015-01, so 2014-12-15 is in train - use a real gap instead.
        ["2013-05-01", "2016-03-01", "2015-02-01", "2015-08-01"],
    )
    target = pd.Series([0, 1, 0, 1])

    split = split_by_time(features, target, train=TRAIN, validation=VALIDATION, test=TEST)

    assert split.rows_in == 4
    assert split.rows_out == 3
    assert split.rows_outside_windows == 1
    assert split.rows_unparseable_date == 0
    assert "dropped_gap=1" in split.summary()


def test_b09_an_unparseable_date_is_quarantined_rather_than_aborting_the_run() -> None:
    features = dated(["2013-05-01", None, "2015-02-01", "2015-08-01"])
    target = pd.Series([0, 1, 0, 1])

    split = split_by_time(features, target, train=TRAIN, validation=VALIDATION, test=TEST)

    assert split.rows_unparseable_date == 1
    assert split.rows_out == 3
    assert "dropped_bad_date=1" in split.summary()


def test_b08_an_empty_partition_reports_the_observed_date_range() -> None:
    features = dated(["2018-01-01", "2018-02-01", "2018-03-01"])
    target = pd.Series([0, 1, 0])

    with pytest.raises(ValueError, match="data covers 2018-01-01 to 2018-03-01"):
        split_by_time(features, target, train=TRAIN, validation=VALIDATION, test=TEST)


def test_b08_an_all_missing_date_column_explains_itself_instead_of_crashing() -> None:
    """`NaT.date()` raises, so the old handler died inside its own error message."""
    features = dated([None, None, None])
    target = pd.Series([0, 1, 0])

    with pytest.raises(ValueError, match="no parseable dates at all"):
        split_by_time(features, target, train=TRAIN, validation=VALIDATION, test=TEST)


def test_the_empty_partition_error_names_which_partitions_were_empty() -> None:
    features = dated(["2013-05-01", "2014-06-01"])
    target = pd.Series([0, 1])

    with pytest.raises(ValueError, match="\\['validation', 'test'\\]"):
        split_by_time(features, target, train=TRAIN, validation=VALIDATION, test=TEST)


def test_overlapping_windows_are_rejected() -> None:
    """Overlap is the one split error that improves every metric instead of raising."""
    features = dated(["2013-05-01", "2015-02-01", "2015-08-01"])
    target = pd.Series([0, 1, 0])

    with pytest.raises(ValueError, match="Overlapping"):
        split_by_time(
            features,
            target,
            train=("2013-01", "2015-03"),
            validation=VALIDATION,
            test=TEST,
        )


def test_b32_a_column_named_split_date_is_left_alone() -> None:
    """The old implementation assigned and then dropped a `_split_date` helper."""
    features = dated(["2013-05-01", "2015-02-01", "2015-08-01"]).assign(_split_date=[7, 8, 9])
    target = pd.Series([0, 1, 0])

    split = split_by_time(features, target, train=TRAIN, validation=VALIDATION, test=TEST)

    assert split.x_train["_split_date"].tolist() == [7]


def test_a_missing_date_column_is_named() -> None:
    with pytest.raises(KeyError, match="issue_d"):
        split_by_time(
            pd.DataFrame({"loan_amnt": [1, 2]}),
            pd.Series([0, 1]),
            train=TRAIN,
            validation=VALIDATION,
            test=TEST,
        )


def test_an_unparsed_date_column_is_refused_rather_than_reparsed_here() -> None:
    features = pd.DataFrame({"issue_d": ["Mar-2015", "Apr-2015"], "loan_amnt": [1, 2]})
    with pytest.raises(TypeError, match="must already be datetime"):
        split_by_time(features, pd.Series([0, 1]), train=TRAIN, validation=VALIDATION, test=TEST)


def test_a_target_with_a_different_index_is_refused() -> None:
    features = dated(["2013-05-01", "2015-02-01", "2015-08-01"])
    target = pd.Series([0, 1, 0], index=[10, 11, 12])

    with pytest.raises(ValueError, match="share an index"):
        split_by_time(features, target, train=TRAIN, validation=VALIDATION, test=TEST)


def test_a_target_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        split_by_time(
            dated(["2013-05-01", "2015-02-01"]),
            pd.Series([0]),
            train=TRAIN,
            validation=VALIDATION,
            test=TEST,
        )


def test_tz_aware_issue_dates_are_split_by_the_same_windows() -> None:
    features = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2013-05-01", "2015-02-01", "2015-08-01"], utc=True),
            "loan_amnt": [1, 2, 3],
        }
    )
    target = pd.Series([0, 1, 0])

    split = split_by_time(features, target, train=TRAIN, validation=VALIDATION, test=TEST)

    assert split.x_train["loan_amnt"].tolist() == [1]
    assert split.x_test["loan_amnt"].tolist() == [3]


# --- build_preprocessor --------------------------------------------------------


@pytest.fixture
def tiny_spec() -> FeatureSpec:
    """Four numerics and one categorical, so the output width is countable."""
    return build_feature_spec(["loan_amnt", "term", "annual_inc", "issue_d", "purpose"])


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_amnt": [10_000.0, 20_000.0, 10_000.0, 20_000.0],
            "term": [" 36 months"] * 4,
            "annual_inc": [50_000.0, 50_000.0, 100_000.0, 100_000.0],
            "issue_d": ["Mar-2015"] * 4,
            "purpose": ["car", "car", "credit_card", "credit_card"],
        }
    )


def test_the_preprocessor_transforms_exactly_the_declared_columns(tiny_spec: FeatureSpec) -> None:
    preprocessor = build_preprocessor(tiny_spec)
    names = {name: columns for name, _, columns in preprocessor.transformers}
    assert names["numeric"] == list(tiny_spec.numeric_features)
    assert names["categorical"] == list(tiny_spec.categorical_features)
    assert preprocessor.remainder == "drop"


def test_b22_the_one_hot_block_is_sparse_and_the_width_is_the_declared_one(
    tiny_spec: FeatureSpec, tiny_frame: pd.DataFrame
) -> None:
    """Dense one-hot output over 1.8M rows was the other half of the 35 GB defect."""
    pipeline = build_model_pipeline(tiny_spec, FunctionTransformer())
    matrix = pipeline.fit_transform(tiny_frame)
    encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]

    assert encoder.sparse_output is True
    # 4 numerics, one indicator each (B33: the block follows the spec, not the
    # training data's gaps), plus one column per purpose.
    assert matrix.shape == (4, 10)
    assert pipeline.named_steps["preprocess"].get_feature_names_out().tolist() == [
        "loan_amnt",
        "term",
        "annual_inc",
        "loan_to_income_ratio",
        "missingindicator_loan_amnt",
        "missingindicator_term",
        "missingindicator_annual_inc",
        "missingindicator_loan_to_income_ratio",
        "purpose_car",
        "purpose_credit_card",
    ]


def test_numeric_features_are_standardized_to_exact_values(
    tiny_spec: FeatureSpec, tiny_frame: pd.DataFrame
) -> None:
    """loan_amnt is 10k/20k/10k/20k, so mean 15k and population sd 5k."""
    pipeline = build_model_pipeline(tiny_spec, FunctionTransformer())
    matrix = _dense(pipeline.fit_transform(tiny_frame))
    assert matrix[:, 0].tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_a_constant_column_becomes_zero_rather_than_nan(
    tiny_spec: FeatureSpec, tiny_frame: pd.DataFrame
) -> None:
    """`term` is 36 for every row: sd 0, which a naive scaler turns into NaN."""
    pipeline = build_model_pipeline(tiny_spec, FunctionTransformer())
    matrix = _dense(pipeline.fit_transform(tiny_frame))
    assert matrix[:, 1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_an_entirely_missing_numeric_column_keeps_its_place_in_the_matrix() -> None:
    """keep_empty_features: without it the width stops matching the spec at serve time."""
    spec = build_feature_spec(["loan_amnt", "term", "annual_inc", "issue_d", "dti"])
    frame = pd.DataFrame(
        {
            "loan_amnt": [10_000.0, 20_000.0],
            "term": [" 36 months"] * 2,
            "annual_inc": [50_000.0, 100_000.0],
            "issue_d": ["Mar-2015"] * 2,
            "dti": [np.nan, np.nan],
        }
    )
    pipeline = build_model_pipeline(spec, FunctionTransformer())
    matrix = _dense(pipeline.fit_transform(frame))
    names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()

    # The column still exists, which is the whole point: the design matrix keeps
    # the width the spec declares, so a serving request cannot be a column short.
    assert "dti_clean" in names
    assert matrix[:, names.index("dti_clean")].tolist() == [0.0, 0.0]


def test_a_partly_missing_numeric_column_gets_a_missingness_indicator() -> None:
    """Missing income data is informative, so the fact of it is a feature."""
    spec = build_feature_spec(["loan_amnt", "term", "annual_inc", "issue_d", "dti"])
    frame = pd.DataFrame(
        {
            "loan_amnt": [10_000.0, 20_000.0, 30_000.0, 40_000.0],
            "term": [" 36 months"] * 4,
            "annual_inc": [50_000.0, 100_000.0, 60_000.0, 90_000.0],
            "issue_d": ["Mar-2015"] * 4,
            "dti": [10.0, 20.0, np.nan, 30.0],
        }
    )
    pipeline = build_model_pipeline(spec, FunctionTransformer())
    matrix = _dense(pipeline.fit_transform(frame))
    names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()

    # Plain 0/1, because the indicator rides in its own branch and never reaches
    # the scaler (B33). Standardized, a column with one missing row in 629 has an
    # sd near 0.016, so a serving row that omits the field lands 60 sd out.
    indicator = matrix[:, names.index("missingindicator_dti_clean")]
    assert indicator.tolist() == [0.0, 0.0, 1.0, 0.0]
    # The imputed value is the median of the three observed rows.
    assert matrix[2, names.index("dti_clean")] == pytest.approx(matrix[1, names.index("dti_clean")])


def test_b33_omitting_an_optional_field_does_not_dominate_the_score() -> None:
    """A rare-in-train gap must not arrive 60 standard deviations out at serve time.

    Reproduces the bug from the serving end, which is the only end it was visible
    from: one missing `dti` in a 200-row fit gives the indicator an sd near 0.07,
    so standardizing it sent a request that simply omitted the field to +14 and the
    reason codes attributed more log-odds to `dti_clean: null` than to every real
    signal combined.
    """
    spec = build_feature_spec(["loan_amnt", "term", "annual_inc", "issue_d", "dti"])
    rows = 200
    frame = pd.DataFrame(
        {
            "loan_amnt": np.linspace(5_000.0, 35_000.0, rows),
            "term": [" 36 months"] * rows,
            "annual_inc": np.linspace(30_000.0, 200_000.0, rows),
            "issue_d": ["Mar-2015"] * rows,
            # Exactly one gap, which is what makes the indicator's sd tiny.
            "dti": [np.nan] + [12.0 + (index % 17) for index in range(rows - 1)],
        }
    )
    pipeline = build_model_pipeline(spec, FunctionTransformer())
    pipeline.fit(frame)
    names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()

    served = _dense(pipeline.transform(frame.head(1).assign(dti=[np.nan])))
    indicator = served[0, names.index("missingindicator_dti_clean")]

    assert indicator == 1.0
    # The real assertion: bounded, so a linear model's contribution for it is its
    # coefficient rather than its coefficient times an arbitrary multiple.
    assert abs(served[0, names.index("missingindicator_dti_clean")]) <= 1.0


def test_b33_every_declared_numeric_column_has_an_indicator() -> None:
    """`features="all"`: the matrix width follows the spec, not train's gaps.

    With sklearn's default `"missing-only"` the indicator block depends on which
    columns happened to have a gap during training, so a column that starts
    arriving with gaps after deployment has no way to say so - and the width of the
    design matrix becomes a property of the training data rather than of the spec.
    """
    spec = build_feature_spec(["loan_amnt", "term", "annual_inc", "issue_d", "dti"])
    frame = pd.DataFrame(
        {
            "loan_amnt": [10_000.0, 20_000.0, 30_000.0],
            "term": [" 36 months"] * 3,
            "annual_inc": [50_000.0, 100_000.0, 60_000.0],
            "issue_d": ["Mar-2015"] * 3,
            "dti": [10.0, 20.0, 30.0],
        }
    )
    pipeline = build_model_pipeline(spec, FunctionTransformer())
    pipeline.fit(frame)
    names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()

    # Nothing is missing anywhere in this frame, so "missing-only" would emit none.
    assert [f"missingindicator_{name}" for name in spec.numeric_features] == [
        name for name in names if name.startswith("missingindicator_")
    ]


def test_an_unseen_category_does_not_break_a_served_row(
    tiny_spec: FeatureSpec, tiny_frame: pd.DataFrame
) -> None:
    pipeline = build_model_pipeline(tiny_spec, FunctionTransformer())
    pipeline.fit(tiny_frame)
    served = pipeline.transform(tiny_frame.head(1).assign(purpose=["renewable_energy"]))

    assert served.shape == (1, 10)
    # Unknown, so neither known level fires.
    assert _dense(served)[0, 8:].tolist() == [0.0, 0.0]


def test_a_spec_with_only_categoricals_still_builds_a_preprocessor() -> None:
    spec = FeatureSpec(
        raw_inputs=("purpose",),
        required_raw_inputs=(),
        engineered=(),
        numeric_features=(),
        categorical_features=("purpose",),
    )
    assert [name for name, _, _ in build_preprocessor(spec).transformers] == ["categorical"]


def test_a_spec_declaring_no_features_cannot_be_constructed() -> None:
    """An empty spec is a configuration mistake, not an empty matrix.

    Asserted at the constructor rather than at `build_preprocessor`, because that
    is where it is actually caught - which is the reason the same guard inside
    `build_preprocessor` is marked unreachable. `ColumnTransformer` with no
    transformers fits happily and produces zero columns, so without either guard
    the failure surfaces as an unintelligible sklearn error inside `fit`, or as a
    model trained on nothing at all.
    """
    with pytest.raises(ValueError, match="declares no model features"):
        FeatureSpec(
            raw_inputs=(),
            required_raw_inputs=(),
            engineered=(),
            numeric_features=(),
            categorical_features=(),
        )


def test_categorical_choices_is_empty_rather_than_raising_without_a_categorical_branch() -> None:
    """Two shapes of pipeline that legitimately have no categories to report.

    `/api/schema` calls this on whatever bundle is loaded, so an empty mapping is
    the answer for a numeric-only spec. Raising would take the route down over a
    model that is working correctly.
    """
    assert categorical_choices(Pipeline(steps=[("model", FunctionTransformer())])) == {}

    numeric_only = FeatureSpec(
        raw_inputs=("loan_amnt",),
        required_raw_inputs=(),
        engineered=(),
        numeric_features=("loan_amnt",),
        categorical_features=(),
    )
    frame = pd.DataFrame({"loan_amnt": [1000.0, 2000.0]})
    fitted = build_preprocessor(numeric_only).fit(frame)
    assert categorical_choices(Pipeline(steps=[("preprocess", fitted)])) == {}


def test_engineering_prefix_rejects_a_pipeline_with_no_engineer_step() -> None:
    """The name-based slice fails loudly rather than returning a wrong prefix.

    Reason codes and the drift tables both read the engineered frame from here, so
    silently handing back an un-engineered one would make the derived features look
    absent from the model rather than absent from the slice.
    """
    with pytest.raises(ValueError, match="expected an `engineer` step"):
        engineering_prefix(Pipeline(steps=[("model", FunctionTransformer())]))


# --- the assembled pipeline ----------------------------------------------------


@pytest.fixture
def raw_split(raw_loans: pd.DataFrame) -> TimeSplit:
    """The synthetic extract, parsed and partitioned the way the pipeline does it."""
    loans, _ = normalize_credit_schema(raw_loans)
    target = pd.Series(
        (loans["loan_status"] == "Charged Off").astype(int), index=loans.index, name="default_flag"
    )
    return split_by_time(
        loans,
        target,
        train=("2013-01", "2014-12"),
        validation=("2015-01", "2015-12"),
        test=("2016-01", "2016-12"),
    )


def test_the_fitted_pipeline_scores_a_raw_frame_end_to_end(raw_loans: pd.DataFrame) -> None:
    spec = build_feature_spec(raw_loans.columns)
    target = pd.Series(
        (raw_loans["loan_status"] == "Charged Off").astype(int), index=raw_loans.index
    )

    model = train_logistic_regression(raw_loans, target, spec=spec)
    probabilities = model.predict_proba(raw_loans)[:, 1]

    assert probabilities.min() > 0.0
    assert probabilities.max() < 1.0
    assert model.named_steps["preprocess"].get_feature_names_out().size > len(spec.model_features)


def test_one_row_scores_identically_alone_and_in_the_batch(raw_loans: pd.DataFrame) -> None:
    """The serving guarantee, all the way through the estimator."""
    spec = build_feature_spec(raw_loans.columns)
    target = pd.Series(
        (raw_loans["loan_status"] == "Charged Off").astype(int), index=raw_loans.index
    )
    model = train_logistic_regression(raw_loans, target, spec=spec)

    batch = model.predict_proba(raw_loans)[:, 1]
    single = model.predict_proba(raw_loans.iloc[[11]])[:, 1]

    assert single[0] == pytest.approx(batch[11], rel=1e-12)


def test_b05_the_logistic_baseline_is_not_class_weighted() -> None:
    """`class_weight='balanced'` inflates every probability, which is what the
    Brier score and the calibration curve are supposed to be measuring."""
    assert DEFAULT_LOGISTIC_PARAMS["class_weight"] is None


def test_a_caller_can_override_a_model_parameter(raw_loans: pd.DataFrame) -> None:
    spec = build_feature_spec(raw_loans.columns)
    target = pd.Series(
        (raw_loans["loan_status"] == "Charged Off").astype(int), index=raw_loans.index
    )
    model = train_logistic_regression(raw_loans, target, spec=spec, config={"C": 0.25})
    assert model.named_steps["classifier"].C == 0.25


def test_an_unsupported_model_type_names_the_supported_ones(
    raw_loans: pd.DataFrame, raw_split: TimeSplit
) -> None:
    spec = build_feature_spec(raw_loans.columns)
    with pytest.raises(ValueError, match="logistic_regression"):
        train_model("random_forest", raw_split, spec=spec)


# --- validation monitoring -----------------------------------------------------


class RecordingClassifier(BaseEstimator, ClassifierMixin):
    """Stands in for ``XGBClassifier`` so the assembly is testable without libomp.

    XGBoost cannot be imported on this machine until ``libomp`` is installed, and
    the part of validation monitoring that can actually break is not XGBoost's
    early stopping - it is *this* project's fit-transform-reassemble dance. So
    that part is verified against a stub that records what it was handed and
    scores deterministically from it.
    """

    def fit(
        self,
        x: Any,
        y: Any,
        eval_set: list[tuple[Any, Any]] | None = None,
        verbose: bool | None = None,
    ) -> RecordingClassifier:
        self.classes_ = np.unique(y)
        self.n_features_in_ = x.shape[1]
        self.train_shape_ = x.shape
        self.eval_shapes_ = [(matrix.shape, len(labels)) for matrix, labels in eval_set or []]
        self.verbose_ = verbose
        return self

    def predict_proba(self, x: Any) -> npt.NDArray[np.float64]:
        # A deterministic function of the row, so "the assembled pipeline predicts
        # what the manual two-step path predicts" is an exact-equality claim.
        totals = np.asarray(_dense(x).sum(axis=1), dtype=np.float64)
        positive = 1.0 / (1.0 + np.exp(-totals / 10.0))
        return np.column_stack([1.0 - positive, positive])


@pytest.fixture
def monitoring_frames() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train and validation frames whose numbers make leakage visible.

    Train ``loan_amnt`` is 10k/20k so the fitted mean is 15k. Validation is 100k
    - wildly out of range on purpose, because if it reached the scaler's ``fit``
    the mean would move and the assertion below would fail.
    """
    train = pd.DataFrame(
        {
            "loan_amnt": [10_000.0, 20_000.0],
            "term": [" 36 months"] * 2,
            "annual_inc": [50_000.0, 100_000.0],
            "issue_d": ["Mar-2013"] * 2,
            "purpose": ["car", "credit_card"],
        }
    )
    validation = pd.DataFrame(
        {
            "loan_amnt": [100_000.0],
            "term": [" 36 months"],
            "annual_inc": [60_000.0],
            "issue_d": ["Mar-2015"],
            "purpose": ["car"],
        }
    )
    return train, pd.Series([0, 1]), validation, pd.Series([1])


def test_the_estimator_receives_a_transformed_validation_matrix(
    tiny_spec: FeatureSpec,
    monitoring_frames: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Raw validation rows through `classifier__eval_set` would be strings."""
    x_train, y_train, x_validation, y_validation = monitoring_frames
    estimator = RecordingClassifier()

    fit_with_validation_monitoring(
        tiny_spec, estimator, x_train, y_train, x_validation, y_validation
    )

    # Same width as train - one design matrix, one column order.
    assert estimator.eval_shapes_ == [((1, estimator.train_shape_[1]), 1)]
    assert estimator.verbose_ is False


def test_the_assembled_pipeline_predicts_what_the_manual_two_step_path_predicts(
    tiny_spec: FeatureSpec,
    monitoring_frames: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Reassembly relies on sklearn's Pipeline not cloning its steps. If that
    ever changes, the returned pipeline holds unfitted transformers and this is
    the test that says so."""
    x_train, y_train, x_validation, y_validation = monitoring_frames
    estimator = RecordingClassifier()

    pipeline = fit_with_validation_monitoring(
        tiny_spec, estimator, x_train, y_train, x_validation, y_validation
    )

    prefix = Pipeline(steps=pipeline.steps[:-1])
    expected = estimator.predict_proba(prefix.transform(x_validation))[:, 1]
    assert pipeline.predict_proba(x_validation)[:, 1].tolist() == expected.tolist()


def test_the_validation_partition_never_reaches_the_preprocessor_fit(
    tiny_spec: FeatureSpec,
    monitoring_frames: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Value-exact: train is 10k and 20k, so the scaler's mean is 15k. Had the
    100k validation row been fitted on, it would be 43,333."""
    x_train, y_train, x_validation, y_validation = monitoring_frames

    pipeline = fit_with_validation_monitoring(
        tiny_spec, RecordingClassifier(), x_train, y_train, x_validation, y_validation
    )
    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]

    assert scaler.mean_[0] == 15_000.0


@requires_xgboost
def test_xgboost_stops_early_on_the_validation_partition(raw_split: TimeSplit) -> None:
    """The real thing, when the local wheel can load. Unverified until then."""
    spec = build_feature_spec(raw_split.x_train.columns)

    model = train_xgboost_model(
        raw_split.x_train,
        raw_split.y_train,
        spec=spec,
        x_validation=raw_split.x_validation,
        y_validation=raw_split.y_validation,
        # Deliberately far more rounds than a 900-row fixture needs, so stopping
        # early is the only way best_iteration lands below the budget.
        config={"n_estimators": 200},
        early_stopping_rounds=5,
    )
    classifier = model.named_steps["classifier"]

    assert classifier.best_iteration < 199
    assert model.predict_proba(raw_split.x_test)[:, 1].min() > 0.0


@requires_xgboost
def test_xgboost_without_a_validation_partition_uses_its_whole_budget(
    raw_split: TimeSplit,
) -> None:
    spec = build_feature_spec(raw_split.x_train.columns)

    model = train_xgboost_model(
        raw_split.x_train, raw_split.y_train, spec=spec, config={"n_estimators": 12}
    )

    # No eval_set, so asking XGBoost to stop early would raise.
    assert model.named_steps["classifier"].get_params()["early_stopping_rounds"] is None
