"""Tests for the declared feature contract and the in-Pipeline transformers.

The load-bearing claim of this module is negative: no string column can reach
``OneHotEncoder`` by accident, because nothing infers a dtype. The tests that
prove it are ``test_b01_*`` and ``test_b23_*``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from risk_score.features import (
    ENGINEERED_BY_NAME,
    ParseKind,
    column_spec,
    resolvable_engineered_features,
)
from risk_score.sample_data import make_synthetic_loans
from risk_score.transformers import (
    MISSING_CATEGORY,
    SPEC_VERSION,
    CanonicalizeFrame,
    EngineerFeatures,
    FeatureSpec,
    build_feature_spec,
)


@pytest.fixture
def raw_loans() -> pd.DataFrame:
    """A raw-format extract: source names, percent strings, ' 36 months'."""
    return make_synthetic_loans(n_rows=400, seed=7)


@pytest.fixture
def spec(raw_loans: pd.DataFrame) -> FeatureSpec:
    return build_feature_spec(raw_loans.columns)


# --- FeatureSpec ---------------------------------------------------------------


def test_the_spec_reflects_the_columns_the_extract_actually_has() -> None:
    spec = build_feature_spec(["loan_amnt", "term", "annual_inc", "issue_d", "purpose"])
    assert spec.numeric_features == ("loan_amnt", "term", "annual_inc", "loan_to_income_ratio")
    assert spec.categorical_features == ("purpose",)
    assert spec.engineered == ("loan_to_income_ratio",)


def test_absent_fico_columns_produce_no_fico_features() -> None:
    """Both real extracts lack fico_range_low/high; feature_config.yaml claimed otherwise."""
    spec = build_feature_spec(["loan_amnt", "term", "annual_inc", "issue_d"])
    assert "fico_midpoint" not in spec.model_features
    assert "fico_band" not in spec.model_features


def test_fico_features_appear_when_the_columns_do() -> None:
    spec = build_feature_spec(
        ["loan_amnt", "term", "annual_inc", "issue_d", "fico_range_low", "fico_range_high"]
    )
    assert "fico_midpoint" in spec.numeric_features
    assert "fico_band" in spec.categorical_features
    # Consumed, so the raw bounds do not also reach the model.
    assert "fico_range_low" not in spec.model_features
    assert "fico_range_high" not in spec.model_features
    # But they are still read, because the midpoint needs them.
    assert "fico_range_low" in spec.raw_inputs


def test_utilization_is_built_from_the_bureau_fallback_when_revol_util_is_absent() -> None:
    """`requires_any`: the loans_full_schema extract has no revol_util column."""
    spec = build_feature_spec(
        [
            "loan_amnt",
            "term",
            "annual_inc",
            "issue_d",
            "total_credit_utilized",
            "total_credit_limit",
        ]
    )
    assert "credit_utilization" in spec.numeric_features
    # The levels are not consumed; only revol_util is, because it *is* the ratio.
    assert "total_credit_utilized" in spec.numeric_features
    assert "total_credit_limit" in spec.numeric_features


def test_revol_util_wins_over_the_fallback_and_is_consumed() -> None:
    spec = build_feature_spec(["loan_amnt", "term", "annual_inc", "issue_d", "revol_util"])
    assert "credit_utilization" in spec.numeric_features
    assert "revol_util" not in spec.model_features
    assert "revol_util" in spec.raw_inputs


def test_lender_priced_columns_are_absent_by_default_and_present_on_request(
    raw_loans: pd.DataFrame,
) -> None:
    default = build_feature_spec(raw_loans.columns)
    opted_in = build_feature_spec(raw_loans.columns, include_lender_priced=True)
    for name in ("int_rate", "grade", "sub_grade", "installment"):
        assert name not in default.model_features
        assert name in opted_in.model_features
    assert default.include_lender_priced is False
    assert opted_in.include_lender_priced is True


def test_b01_no_date_column_is_ever_a_model_feature(spec: FeatureSpec) -> None:
    """earliest_cr_line as a raw string was one-hot encoded into 655 columns."""
    for name in spec.model_features:
        if name in ENGINEERED_BY_NAME:
            continue
        assert column_spec(name).parse is not ParseKind.MONTH_DATE
    assert "earliest_cr_line" in spec.raw_inputs
    assert "credit_history_months" in spec.numeric_features


def test_b23_every_declared_categorical_is_low_cardinality(spec: FeatureSpec) -> None:
    """The dtype-routing bug is only dangerous because the encoder is unbounded."""
    for name in spec.categorical_features:
        if name in ENGINEERED_BY_NAME:
            continue
        assert column_spec(name).parse is ParseKind.CATEGORY


def test_model_features_are_numeric_then_categorical(spec: FeatureSpec) -> None:
    assert spec.model_features == (*spec.numeric_features, *spec.categorical_features)


def test_a_column_cannot_be_declared_numeric_and_categorical() -> None:
    with pytest.raises(ValueError, match="both numeric and categorical"):
        FeatureSpec(
            raw_inputs=("dti",),
            required_raw_inputs=(),
            engineered=(),
            numeric_features=("dti",),
            categorical_features=("dti",),
        )


def test_a_spec_with_no_features_is_rejected() -> None:
    with pytest.raises(ValueError, match="no model features"):
        build_feature_spec(["id", "url", "loan_status"])


def test_a_required_input_outside_raw_inputs_is_rejected() -> None:
    with pytest.raises(ValueError, match="required_raw_inputs not in raw_inputs"):
        FeatureSpec(
            raw_inputs=("dti",),
            required_raw_inputs=("annual_inc",),
            engineered=(),
            numeric_features=("dti",),
            categorical_features=(),
        )


def test_an_unknown_engineered_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown engineered features"):
        FeatureSpec(
            raw_inputs=("dti",),
            required_raw_inputs=(),
            engineered=("magic_score",),
            numeric_features=("dti",),
            categorical_features=(),
        )


def test_parse_kinds_covers_every_raw_input(spec: FeatureSpec) -> None:
    assert set(spec.parse_kinds()) == set(spec.raw_inputs)


def test_the_spec_pins_the_date_formats_and_its_version(spec: FeatureSpec) -> None:
    assert spec.spec_version == SPEC_VERSION
    assert "%b-%Y" in spec.date_formats


def test_summary_states_the_tier_setting(raw_loans: pd.DataFrame) -> None:
    assert "lender_priced=off" in build_feature_spec(raw_loans.columns).summary()
    assert (
        "lender_priced=on"
        in build_feature_spec(raw_loans.columns, include_lender_priced=True).summary()
    )


def test_the_spec_and_build_feature_matrix_resolve_the_same_features(
    raw_loans: pd.DataFrame, spec: FeatureSpec
) -> None:
    """One resolver, so a served row cannot get a different feature set."""
    canonical = CanonicalizeFrame(spec).fit_transform(raw_loans)
    buildable, _blocked = resolvable_engineered_features(canonical.columns)
    assert tuple(feature.name for feature in buildable) == spec.engineered


# --- CanonicalizeFrame ---------------------------------------------------------


def test_canonicalize_resolves_aliases_and_declared_dtypes(
    raw_loans: pd.DataFrame, spec: FeatureSpec
) -> None:
    result = CanonicalizeFrame(spec).fit_transform(raw_loans)

    assert list(result.columns) == list(spec.raw_inputs)
    assert result["term"].dtype == "float64"
    assert set(result["term"].dropna().unique()) <= {36.0, 60.0}
    assert result["issue_d"].dtype.kind == "M"
    assert result["earliest_cr_line"].dtype.kind == "M"
    # 13.56% -> 0.1356, not 13.56 and not a string.
    assert result["revol_util"].dtype == "float64"
    assert result["revol_util"].max() <= 5.0


def test_b01_canonicalize_drops_columns_outside_the_declared_inputs(
    raw_loans: pd.DataFrame, spec: FeatureSpec
) -> None:
    """Post-origination columns and identifiers cannot survive the reindex."""
    result = CanonicalizeFrame(spec).fit_transform(raw_loans)
    for name in ("id", "url", "recoveries", "total_pymnt", "last_fico_range_high", "loan_status"):
        assert name not in result.columns


def test_canonicalize_creates_an_absent_optional_input_as_all_missing() -> None:
    spec = build_feature_spec(["loan_amnt", "term", "annual_inc", "issue_d", "dti"])
    frame = pd.DataFrame(
        {
            "loan_amnt": [10_000.0],
            "term": [" 36 months"],
            "annual_inc": [60_000.0],
            "issue_d": ["Mar-2015"],
        }
    )
    result = CanonicalizeFrame(spec).fit_transform(frame)
    assert "dti" in result.columns
    assert result["dti"].isna().all()
    assert result["dti"].dtype == "float64"


def test_canonicalize_names_the_required_column_it_is_missing(spec: FeatureSpec) -> None:
    frame = pd.DataFrame({"loan_amnt": [10_000.0], "term": [" 36 months"]})
    with pytest.raises(KeyError, match="annual_inc"):
        CanonicalizeFrame(spec).fit_transform(frame)


def test_b02_canonicalize_resolves_a_duplicate_source_by_alias_priority(
    spec: FeatureSpec,
) -> None:
    """funded_amnt is an alias of loan_amnt; the standard extract has both."""
    frame = pd.DataFrame(
        {
            "funded_amnt": [1.0],
            "loan_amnt": [2.0],
            "term": [" 36 months"],
            "annual_inc": [60_000.0],
            "issue_d": ["Mar-2015"],
        }
    )
    result = CanonicalizeFrame(spec).fit_transform(frame)
    assert list(result.columns).count("loan_amnt") == 1
    assert result["loan_amnt"].iloc[0] == 2.0


def test_canonicalize_accepts_a_single_row_request(spec: FeatureSpec) -> None:
    """The serving path. Nothing here may depend on having a distribution."""
    request = pd.DataFrame(
        {
            "loan_amnt": [12_000.0],
            "term": [" 36 months"],
            "annual_inc": [48_000.0],
            "issue_d": ["Mar-2015"],
            "earliest_cr_line": ["Aug-2003"],
            "dti": [18.4],
            "revol_util": ["42.1%"],
            "purpose": ["debt_consolidation"],
            "emp_length": ["10+ years"],
        }
    )
    result = CanonicalizeFrame(spec).fit_transform(request)
    assert len(result) == 1
    assert result["revol_util"].iloc[0] == pytest.approx(0.421)
    assert result["emp_length"].iloc[0] == 10.0
    assert result["term"].iloc[0] == 36.0


def test_canonicalize_refuses_a_bare_array(spec: FeatureSpec) -> None:
    with pytest.raises(TypeError, match="named columns"):
        CanonicalizeFrame(spec).fit_transform(np.zeros((3, 4)))


# --- EngineerFeatures ----------------------------------------------------------


def test_engineer_emits_exactly_the_declared_model_features(
    raw_loans: pd.DataFrame, spec: FeatureSpec
) -> None:
    canonical = CanonicalizeFrame(spec).fit_transform(raw_loans)
    result = EngineerFeatures(spec).fit_transform(canonical)
    assert list(result.columns) == list(spec.model_features)


def test_b01_engineer_drops_the_raw_sources_its_outputs_replace(
    raw_loans: pd.DataFrame, spec: FeatureSpec
) -> None:
    canonical = CanonicalizeFrame(spec).fit_transform(raw_loans)
    result = EngineerFeatures(spec).fit_transform(canonical)
    for name in ("dti", "revol_util", "earliest_cr_line", "issue_d"):
        assert name not in result.columns
    for name in ("dti_clean", "credit_utilization", "credit_history_months"):
        assert name in result.columns


def test_b01_no_object_dtype_survives_into_the_numeric_features(
    raw_loans: pd.DataFrame, spec: FeatureSpec
) -> None:
    """The dtype-routing bug, stated as an assertion on the output."""
    canonical = CanonicalizeFrame(spec).fit_transform(raw_loans)
    result = EngineerFeatures(spec).fit_transform(canonical)
    for name in spec.numeric_features:
        assert result[name].dtype == "float64", name


def test_missing_categoricals_become_a_level_not_a_gap(spec: FeatureSpec) -> None:
    frame = pd.DataFrame(
        {
            "loan_amnt": [10_000.0, 10_000.0],
            "term": [" 36 months", " 36 months"],
            "annual_inc": [60_000.0, 60_000.0],
            "issue_d": ["Mar-2015", "Mar-2015"],
            "earliest_cr_line": ["Aug-2003", "Aug-2003"],
            "dti": [10.0, 10.0],
            "revol_util": ["10%", "10%"],
            "purpose": ["car", None],
        }
    )
    canonical = CanonicalizeFrame(spec).fit_transform(frame)
    result = EngineerFeatures(spec).fit_transform(canonical)
    assert result["purpose"].tolist() == ["car", MISSING_CATEGORY]
    assert result[list(spec.categorical_features)].notna().all().all()


def test_engineer_names_a_feature_it_cannot_build(spec: FeatureSpec) -> None:
    """require_all=True: at training time a missing input is a misconfiguration."""
    canonical = CanonicalizeFrame(spec).fit_transform(make_synthetic_loans(n_rows=20, seed=1))
    with pytest.raises(KeyError, match="credit_history_months"):
        EngineerFeatures(spec).fit_transform(canonical.drop(columns=["earliest_cr_line"]))


def test_engineer_ignores_a_column_the_declared_spec_does_not_mention() -> None:
    """A row carrying an extra column cannot grow the feature set the model saw."""
    raw = make_synthetic_loans(n_rows=50, seed=3, include_fico=False)
    spec = build_feature_spec(raw.columns)
    assert "fico_midpoint" not in spec.model_features

    canonical = CanonicalizeFrame(spec).fit_transform(raw)
    padded = canonical.assign(fico_range_low=700.0, fico_range_high=704.0)
    result = EngineerFeatures(spec).fit_transform(padded)

    assert list(result.columns) == list(spec.model_features)
    assert "fico_midpoint" not in result.columns


# --- composition ---------------------------------------------------------------


def test_the_two_steps_compose_in_a_pipeline(raw_loans: pd.DataFrame, spec: FeatureSpec) -> None:
    pipeline = Pipeline(
        steps=[("canonicalize", CanonicalizeFrame(spec)), ("engineer", EngineerFeatures(spec))]
    )
    result = pipeline.fit_transform(raw_loans)
    assert list(result.columns) == list(spec.model_features)
    assert pipeline.named_steps["engineer"].get_feature_names_out() == list(spec.model_features)


def test_b22_the_declared_routing_collapses_the_one_hot_width(raw_loans: pd.DataFrame) -> None:
    """The 35 GB defect, measured rather than asserted.

    ``select_dtypes`` sent every unparsed string to ``OneHotEncoder``: on this
    frame that is 17 columns and thousands of levels, and with
    ``sparse_output=False`` over 1.8M real rows it is roughly 35 GB. Declared
    routing caps it, and the cap is what this asserts.
    """
    spec = build_feature_spec(raw_loans.columns)
    steps = Pipeline(
        steps=[("canonicalize", CanonicalizeFrame(spec)), ("engineer", EngineerFeatures(spec))]
    )
    result = steps.fit_transform(raw_loans)

    levels = {name: int(result[name].nunique()) for name in spec.categorical_features}
    assert max(levels.values()) <= 50, levels
    assert sum(levels.values()) <= 100, levels

    inferred = [
        name for name in raw_loans.columns if not pd.api.types.is_numeric_dtype(raw_loans[name])
    ]
    would_have_been = sum(int(raw_loans[name].nunique()) for name in inferred)
    assert would_have_been > 20 * sum(levels.values())


def test_transformers_are_clonable_and_learn_nothing(spec: FeatureSpec) -> None:
    """sklearn.clone drops fitted state; if these carried any, output would change."""
    original = CanonicalizeFrame(spec)
    copy = clone(original)
    assert copy.spec == spec
    assert copy.get_params()["spec"] == spec


def test_one_row_scored_through_the_frame_steps_matches_the_batch(
    raw_loans: pd.DataFrame, spec: FeatureSpec
) -> None:
    """Train and serve share the code path, so row 5 alone must equal row 5 of the batch."""
    steps = Pipeline(
        steps=[("canonicalize", CanonicalizeFrame(spec)), ("engineer", EngineerFeatures(spec))]
    )
    batch = steps.fit_transform(raw_loans)
    single = steps.transform(raw_loans.iloc[[5]])
    pd.testing.assert_frame_equal(single, batch.iloc[[5]])
