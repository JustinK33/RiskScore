"""Tests for per-applicant reason codes and global feature importance.

The suite is built around one property, because it is the property that makes an
explanation worth showing to anybody:

    baseline + sum(contributions) == the model's own margin

If that holds to floating-point tolerance, the contributions are the exact SHAP
values and not an approximation of them - which is what lets this project compute
them without the ``shap`` package. Every other test here is about the parts that
turn those numbers into something readable: one-hot families collapsing to their
source feature, values being the applicant's own, and the ordering being by
magnitude rather than by signed value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from risk_score.explain import (
    BACKGROUND_ROWS,
    Explainer,
    build_background,
    feature_label,
    feature_sources,
    transformed_feature_names,
)
from risk_score.pipeline import RunResult, train_run
from risk_score.sample_data import make_synthetic_loans
from tests.conftest import SMALL_ROWS, requires_xgboost


@pytest.fixture
def explainer(trained_run: RunResult) -> Explainer:
    return Explainer(trained_run.bundle)


def _raw_rows(trained_run: RunResult, count: int = 4) -> pd.DataFrame:
    """Raw applicant rows in the shape ``/predict`` will receive.

    Read back out of the extract the run was fitted from rather than
    reconstructed, so the frame carries the same percent strings and missingness
    a real request would.
    """
    frame = pd.read_csv(trained_run.metadata.dataset_path)
    return frame.head(count)


# --- the additivity property ---------------------------------------------------


def test_the_contributions_reconstruct_the_models_own_margin(
    explainer: Explainer, trained_run: RunResult
) -> None:
    """The claim that makes a reason code auditable rather than decorative.

    Asserted against `decision_function` - the model's own log-odds - so the
    check is independent of how the contributions were computed.
    """
    applicants = _raw_rows(trained_run)

    explanations = explainer.explain(applicants)
    expected = trained_run.bundle.pipeline.decision_function(applicants)

    rebuilt = np.array([item.total_log_odds for item in explanations])
    np.testing.assert_allclose(rebuilt, expected, rtol=0, atol=1e-9)


def test_every_declared_feature_appears_exactly_once_per_explanation(
    explainer: Explainer, trained_run: RunResult
) -> None:
    """Complete and non-duplicating, which is what makes the sum above meaningful:
    a missing feature would silently move its contribution into the baseline."""
    explanation = explainer.explain(_raw_rows(trained_run, 1))[0]

    named = [item.feature for item in explanation.contributions]
    assert sorted(named) == sorted(trained_run.bundle.feature_spec.model_features)
    assert len(named) == len(set(named))


def test_the_baseline_is_the_score_of_an_average_applicant(explainer: Explainer) -> None:
    """The baseline has to be the population's log-odds, not zero and not the
    intercept: an applicant whose every feature sits at the background mean must
    have no contributions at all."""
    mean_row = np.asarray(explainer._mean, dtype=np.float64).reshape(1, -1)
    columns, baseline = explainer._column_contributions(mean_row)

    np.testing.assert_allclose(columns, 0.0, atol=1e-12)
    estimator = explainer.pipeline.steps[-1][1]
    expected = float(estimator.intercept_[0] + np.ravel(estimator.coef_) @ explainer._mean)
    assert baseline == pytest.approx(expected)


# --- readability ---------------------------------------------------------------


def test_one_hot_columns_collapse_into_their_source_feature(
    explainer: Explainer, trained_run: RunResult
) -> None:
    """`addr_state` is one reason code, not fifty.

    The design matrix has one column per state; an applicant has one state. This
    also pins the mapping direction: the reason code names the field the request
    payload carries.
    """
    spec = trained_run.bundle.feature_spec
    assert "addr_state" in spec.categorical_features

    matrix_columns = transformed_feature_names(explainer.pipeline)
    state_columns = [name for name in matrix_columns if name.startswith("addr_state_")]
    assert len(state_columns) > 1, "the fixture must one-hot at least two states"

    sources = feature_sources(explainer.pipeline)
    assert {sources[matrix_columns.index(name)] for name in state_columns} == {"addr_state"}
    assert explainer.features.count("addr_state") == 1


def test_a_missing_value_indicator_is_attributed_to_the_feature_it_is_about(
    explainer: Explainer,
) -> None:
    """`SimpleImputer(add_indicator=True)` emits `missingindicator_<column>`.

    Left unmapped it would read as a separate feature named after an
    implementation detail, and the applicant's own value for it would be absent.
    """
    matrix_columns = transformed_feature_names(explainer.pipeline)
    indicators = [name for name in matrix_columns if name.startswith("missingindicator_")]
    assert indicators, "the fixture must have at least one column with missing values"

    sources = feature_sources(explainer.pipeline)
    for name in indicators:
        assert sources[matrix_columns.index(name)] == name.removeprefix("missingindicator_")


def test_a_reason_code_cites_the_applicants_own_value_not_a_scaled_one(
    explainer: Explainer, trained_run: RunResult
) -> None:
    """-0.42 is not a reason. The value shown is post-engineering and pre-scaling,
    so it is the number the applicant would recognize."""
    applicants = _raw_rows(trained_run, 1)
    engineered = Pipeline(steps=explainer.pipeline.steps[:2]).transform(applicants)

    explanation = explainer.explain(applicants)[0]

    cited = {item.feature: item.value for item in explanation.contributions}
    assert cited["loan_amnt"] == pytest.approx(float(engineered.iloc[0]["loan_amnt"]))
    assert cited["addr_state"] == engineered.iloc[0]["addr_state"]


def test_contributions_are_ordered_by_magnitude_in_either_direction(
    explainer: Explainer, trained_run: RunResult
) -> None:
    """The largest *effect* first, not the largest positive one: a feature that
    strongly reduces risk is as much of an explanation as one that raises it."""
    explanation = explainer.explain(_raw_rows(trained_run, 1))[0]

    magnitudes = [abs(item.log_odds) for item in explanation.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert list(explanation.top(3)) == list(explanation.contributions[:3])


def test_adverse_reasons_are_only_the_ones_that_pushed_towards_default(
    explainer: Explainer, trained_run: RunResult
) -> None:
    """What a declined applicant is owed. A rejection explained by the two things
    that helped them is not an explanation."""
    explanation = explainer.explain(_raw_rows(trained_run, 1))[0]

    reasons = explanation.adverse_reasons(k=3)
    assert reasons, "the fixture row must have at least one risk-increasing feature"
    assert all(item.log_odds > 0 for item in reasons)
    assert all(item.direction == "increases risk" for item in reasons)


def test_a_feature_label_comes_from_the_registry_that_declares_it() -> None:
    """One source of truth for the human wording, so a new feature cannot be
    added with a description and still render as its own column name."""
    assert feature_label("annual_inc") == "Self-reported annual income."
    assert feature_label("loan_to_income_ratio").startswith("Requested principal divided")
    # An unknown name is its own label rather than an error: an unattributable
    # design-matrix column must still be showable.
    assert feature_label("not_a_feature") == "not_a_feature"


def test_the_sentence_form_names_the_feature_the_value_and_the_direction(
    explainer: Explainer, trained_run: RunResult
) -> None:
    top = explainer.explain(_raw_rows(trained_run, 1))[0].top(1)[0]

    sentence = top.sentence()
    assert top.feature in sentence
    assert ("increases risk" in sentence) or ("reduces risk" in sentence)
    assert "log-odds" in sentence


# --- the background ------------------------------------------------------------


def test_the_background_is_capped_and_deterministic(trained_run: RunResult) -> None:
    """Two builds of the same background must be byte-identical, because it is
    pickled into the bundle and a bundle that differs run to run cannot be
    compared to itself."""
    pipeline = trained_run.bundle.pipeline
    applicants = pd.read_csv(trained_run.metadata.dataset_path)

    first = build_background(pipeline, applicants, max_rows=25)
    second = build_background(pipeline, applicants, max_rows=25)

    assert first.shape == (25, len(transformed_feature_names(pipeline)))
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_a_background_smaller_than_the_cap_is_kept_whole(trained_run: RunResult) -> None:
    applicants = pd.read_csv(trained_run.metadata.dataset_path).head(7)

    background = build_background(trained_run.bundle.pipeline, applicants)

    assert len(background) == 7


def test_the_published_bundle_carries_a_background_within_the_cap(
    trained_run: RunResult,
) -> None:
    """The property the service depends on: an explainer is built from the bundle
    alone, so the training extract need not exist at serving time."""
    background = trained_run.bundle.shap_background

    assert background is not None
    assert 0 < len(background) <= BACKGROUND_ROWS
    assert background.shape[1] == len(transformed_feature_names(trained_run.bundle.pipeline))


# --- global importance ---------------------------------------------------------


def test_the_global_summary_ranks_by_mean_absolute_contribution(
    explainer: Explainer, trained_run: RunResult
) -> None:
    """Mean *absolute*, because a feature that pushes half the population up and
    half down is important and its signed mean is zero."""
    summary = explainer.global_summary(_raw_rows(trained_run, 40))

    assert list(summary.columns) == [
        "feature",
        "label",
        "mean_abs_log_odds",
        "mean_log_odds",
        "columns",
        "rank",
    ]
    assert list(summary["rank"]) == list(range(1, len(summary) + 1))
    assert summary["mean_abs_log_odds"].is_monotonic_decreasing
    assert (summary["mean_abs_log_odds"] >= summary["mean_log_odds"].abs()).all()
    assert sorted(summary["feature"]) == sorted(trained_run.bundle.feature_spec.model_features)


def test_the_summary_records_how_many_matrix_columns_each_feature_owns(
    explainer: Explainer, trained_run: RunResult
) -> None:
    """So a reader can see that `addr_state`'s importance is spread over fifty
    columns while `annual_inc`'s is one, which is the honest caveat on comparing
    a categorical's aggregate to a numeric's."""
    summary = explainer.global_summary(_raw_rows(trained_run, 40)).set_index("feature")

    assert summary.loc["addr_state", "columns"] > 1
    assert summary.loc["annual_inc", "columns"] >= 1
    assert summary["columns"].sum() == len(transformed_feature_names(explainer.pipeline))


def test_the_run_publishes_the_summary_and_names_its_top_features(
    trained_run: RunResult,
) -> None:
    """The artifact contract for explainability: a CSV of magnitudes plus the
    names in `metrics.json`, so the dashboard header needs one read."""
    published = pd.read_csv(trained_run.run_dir / "shap_summary.csv")

    assert list(published["feature"][:5]) == trained_run.payload["top_features"]
    assert trained_run.payload["explainer"] == "linear"


# --- the tree path -------------------------------------------------------------


@requires_xgboost
def test_tree_contributions_reconstruct_the_boosters_own_margin(tmp_path: Path) -> None:
    """The same additivity property, through XGBoost's native TreeSHAP.

    Checked against `output_margin=True` rather than against the ``shap``
    package, because the claim being made is that the contributions sum to *this
    model's* prediction - and a library agreeing with itself would not show that.
    """
    dataset = tmp_path / "loans.csv"
    make_synthetic_loans(n_rows=SMALL_ROWS).to_csv(dataset, index=False)
    run = train_run(dataset, output_dir=tmp_path / "reports", model_type="xgboost")
    applicants = pd.read_csv(dataset).head(4)

    explainer = Explainer(run.bundle)
    explanations = explainer.explain(applicants)

    assert explainer.model_kind == "tree"
    engineered = Pipeline(steps=run.bundle.pipeline.steps[:2]).transform(applicants)
    matrix = run.bundle.pipeline.named_steps["preprocess"].transform(engineered)
    expected = run.bundle.pipeline.steps[-1][1].predict(matrix, output_margin=True)
    rebuilt = np.array([item.total_log_odds for item in explanations])
    np.testing.assert_allclose(rebuilt, expected, rtol=1e-6, atol=1e-6)
    assert run.payload["explainer"] == "tree"


# --- refusals ------------------------------------------------------------------


def test_a_model_neither_path_can_explain_is_refused_at_construction(
    trained_run: RunResult,
) -> None:
    """At construction, not at the first request. An explainer that builds and
    then fails per-applicant turns a deployment mistake into an outage."""
    from dataclasses import replace

    from sklearn.dummy import DummyClassifier

    broken = Pipeline(
        steps=[*trained_run.bundle.pipeline.steps[:-1], ("classifier", DummyClassifier())]
    )
    with pytest.raises(TypeError, match="get_booster"):
        Explainer(replace(trained_run.bundle, pipeline=broken))


def test_a_linear_bundle_without_a_background_is_refused(trained_run: RunResult) -> None:
    """A linear model's contributions are measured against the population's means,
    so a bundle written before backgrounds existed cannot produce them - and must
    say so rather than measuring against zero."""
    from dataclasses import replace

    with pytest.raises(ValueError, match="shap_background"):
        Explainer(replace(trained_run.bundle, shap_background=None))


def test_explaining_something_that_is_not_a_frame_is_refused(explainer: Explainer) -> None:
    """These steps route columns by name; an array has none."""
    with pytest.raises(TypeError, match="DataFrame"):
        explainer.explain(np.zeros((2, 3)))
