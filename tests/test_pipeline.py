"""End-to-end tests for the orchestration in :mod:`risk_score.pipeline`.

Every assertion here is against an independently computed value rather than a
range. ``assert 0 <= metrics.auc_roc <= 1`` is exactly why an artifact with
AUC 0.070 next to KS 0.930 - inverted labels - sat in the repository looking
fine, so this suite reloads the persisted model and recomputes the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from risk_score.config import RunConfig, SplitConfig
from risk_score.data_loading import create_default_target, load_lending_club_data
from risk_score.evaluation import (
    ClassificationMetrics,
    ValidationScores,
    select_threshold_by_cost,
)
from risk_score.modeling import TimeSplit, split_by_time
from risk_score.pipeline import run_baseline_pipeline

# The synthetic extract covers 2013-01..2016-12, so three years of vintages
# split three ways with positives in each. These are the shipped defaults too;
# restated here so the split these tests reproduce by hand is explicit.
TRAIN = ("2013-01", "2014-12")
VALIDATION = ("2015-01", "2015-12")
TEST = ("2016-01", "2016-12")

SPLIT = SplitConfig(train=TRAIN, validation=VALIDATION, test=TEST)

#: The pipeline's own default, restated so the threshold assertion below compares
#: against a stated cost matrix rather than whatever the default happens to be.
COSTS = RunConfig().cost_matrix


def run(
    raw_path: Path,
    output_dir: Path,
    model_type: str = "logistic_regression",
    **config_kwargs: Any,
) -> ClassificationMetrics:
    """The pipeline on one fixed set of windows; only the extract varies."""
    return run_baseline_pipeline(
        raw_path,
        output_dir=output_dir,
        model_type=model_type,
        config=RunConfig(split=SPLIT, **config_kwargs),
    )


def rebuild_split(raw_path: Path) -> TimeSplit:
    """Redo the load and split, so a test can check the pipeline's numbers.

    Deliberately a second spelling of the *ordering* rather than a call into the
    pipeline: a test that asks the code under test to compute its own expected
    value can only prove that code is self-consistent.
    """
    loans = load_lending_club_data(raw_path)
    loans = loans.assign(default_flag=create_default_target(loans))
    loans = loans.dropna(subset=["default_flag"])
    return split_by_time(
        loans.drop(columns=["default_flag"]),
        loans["default_flag"].astype(int),
        train=TRAIN,
        validation=VALIDATION,
        test=TEST,
    )


def load_model(output_dir: Path, model_type: str = "logistic_regression") -> Pipeline:
    """The artifact the run wrote, loaded the way a serving process would."""
    model: Pipeline = joblib.load(output_dir / "models" / f"{model_type}.joblib")
    return model


def score(model: Pipeline, features: pd.DataFrame) -> npt.NDArray[np.float64]:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)


def read_metrics(output_dir: Path, model_type: str = "logistic_regression") -> dict[str, Any]:
    path = output_dir / "metrics" / f"{model_type}_metrics.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def test_the_pipeline_writes_every_documented_artifact(raw_csv: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    run(raw_csv, output_dir)

    for relative in (
        "metrics/logistic_regression_metrics.json",
        "metrics/logistic_regression_calibration.csv",
        "metrics/logistic_regression_threshold_costs.csv",
        "figures/logistic_regression_calibration.png",
        "models/logistic_regression.joblib",
    ):
        assert (output_dir / relative).exists(), relative


def test_the_reported_metrics_are_the_test_partitions_own_numbers(
    raw_csv: Path, tmp_path: Path
) -> None:
    """Recomputed from the persisted model rather than read back from the run."""
    output_dir = tmp_path / "reports"
    metrics = run(raw_csv, output_dir)

    split = rebuild_split(raw_csv)
    scores_test = score(load_model(output_dir), split.x_test)

    assert metrics.auc_roc == pytest.approx(roc_auc_score(split.y_test, scores_test), rel=1e-12)
    assert metrics.default_rate == pytest.approx(float(split.y_test.mean()), rel=1e-12)
    # An AUC below 0.5 alongside a high KS is the inverted-label signature the
    # committed artifact had; asserting the direction is what catches it.
    assert metrics.auc_roc > 0.5


def test_b04_the_threshold_is_selected_on_validation_not_on_test(
    raw_csv: Path, tmp_path: Path
) -> None:
    """Tuning the decision rule on test turns the reported cost into a self-report."""
    output_dir = tmp_path / "reports"
    run(raw_csv, output_dir)
    payload = read_metrics(output_dir)

    split = rebuild_split(raw_csv)
    model = load_model(output_dir)
    scores_validation = pd.Series(score(model, split.x_validation), index=split.x_validation.index)
    expected = select_threshold_by_cost(
        ValidationScores(y_true=split.y_validation, y_score=scores_validation),
        cost_matrix=COSTS,
    )

    assert payload["selected_threshold"] == expected
    assert payload["threshold_selected_on"] == "validation"


def test_the_approval_rate_is_measured_on_the_population_being_scored(
    raw_csv: Path, tmp_path: Path
) -> None:
    """Not read out of the validation cost table, where the rule was tuned."""
    output_dir = tmp_path / "reports"
    metrics = run(raw_csv, output_dir)

    split = rebuild_split(raw_csv)
    scores_test = score(load_model(output_dir), split.x_test)
    threshold = read_metrics(output_dir)["selected_threshold"]

    assert metrics.approval_rate == pytest.approx(
        float((scores_test < threshold).mean()), rel=1e-12
    )


def test_the_metrics_file_records_the_size_of_every_partition(
    raw_csv: Path, tmp_path: Path
) -> None:
    """0.71 AUC on 116 rows with one positive is not the claim 0.71 on 40,000 is."""
    output_dir = tmp_path / "reports"
    run(raw_csv, output_dir)
    payload = read_metrics(output_dir)

    split = rebuild_split(raw_csv)
    assert payload["rows_train"] == len(split.x_train)
    assert payload["rows_validation"] == len(split.x_validation)
    assert payload["rows_test"] == len(split.x_test)
    assert payload["include_lender_priced"] is False


def test_the_three_partitions_are_disjoint_and_chronological(raw_csv: Path, tmp_path: Path) -> None:
    """Overlap is the split error that improves every metric instead of raising."""
    split = rebuild_split(raw_csv)

    assert split.x_train["issue_d"].max() < split.x_validation["issue_d"].min()
    assert split.x_validation["issue_d"].max() < split.x_test["issue_d"].min()
    assert set(split.x_train.index).isdisjoint(split.x_test.index)


def test_b01_the_persisted_model_never_saw_a_post_origination_column(
    raw_csv: Path, tmp_path: Path
) -> None:
    """The raw frame carries `recoveries` and `last_fico_range_high`; the spec does not."""
    output_dir = tmp_path / "reports"
    run(raw_csv, output_dir)
    spec = load_model(output_dir).named_steps["canonicalize"].spec

    for name in ("recoveries", "total_pymnt", "last_fico_range_high", "id", "url", "loan_status"):
        assert name not in spec.raw_inputs
        assert name not in spec.model_features
    # Off by default: the lender's own price does not exist for an applicant
    # nobody has priced yet.
    for name in ("int_rate", "grade", "sub_grade", "installment"):
        assert name not in spec.model_features


def test_the_lender_priced_tier_can_be_opted_into(raw_csv: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    run(raw_csv, output_dir, include_lender_priced=True)
    spec = load_model(output_dir).named_steps["canonicalize"].spec

    for name in ("int_rate", "grade", "sub_grade", "installment"):
        assert name in spec.model_features


def test_the_persisted_model_scores_a_single_raw_applicant(
    raw_csv: Path, tmp_path: Path, raw_loans: pd.DataFrame
) -> None:
    """The serving path: one raw row off the extract, straight through the pickle."""
    output_dir = tmp_path / "reports"
    run(raw_csv, output_dir)

    probability = score(load_model(output_dir), raw_loans.iloc[[0]])[0]
    assert 0.0 < probability < 1.0


def test_an_alias_named_extract_runs_unchanged(raw_loans: pd.DataFrame, tmp_path: Path) -> None:
    """The `loans_full_schema` spelling of the same columns."""
    raw_path = tmp_path / "aliased.csv"
    raw_loans.drop(columns=["funded_amnt"]).rename(
        columns={
            "loan_amnt": "loan_amount",
            "annual_inc": "annual_income",
            "dti": "debt_to_income",
            "issue_d": "issue_month",
            "loan_status": "status",
        }
    ).to_csv(raw_path, index=False)

    run(raw_path, tmp_path / "reports")
    spec = load_model(tmp_path / "reports").named_steps["canonicalize"].spec

    # Canonical names, not the extract's: everything downstream sees one vocabulary.
    assert "loan_amnt" in spec.raw_inputs
    assert "loan_amount" not in spec.raw_inputs


def test_an_extract_without_revol_util_derives_utilization_from_the_bureau_columns(
    raw_loans: pd.DataFrame, tmp_path: Path
) -> None:
    """`requires_any`: the alternate extract carries balance and limit, not a ratio."""
    raw_path = tmp_path / "bureau.csv"
    utilized = raw_loans["revol_bal"]
    raw_loans.drop(columns=["revol_util"]).assign(
        total_credit_utilized=utilized,
        total_credit_limit=utilized * 2.5 + 1_000.0,
    ).to_csv(raw_path, index=False)

    run(raw_path, tmp_path / "reports")
    spec = load_model(tmp_path / "reports").named_steps["canonicalize"].spec

    assert "credit_utilization" in spec.numeric_features
    assert "revol_util" not in spec.raw_inputs


def test_an_unsupported_model_type_is_rejected_before_any_work(
    raw_csv: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "reports"
    with pytest.raises(ValueError, match="Supported model types"):
        run(raw_csv, output_dir, model_type="random_forest")
    # Rejected before the run created anything, so a bad argument leaves no
    # half-populated report tree behind.
    assert not output_dir.exists()
