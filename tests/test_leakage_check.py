"""Tests for the leakage audit."""

from __future__ import annotations

import pandas as pd
import pytest

from risk_score.leakage_check import audit_columns, select_model_features


def test_post_origination_columns_are_refused_with_a_reason() -> None:
    audit = audit_columns(["loan_amnt", "total_pymnt", "recoveries", "last_fico_range_high"])

    assert audit.admitted == ("loan_amnt",)
    assert set(audit.refused) == {"total_pymnt", "recoveries", "last_fico_range_high"}
    assert "after the loan was funded" in audit.refused["total_pymnt"]


def test_lender_priced_columns_are_refused_by_default_and_admitted_on_request() -> None:
    columns = ["loan_amnt", "int_rate", "grade", "sub_grade", "installment"]

    default = audit_columns(columns)
    opted_in = audit_columns(columns, include_lender_priced=True)

    assert default.admitted == ("loan_amnt",)
    assert "lender's own risk estimate" in default.refused["int_rate"]
    assert set(opted_in.admitted) == set(columns)
    assert opted_in.refused == {}


def test_an_unrecognized_column_is_blocked_and_flagged_not_admitted() -> None:
    """The load-bearing difference between an allow list and a deny list. A future
    extract adding `settlement_amount` - post-charge-off, a near-perfect
    predictor - sailed through the old deny list as an ordinary feature."""
    audit = audit_columns(["loan_amnt", "settlement_amount"])

    assert audit.admitted == ("loan_amnt",)
    assert audit.unclassified == ("settlement_amount",)
    assert not audit.is_clean


def test_engineered_features_are_admitted() -> None:
    audit = audit_columns(["dti_clean", "credit_utilization", "credit_history_months"])

    assert len(audit.admitted) == 3
    assert audit.is_clean


def test_kept_columns_pass_through_without_counting_as_features() -> None:
    """The target has to survive the pass, but it is not a feature and must not
    be reported as one."""
    audit = audit_columns(["loan_amnt", "default_flag"], keep_columns=("default_flag",))

    assert audit.admitted == ("loan_amnt",)
    assert "default_flag" not in audit.refused
    assert "default_flag" not in audit.unclassified


def test_b26_select_model_features_is_one_pass_not_two() -> None:
    """The old pipeline dropped post-origination columns and then immediately
    selected origination-time ones, so the first pass copied the whole frame to
    drop columns the second was about to drop anyway."""
    loans = pd.DataFrame(
        {
            "loan_amnt": [1000],
            "annual_inc": [50_000],
            "total_pymnt": [900.0],
            "int_rate": [0.13],
            "url": ["http://example.com/1"],
            "default_flag": [1],
            "issue_d": pd.to_datetime(["2015-01-01"]),
        }
    )

    result, audit = select_model_features(loans)

    assert set(result.columns) == {"loan_amnt", "annual_inc", "default_flag", "issue_d"}
    assert set(audit.refused) == {"total_pymnt", "int_rate", "url"}


def test_select_model_features_keeps_issue_d_for_the_split() -> None:
    """The split needs it and drops it before the model sees it, so it is kept
    but never counted as a feature."""
    loans = pd.DataFrame({"loan_amnt": [1000], "issue_d": pd.to_datetime(["2015-01-01"])})

    result, audit = select_model_features(loans)

    assert "issue_d" in result.columns
    assert "issue_d" not in audit.admitted


def test_strict_mode_refuses_to_run_on_an_unclassified_column() -> None:
    """Onboarding a new extract should be deliberate, not something a training
    run does quietly."""
    loans = pd.DataFrame({"loan_amnt": [1000], "settlement_amount": [500.0]})

    with pytest.raises(ValueError, match="Unclassified columns present"):
        select_model_features(loans, strict=True)

    # Non-strict is the audit path: it drops and records instead of aborting.
    result, audit = select_model_features(loans)
    assert "settlement_amount" not in result.columns
    assert audit.unclassified == ("settlement_amount",)


def test_audit_summary_states_the_tier_setting() -> None:
    """A metrics file reporting a suspiciously good AUC needs to say, on its own,
    whether the lender's price was one of the inputs."""
    assert "lender_priced=off" in audit_columns(["loan_amnt"]).summary()
    assert "lender_priced=on" in audit_columns(["loan_amnt"], include_lender_priced=True).summary()
