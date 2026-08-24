"""End-to-end orchestration: read an extract, fit a model, report on holdout.

This module holds no logic of its own beyond ordering, and the order is the
point (see ``docs/architecture.md``)::

    raw -> closed statuses -> maturity embargo -> term filter -> label
        -> leakage audit -> feature spec -> tri-split
      TRAIN      fit the pipeline. Nothing else.
      VALIDATION fit the calibrator, then select the decision threshold.
      TEST       score once, report, never fit anything.

Every row filter sits left of the split, so all three partitions share one
outcome definition. Four properties are worth stating because all four were
previously violated:

* **Immature loans are removed before anything else.** Filtering to closed
  statuses alone keeps young loans only when they defaulted, which inflates the
  label for exactly the most recent vintages - the ones a time split reports on.
  The run records the per-vintage default rate before and after the correction.
* **The threshold is chosen on validation, not on test.** Choosing it on the
  same rows the headline metrics come from makes those metrics a description of
  the selection procedure (audit B04).
* **Calibration is applied, not only measured.** The old run drew a calibration
  curve and then threw the correction away, so the reported Brier score
  described a model nobody would have shipped (audit B05).
* **Feature engineering is not done here.** It happens inside the fitted
  ``Pipeline``, so the artifact this writes is self-contained and ``POST
  /predict`` cannot preprocess a request differently from how the model was fit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from risk_score.calibration import (
    build_calibration_report,
    fit_calibrator,
    plot_calibration_curve,
)
from risk_score.config import RunConfig
from risk_score.data_loading import (
    apply_outcome_maturity_embargo,
    create_default_target,
    filter_to_terms,
    load_lending_club_data,
)
from risk_score.evaluation import (
    ClassificationMetrics,
    ValidationScores,
    compute_auc_roc,
    compute_average_precision,
    compute_brier_score,
    compute_ks_statistic,
    compute_threshold_cost_table,
    select_threshold_by_cost,
)
from risk_score.leakage_check import audit_columns
from risk_score.modeling import SUPPORTED_MODEL_TYPES, split_by_time, train_model
from risk_score.transformers import build_feature_spec


def _predict_default_probability(estimator: Any, features: pd.DataFrame) -> pd.Series:
    """Positive-class default probabilities, indexed like the input frame.

    Takes any fitted estimator with ``predict_proba``, because the same call is
    made against the raw ``Pipeline`` and against the ``CalibratedClassifierCV``
    that wraps it - and both accept the same *raw* frame, since canonicalization
    is the pipeline's first step.
    """
    probabilities = estimator.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=features.index, name="default_probability")


def run_baseline_pipeline(
    raw_data_path: str | Path,
    *,
    config: RunConfig | None = None,
    output_dir: str | Path = "reports",
    model_type: str = "logistic_regression",
) -> ClassificationMetrics:
    """Fit one model on one extract and write its metrics and figures.

    Everything the run varies - split windows, cost matrix, hyperparameters,
    feature tier, extra column aliases - arrives in one validated
    :class:`~risk_score.config.RunConfig`, so there is no second place a window
    or a cost can be specified and disagree. Omitting it uses the shipped
    defaults, which is a complete configuration.

    ``config.include_lender_priced`` admits ``int_rate``/``grade``/``sub_grade``/
    ``installment``. Off by default: they are the lender's own price, so a model
    using them cannot score an applicant nobody has priced yet. See
    ``docs/decisions/0005-lender-priced-feature-tier.md``.
    """
    # Checked before anything is created, so a typo leaves no half-written
    # report tree behind.
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Supported model types are {list(SUPPORTED_MODEL_TYPES)}; got {model_type!r}."
        )

    config = config or RunConfig()
    output_path = Path(output_dir)
    metrics_path = output_path / "metrics"
    figures_path = output_path / "figures"
    models_path = output_path / "models"
    for directory in (metrics_path, figures_path, models_path):
        directory.mkdir(parents=True, exist_ok=True)

    cost_matrix = config.cost_matrix
    include_lender_priced = config.include_lender_priced
    date_column = config.split.date_column

    # --- 1. which rows are admissible at all ------------------------------------
    # Every row filter happens here, before the split, so all three partitions
    # share one outcome definition. A filter applied per partition is how two
    # partitions end up answering different questions (see modeling.py's "What
    # must NOT live here").
    loans = load_lending_club_data(
        raw_data_path,
        column_aliases=config.column_aliases or None,
    )
    # The correction this project exists to demonstrate. "Closed" is measured
    # against the extract's snapshot, so a loan too young to have finished paying
    # can only be closed by having defaulted - and the resulting bias grows with
    # vintage, which reads as credit-quality drift. The embargo keeps only loans
    # whose full term had elapsed by the snapshot.
    embargo = apply_outcome_maturity_embargo(
        loans, snapshot=config.data.snapshot, date_column=date_column
    )
    # Downstream of the embargo, not an independent choice: 60-month loans survive
    # it only in the earliest vintages, so training on them and never seeing one
    # in validation or test is a term-mix cliff, not extra data.
    loans = filter_to_terms(embargo.loans, terms=config.data.term_months_in)
    loans = loans.assign(default_flag=create_default_target(loans))
    loans = loans.dropna(subset=["default_flag"])

    # The audit is a record, not a filter. What actually keeps a post-origination
    # column out of the model is that `build_feature_spec` never puts one in the
    # spec, and `CanonicalizeFrame` reindexes to exactly the spec's inputs.
    leakage_audit = audit_columns(
        loans.columns,
        include_lender_priced=include_lender_priced,
        keep_columns=("default_flag", "loan_status", date_column),
    )

    spec = build_feature_spec(loans.columns, include_lender_priced=include_lender_priced)
    target = loans["default_flag"].astype(int)
    # The raw frame goes in whole. The pipeline's first step reduces it to the
    # declared inputs, so there is nothing to select here.
    split = split_by_time(
        loans.drop(columns=["default_flag"]),
        target,
        date_column=date_column,
        train=config.split.train,
        validation=config.split.validation,
        test=config.split.test,
    )

    # The whole split goes in, and `train_model` decides what each model type is
    # allowed to see: the logistic baseline gets train only, XGBoost additionally
    # monitors validation to stop boosting early.
    model = train_model(model_type, split, spec=spec, config=config.model_params(model_type))

    # --- 2. the probability correction, fitted on validation only ---------------
    # `split.validation` rather than two frames: `fit_calibrator` accepts only the
    # wrapper, so the partition a fitted object came from is stated at the call
    # site rather than assumed.
    calibrator = fit_calibrator(model, split.validation)
    # Two files, and they cannot drift: the calibrator holds this exact `model`
    # object by reference inside its FrozenEstimator, so the pickle contains both.
    # The bare pipeline is written too because it is the only way to inspect the
    # fitted preprocessing without unwrapping a calibrator.
    joblib.dump(model, models_path / f"{model_type}.joblib")
    joblib.dump(calibrator, models_path / f"{model_type}_calibrator.joblib")

    # --- 3. the decision rule, chosen on validation only ------------------------
    # Scored through the calibrator, because that is what serving compares to the
    # threshold. A threshold picked on uncalibrated scores and then applied to
    # calibrated ones is a different policy than the one that was costed.
    #
    # Known limit: the calibrator was fitted on these same rows, so the validation
    # cost table is mildly optimistic. Reported anyway rather than silently split
    # a fourth partition off a dataset this size; the honest number is the test
    # one below, which the calibrator never saw.
    scores_validation = _predict_default_probability(calibrator, split.x_validation)
    threshold = select_threshold_by_cost(
        ValidationScores(y_true=split.y_validation, y_score=scores_validation),
        cost_matrix=cost_matrix,
    )
    threshold_costs = compute_threshold_cost_table(
        split.y_validation, scores_validation, cost_matrix=cost_matrix
    )
    # Located by nearest value rather than `threshold_costs["threshold"] == threshold`:
    # both come from the same np.arange, so equality happens to hold today, and
    # would stop holding the moment a caller passes its own threshold grid -
    # raising IndexError on `.iloc[0]` of an empty selection (audit B11).
    selected = threshold_costs.loc[threshold_costs["threshold"].sub(threshold).abs().idxmin()]

    # --- 4. test is scored once, and only reported ------------------------------
    scores_test = _predict_default_probability(calibrator, split.x_test)
    # The uncalibrated score is kept for one number only: the Brier score the
    # correction was supposed to improve. Reporting the calibrated Brier without
    # it makes the calibration step unfalsifiable.
    scores_test_raw = _predict_default_probability(model, split.x_test)
    metrics = ClassificationMetrics(
        auc_roc=compute_auc_roc(split.y_test, scores_test),
        average_precision=compute_average_precision(split.y_test, scores_test),
        ks_statistic=compute_ks_statistic(split.y_test, scores_test),
        brier_score=compute_brier_score(split.y_test, scores_test),
        default_rate=float(split.y_test.mean()),
        # Measured on test, not read out of the validation cost table: the
        # approval rate a lender would actually see is a property of the
        # population being scored, not of the population the rule was tuned on.
        approval_rate=float((scores_test < threshold).mean()),
    )

    calibration_test = build_calibration_report(split.y_test, scores_test)
    calibration_validation = build_calibration_report(split.y_validation, scores_validation)

    metrics_payload = {
        "auc_roc": metrics.auc_roc,
        "average_precision": metrics.average_precision,
        "ks_statistic": metrics.ks_statistic,
        "brier_score": metrics.brier_score,
        # The pair that says whether calibrating helped. If the calibrated number
        # is not lower, the correction is not earning its place in the artifact.
        "brier_score_uncalibrated": compute_brier_score(split.y_test, scores_test_raw),
        "expected_calibration_error": calibration_test.expected_calibration_error,
        # In-sample for the calibrator, and labelled as such: it is the floor the
        # test number should be compared against, not a second result.
        "expected_calibration_error_validation_in_sample": (
            calibration_validation.expected_calibration_error
        ),
        "calibration_method": calibrator.method,
        "calibration_fitted_on": "validation",
        "default_rate": metrics.default_rate,
        "approval_rate": metrics.approval_rate,
        "selected_threshold": threshold,
        "selected_threshold_total_cost": float(selected["total_cost"]),
        "threshold_selected_on": "validation",
        "false_negative_cost": cost_matrix.false_negative_cost,
        "false_positive_cost": cost_matrix.false_positive_cost,
        "model_type": model_type,
        # Sample sizes travel with the metrics because a 0.71 AUC on 116 rows
        # with one positive is not the same claim as 0.71 on 40,000.
        "rows_train": len(split.x_train),
        "rows_validation": len(split.x_validation),
        "rows_test": len(split.x_test),
        "split": split.summary(),
        "features": spec.summary(),
        "leakage": leakage_audit.summary(),
        "include_lender_priced": include_lender_priced,
        # The embargo's own numbers, because "we corrected for survivorship bias"
        # is an assertion and these two dicts are the evidence. Years are stringly
        # keyed because JSON object keys are strings either way, and doing it here
        # keeps the round-trip symmetric.
        "embargo": embargo.summary(),
        "embargo_snapshot": str(embargo.snapshot.date()),
        "embargo_rows_immature": embargo.rows_immature,
        "embargo_rows_unknown_maturity": embargo.rows_unknown_maturity,
        "default_rate_by_vintage_before_embargo": {
            str(year): rate for year, rate in embargo.default_rate_before.items()
        },
        "default_rate_by_vintage_after_embargo": {
            str(year): rate for year, rate in embargo.default_rate_after.items()
        },
        "term_months_in": list(config.data.term_months_in),
        "rows_after_embargo_and_term_filter": len(loans),
    }
    (metrics_path / f"{model_type}_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    calibration_test.curve.to_csv(metrics_path / f"{model_type}_calibration.csv", index=False)
    threshold_costs.to_csv(metrics_path / f"{model_type}_threshold_costs.csv", index=False)
    plot_calibration_curve(
        calibration_test.curve,
        output_path=figures_path / f"{model_type}_calibration.png",
    )

    return metrics
