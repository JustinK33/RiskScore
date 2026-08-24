"""End-to-end orchestration: read an extract, fit a model, publish a run.

This module holds no logic of its own beyond ordering, and the order is the
point (see ``docs/architecture.md``)::

    raw -> closed statuses -> maturity embargo -> term filter -> label
        -> leakage audit -> feature spec -> tri-split
      TRAIN      fit the pipeline. Nothing else.
      VALIDATION fit the calibrator, then select the decision threshold.
      TEST       score once, report, never fit anything.

Every row filter sits left of the split, so all three partitions share one
outcome definition. Five properties are worth stating because all five were
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
* **The output is one atomic run directory, not a tree of loose files.** The old
  run wrote ``reports/models/logistic_regression.joblib`` and overwrote it on the
  next invocation, with the threshold in a separate JSON that a reader could pair
  with the wrong pickle. A run now publishes a
  :class:`~risk_score.artifacts.ScoringBundle` under an immutable id, and becomes
  visible only once it is complete - so a dashboard polling the tree can never
  read new metrics against an old calibration curve.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from risk_score.artifacts import (
    DEFAULT_RETENTION,
    HEADLINE_METRICS,
    RunMetadata,
    ScoringBundle,
    build_run_id,
    dataset_fingerprint,
    feature_tier,
    git_commit,
    library_versions,
    now_iso,
    prune_runs,
    prune_staging,
    register_run,
    save_bundle,
    staged_run,
)
from risk_score.cache import read_raw_loans_cached
from risk_score.calibration import (
    build_calibration_report,
    fit_calibrator,
    plot_calibration_curve,
)
from risk_score.config import RunConfig
from risk_score.data_loading import (
    DEFAULT_STATUSES,
    PAID_STATUSES,
    EmbargoResult,
    apply_outcome_maturity_embargo,
    create_default_target,
    filter_to_closed_loans,
    filter_to_terms,
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
from risk_score.logging_setup import bind_run_id, capture_run_log
from risk_score.modeling import SUPPORTED_MODEL_TYPES, split_by_time, train_model
from risk_score.transformers import build_feature_spec

LOGGER = logging.getLogger(__name__)

#: File names inside a run directory. Named here rather than inlined because the
#: API serves them from an allowlist and the dashboard fetches them by name, so
#: there is one spelling of each.
METRICS_FILENAME = "metrics.json"
CALIBRATION_TEST_FILENAME = "calibration_test.csv"
CALIBRATION_VALIDATION_FILENAME = "calibration_validation.csv"
THRESHOLD_COSTS_FILENAME = "threshold_costs_validation.csv"
CALIBRATION_FIGURE = "figures/calibration_test.png"
RUN_LOG_FILENAME = "run.log"


@dataclass(frozen=True, slots=True)
class RunResult:
    """What a completed run is, for the caller that started it.

    The ``run_id`` is here because the CLI, the model card, and the API's
    rollback all need to name the run afterwards, and deriving it a second time
    would produce a different timestamp.
    """

    run_id: str
    run_dir: Path
    metrics: ClassificationMetrics
    metadata: RunMetadata
    bundle: ScoringBundle
    #: The full ``metrics.json`` payload, so a caller does not read back a file
    #: it just wrote.
    payload: dict[str, Any]


def target_definition() -> str:
    """The label rule, in words, for the manifest and the model card.

    Built from the status sets rather than written out, because a hand-written
    copy is one edit away from describing a different label than the one the run
    used - and it is the manifest's job to be trustworthy about exactly that.
    """
    return (
        f"1 = {sorted(DEFAULT_STATUSES)}; 0 = {sorted(PAID_STATUSES)}; "
        "any other status is excluded rather than treated as repaid"
    )


def _predict_default_probability(estimator: Any, features: pd.DataFrame) -> pd.Series:
    """Positive-class default probabilities, indexed like the input frame.

    Takes any fitted estimator with ``predict_proba``, because the same call is
    made against the raw ``Pipeline`` and against the ``CalibratedClassifierCV``
    that wraps it - and both accept the same *raw* frame, since canonicalization
    is the pipeline's first step.
    """
    probabilities = estimator.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=features.index, name="default_probability")


def train_run(
    raw_data_path: str | Path,
    *,
    config: RunConfig | None = None,
    output_dir: str | Path = "reports",
    model_type: str = "logistic_regression",
    make_active: bool = True,
    keep_runs: int = DEFAULT_RETENTION,
    cache_dir: str | Path | None = None,
) -> RunResult:
    """Fit one model on one extract and publish it as a run directory.

    Everything the run varies - split windows, cost matrix, hyperparameters,
    feature tier, extra column aliases - arrives in one validated
    :class:`~risk_score.config.RunConfig`, so there is no second place a window
    or a cost can be specified and disagree. Omitting it uses the shipped
    defaults, which is a complete configuration.

    ``config.include_lender_priced`` admits ``int_rate``/``grade``/``sub_grade``/
    ``installment``. Off by default: they are the lender's own price, so a model
    using them cannot score an applicant nobody has priced yet. See
    ``docs/decisions/0005-lender-priced-feature-tier.md``.

    ``make_active=False`` publishes and registers the run without pointing the
    service at it, which is what ``riskscore compare`` needs: it fits two models
    and only one of them should be served.

    ``cache_dir`` enables the parquet cache for the canonicalized extract, which
    is what makes a second run on the 1.19 GB file fast. Off by default, because
    a library call that writes files into the working directory unasked is a
    surprise; the CLI turns it on.
    """
    # Checked before anything is created, so a typo leaves no half-written run
    # directory behind.
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Supported model types are {list(SUPPORTED_MODEL_TYPES)}; got {model_type!r}."
        )

    config = config or RunConfig()
    root = Path(output_dir)
    dataset = Path(raw_data_path)
    # Same reason as the model-type check: the first thing the run does otherwise
    # is create `output_dir/runs/`, so a mistyped path used to leave an empty
    # report tree behind and the error arrived from inside the staging block.
    if not dataset.is_file():
        raise FileNotFoundError(f"No extract at {dataset}")
    commit = git_commit()
    # One instant, two spellings: the run id needs a filename-safe basic form and
    # the manifest needs the extended one the registry sorts by. Taking the clock
    # twice would let them disagree by a second.
    started_at = datetime.now(UTC)
    run_id = build_run_id(
        model_type=model_type,
        include_lender_priced=config.include_lender_priced,
        created_at=started_at,
        commit=commit,
    )
    # Anything left by a previous SIGKILL - an OOM during a fit on the real
    # extract - is swept here rather than by a cron nobody wrote. Age-gated, so a
    # fit running in another process is never touched.
    prune_staging(root)

    # The log is written *inside* the staging directory - context managers are
    # entered left to right, so `staging` is already bound - which means it is
    # published by the same rename as the metrics, and a failed run discards its
    # log along with the artifacts it describes.
    with (
        bind_run_id(run_id),
        staged_run(root, run_id) as staging,
        capture_run_log(staging / RUN_LOG_FILENAME),
    ):
        result = _execute_run(
            dataset,
            staging=staging,
            config=config,
            model_type=model_type,
            run_id=run_id,
            created_at=now_iso(started_at),
            commit=commit,
            cache_dir=cache_dir,
        )

    register_run(
        root,
        metadata=result.metadata,
        metrics={key: result.payload[key] for key in HEADLINE_METRICS},
        make_active=make_active,
    )
    removed = prune_runs(root, keep=keep_runs)
    if removed:
        LOGGER.info("retention removed %d run(s)", len(removed), extra={"removed": removed})
    return RunResult(
        run_id=run_id,
        run_dir=root / "runs" / run_id,
        metrics=result.metrics,
        metadata=result.metadata,
        bundle=result.bundle,
        payload=result.payload,
    )


def _execute_run(
    dataset: Path,
    *,
    staging: Path,
    config: RunConfig,
    model_type: str,
    run_id: str,
    created_at: str,
    commit: str,
    cache_dir: str | Path | None,
) -> RunResult:
    """The run itself, writing into ``staging``. Split out so the atomicity and
    retention wiring above stays readable, and so every ``return`` inside it is
    still covered by the staging directory's cleanup."""
    cost_matrix = config.cost_matrix
    include_lender_priced = config.include_lender_priced
    date_column = config.split.date_column

    # --- 1. which rows are admissible at all ------------------------------------
    # Every row filter happens here, before the split, so all three partitions
    # share one outcome definition. A filter applied per partition is how two
    # partitions end up answering different questions (see modeling.py's "What
    # must NOT live here").
    # Hashed once and used twice - as the cache key and as the manifest's record
    # of which extract this is - because hashing the file is itself a full read.
    fingerprint = dataset_fingerprint(dataset)
    raw, schema_report = read_raw_loans_cached(
        dataset,
        fingerprint=fingerprint,
        cache_dir=cache_dir,
        column_aliases=config.column_aliases or None,
    )
    rows: dict[str, int] = {"raw": len(raw)}
    loans = filter_to_closed_loans(raw)
    rows["closed"] = len(loans)
    LOGGER.info(
        "read %d rows, %d closed", rows["raw"], rows["closed"], extra={"dataset": str(dataset)}
    )

    # The correction this project exists to demonstrate. "Closed" is measured
    # against the extract's snapshot, so a loan too young to have finished paying
    # can only be closed by having defaulted - and the resulting bias grows with
    # vintage, which reads as credit-quality drift. The embargo keeps only loans
    # whose full term had elapsed by the snapshot.
    embargo = apply_outcome_maturity_embargo(
        loans, snapshot=config.data.snapshot, date_column=date_column
    )
    rows["mature"] = len(embargo.loans)
    # Downstream of the embargo, not an independent choice: 60-month loans survive
    # it only in the earliest vintages, so training on them and never seeing one
    # in validation or test is a term-mix cliff, not extra data.
    loans = filter_to_terms(embargo.loans, terms=config.data.term_months_in)
    rows["in_scope_terms"] = len(loans)
    loans = loans.assign(default_flag=create_default_target(loans))
    loans = loans.dropna(subset=["default_flag"])
    rows["labelled"] = len(loans)
    LOGGER.info("%s", embargo.summary())

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
    rows.update(
        train=len(split.x_train),
        validation=len(split.x_validation),
        test=len(split.x_test),
    )
    LOGGER.info("%s", split.summary())

    # The whole split goes in, and `train_model` decides what each model type is
    # allowed to see: the logistic baseline gets train only, XGBoost additionally
    # monitors validation to stop boosting early.
    model = train_model(model_type, split, spec=spec, config=config.model_params(model_type))

    # --- 2. the probability correction, fitted on validation only ---------------
    # `split.validation` rather than two frames: `fit_calibrator` accepts only the
    # wrapper, so the partition a fitted object came from is stated at the call
    # site rather than assumed.
    calibrator = fit_calibrator(model, split.validation)

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
    LOGGER.info(
        "test auc_roc=%.4f brier=%.4f approval_rate=%.3f",
        metrics.auc_roc,
        metrics.brier_score,
        metrics.approval_rate,
    )

    calibration_test = build_calibration_report(split.y_test, scores_test)
    calibration_validation = build_calibration_report(split.y_validation, scores_validation)

    # --- 5. publish -------------------------------------------------------------
    embargo_report = _embargo_summary(embargo)
    metadata = RunMetadata(
        run_id=run_id,
        created_at=created_at,
        model_type=model_type,
        feature_tier=feature_tier(include_lender_priced),
        git_commit=commit,
        dataset_path=str(dataset),
        dataset_sha256=fingerprint,
        dataset_bytes=dataset.stat().st_size,
        target_definition=target_definition(),
        rows=rows,
        split_windows={window.name: window.label() for window in split.windows},
        embargo=embargo_report,
        cost_matrix={
            "false_negative_cost": cost_matrix.false_negative_cost,
            "false_positive_cost": cost_matrix.false_positive_cost,
        },
        features={
            "tier": feature_tier(include_lender_priced),
            "summary": spec.summary(),
            "model_features": list(spec.model_features),
            "numeric": list(spec.numeric_features),
            "categorical": list(spec.categorical_features),
            "raw_inputs": list(spec.raw_inputs),
            "leakage": leakage_audit.summary(),
            "schema": schema_report.summary(),
        },
        library_versions=library_versions(),
    )
    bundle = ScoringBundle(
        pipeline=model,
        calibrator=calibrator,
        threshold=threshold,
        feature_spec=spec,
        metadata=metadata,
    )
    save_bundle(bundle, staging)

    payload = {
        "run_id": run_id,
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
        "rows_train": rows["train"],
        "rows_validation": rows["validation"],
        "rows_test": rows["test"],
        "rows": rows,
        "split": split.summary(),
        "features": spec.summary(),
        "leakage": leakage_audit.summary(),
        "include_lender_priced": include_lender_priced,
        # The embargo's own numbers, because "we corrected for survivorship bias"
        # is an assertion and these dicts are the evidence.
        "embargo": embargo_report["summary"],
        "embargo_snapshot": embargo_report["snapshot"],
        "embargo_rows_immature": embargo_report["rows_immature"],
        "embargo_rows_unknown_maturity": embargo_report["rows_unknown_maturity"],
        "default_rate_by_vintage_before_embargo": embargo_report["default_rate_by_vintage_before"],
        "default_rate_by_vintage_after_embargo": embargo_report["default_rate_by_vintage_after"],
        "term_months_in": list(config.data.term_months_in),
        "rows_after_embargo_and_term_filter": rows["in_scope_terms"],
    }
    (staging / METRICS_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    calibration_test.curve.to_csv(staging / CALIBRATION_TEST_FILENAME, index=False)
    calibration_validation.curve.to_csv(staging / CALIBRATION_VALIDATION_FILENAME, index=False)
    threshold_costs.to_csv(staging / THRESHOLD_COSTS_FILENAME, index=False)
    # `plot_calibration_curve` writes where it is told and does not create
    # directories, which is correct for a plotting helper and means the caller
    # makes the subdirectory.
    (staging / "figures").mkdir(parents=True, exist_ok=True)
    plot_calibration_curve(calibration_test.curve, output_path=staging / CALIBRATION_FIGURE)

    return RunResult(
        run_id=run_id,
        run_dir=staging,
        metrics=metrics,
        metadata=metadata,
        bundle=bundle,
        payload=payload,
    )


def _embargo_summary(embargo: EmbargoResult) -> dict[str, Any]:
    """The embargo's numbers, for the manifest and for ``metrics.json``.

    Years are stringly keyed because JSON object keys are strings either way, and
    converting here keeps a round-trip through the file symmetric.
    """
    return {
        "summary": embargo.summary(),
        "snapshot": str(embargo.snapshot.date()),
        "rows_immature": embargo.rows_immature,
        "rows_unknown_maturity": embargo.rows_unknown_maturity,
        "default_rate_by_vintage_before": {
            str(year): rate for year, rate in embargo.default_rate_before.items()
        },
        "default_rate_by_vintage_after": {
            str(year): rate for year, rate in embargo.default_rate_after.items()
        },
    }
