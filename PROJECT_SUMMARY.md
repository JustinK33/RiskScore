# RiskScore Project Summary

## Executive Summary

RiskScore is a credit default risk modeling project built around a reproducible Python pipeline.

The pipeline ingests Lending Club-like loan CSV or parquet files, normalizes common dataset schemas, filters to terminal loan outcomes, creates a binary default target, removes known post-origination leakage fields, engineers underwriting-time risk features, trains a baseline classifier, evaluates model quality, and writes report artifacts.

The project now also includes a lightweight local dashboard that lets users visualize model metrics and upload their own compatible CSV files from a browser.

## High-Level Workflow

1. A user provides a raw loan dataset as CSV or parquet.
2. The loader normalizes common column names into the internal schema.
3. The loader filters rows to closed loan outcomes.
4. The target builder maps loan outcomes into `default_flag`.
5. Leakage controls remove known post-origination fields.
6. The pipeline keeps only documented origination-time columns.
7. Feature engineering adds model-ready risk features.
8. A time-based train/test split simulates future scoring.
9. The selected model trains on the training vintage.
10. The model scores the test vintage.
11. Evaluation writes metrics, calibration data, a calibration plot, and a model artifact.
12. The dashboard reads the latest report artifacts and displays them.

## Repository Infrastructure

The project is organized as a Python package under `src/risk_score`.

The command-line entrypoint lives in `scripts/run_baseline.py`.

The local dashboard server lives in `scripts/serve_dashboard.py`.

Static dashboard assets live under `dashboard/`.

Model and schema settings live under `configs/`.

Raw datasets live under `data/raw/`.

Processed datasets live under `data/processed/`.

Metrics, figures, and fitted models are written under `reports/`.

Automated tests live under `tests/`.

Raw data, processed data, reports, models, virtual environments, caches, and local download helpers are ignored by git.

Only `.gitkeep` placeholders are tracked in the data and report artifact directories.

## Dataset Compatibility Layer

The project now includes `src/risk_score/schema.py`.

This module maps common Lending Club-like column names into one internal schema.

The canonical schema uses names such as `loan_status`, `issue_d`, `loan_amnt`, `annual_inc`, `dti`, and `revol_util`.

The compatibility layer accepts alternate names such as `status`, `issue_month`, `loan_amount`, `annual_income`, `debt_to_income`, and `revolUtil`.

Configurable aliases live in `configs/run.yaml`.

When a new CSV uses different column names, the intended workflow is to add those names to `configs/run.yaml`.

The pipeline code should not need to change for ordinary schema-name variations.

The date normalizer supports mixed common date formats, including month strings such as `Jan-2018`.

FICO range columns are used when present.

The pipeline can still run when `fico_range_low` and `fico_range_high` are both absent.

If only one FICO range column is present, the pipeline raises a clear schema error.

## Data Loading

Data loading is handled by `src/risk_score/data_loading.py`.

The loader supports `.csv`, `.parquet`, and `.pq` inputs.

CSV files are loaded with `pandas.read_csv`.

Parquet files are loaded with `pandas.read_parquet`.

After loading, the dataset is normalized through the schema compatibility layer.

The loader requires a normalized `loan_status` column.

The loader filters to closed statuses only.

Closed statuses currently include `Fully Paid`, `Charged Off`, `Default`, `Does not meet the credit policy. Status:Fully Paid`, and `Does not meet the credit policy. Status:Charged Off`.

This filtering prevents non-terminal outcomes such as `Current` from entering supervised default modeling.

## Target Creation

Target creation is handled by `create_default_target`.

The target column is named `default_flag`.

Paid loans map to `0`.

Charged-off and defaulted loans map to `1`.

Ambiguous or unsupported statuses are left missing.

The pipeline drops rows where `default_flag` is missing before training.

## Leakage Controls

Leakage controls live in `src/risk_score/leakage_check.py`.

The project distinguishes origination-time fields from post-origination fields.

Origination-time fields are borrower, loan, and underwriting attributes that would plausibly be available when the loan is issued.

Post-origination fields include payment totals, recoveries, last payment dates, current outstanding principal, and later credit pulls.

Known post-origination columns are removed before feature engineering.

The pipeline also selects only documented origination-time columns before modeling.

This prevents raw dataset extras such as IDs, URLs, free-text descriptions, and unknown servicing fields from accidentally reaching the model.

This design is important because credit risk models are highly vulnerable to leakage.

## Feature Engineering

Feature engineering lives in `src/risk_score/feature_engineering.py`.

`dti_clean` is created by coercing the raw DTI field into numeric form.

`credit_utilization` is created from `revol_util`.

Percentage strings such as `47.5%` are parsed correctly.

Numeric utilization values are also supported.

`fico_midpoint` and `fico_band` are created when both FICO range columns exist.

FICO bands are currently `poor`, `fair`, `good`, `very_good`, and `exceptional`.

`loan_to_income_ratio` is created as loan amount divided by annual income.

Zero annual income is treated as missing for that ratio to avoid division by zero.

## Modeling

Model training lives in `src/risk_score/modeling.py`.

The MVP model is a scikit-learn logistic regression pipeline.

The model uses a `ColumnTransformer` to handle numeric and categorical features.

Numeric features are median-imputed and standardized.

Categorical features are most-frequent-imputed and one-hot encoded.

The logistic regression model defaults to balanced class weights.

The pipeline also includes an XGBoost training path.

XGBoost is selected with `--model-type xgboost`.

The model type and parameters are configured through `configs/run.yaml`.

## Time-Based Validation

The pipeline uses a time-based train/test split.

This is more realistic than a random split for credit risk modeling.

Earlier loan vintages are used for training.

Later loan vintages are used for testing.

The split is controlled by `train_end_date` and `test_start_date`.

The default command-line date column is `issue_d`.

The split raises an error if either side is empty.

The split raises an error if date parsing fails.

## Evaluation

Evaluation lives in `src/risk_score/evaluation.py`.

The baseline reports AUC ROC.

The baseline reports average precision.

The baseline reports the KS statistic.

The baseline selects a cost-aware decision threshold.

False negative and false positive costs are configurable from the CLI.

The default false negative cost is `5.0`.

The default false positive cost is `1.0`.

Metrics are written to `reports/metrics/logistic_regression_metrics.json`.

Calibration data is written to `reports/metrics/logistic_regression_calibration.csv`.

The calibration plot is written to `reports/figures/logistic_regression_calibration.png`.

The fitted model is written to `reports/models/logistic_regression.joblib`.

## Dashboard

The dashboard is a static frontend served by a small Python HTTP server.

It does not require Node, React, Vite, or a frontend build step.

The server is started with `python scripts/serve_dashboard.py`.

The default dashboard URL is `http://127.0.0.1:8765`.

The dashboard reads metrics from `/api/metrics`.

The dashboard reads calibration rows from `/api/calibration`.

The dashboard links to report artifacts under `/artifacts/`.

The dashboard displays metric cards for AUC ROC, average precision, KS statistic, and selected threshold.

The dashboard draws a calibration chart on an HTML canvas.

The dashboard includes a CSV upload form.

Users can upload a Lending Club-like CSV, choose split dates, and run the baseline locally.

The upload endpoint writes user CSVs under `data/raw/uploads/`.

Uploaded CSV files are ignored by git.

After a successful upload run, the dashboard refreshes to show the latest metrics.

## Command-Line Usage

Run the logistic regression baseline with a compatible CSV.

```bash
python scripts/run_baseline.py \
  --raw-data-path data/raw/lending_club_loans.csv \
  --train-end-date 2016-12-31 \
  --test-start-date 2017-01-01
```

Run the baseline with the dataset schema alias config.

```bash
python scripts/run_baseline.py \
  --raw-data-path data/raw/my_loans.csv \
  --train-end-date 2016-12-31 \
  --test-start-date 2017-01-01 \
  --schema-config configs/run.yaml
```

Start the dashboard.

```bash
python scripts/serve_dashboard.py
```

Run tests.

```bash
pytest
```

Run lint.

```bash
ruff check dashboard scripts src tests
```

## Metrics

The real larger Lending Club-style run used `data/raw/1/loan.csv`.

That file contained about 1.3 million closed loans after filtering.

The closed-loan outcomes included roughly 1,041,952 `Fully Paid`, 261,655 `Charged Off`, 31 `Default`, 1,988 policy-status fully paid loans, and 761 policy-status charged-off loans.

The full-data logistic regression result was:

| Metric | Value |
| --- | ---: |
| AUC ROC | 0.7007 |
| Average Precision | 0.3611 |
| KS Statistic | 0.2925 |
| Selected Threshold | 0.42 |
| False Negative Cost | 5.0 |
| False Positive Cost | 1.0 |

The current local report artifact was later overwritten by a tiny dashboard upload smoke test.

That smoke test is useful only for verifying the upload endpoint.

Its metrics are not meaningful as model-performance evidence because the dataset had only eight rows.

The smoke-test artifact currently reports:

| Metric | Value |
| --- | ---: |
| AUC ROC | 1.0000 |
| Average Precision | 1.0000 |
| KS Statistic | 1.0000 |
| Selected Threshold | 0.16 |
| False Negative Cost | 5.0 |
| False Positive Cost | 1.0 |

## Validation Performed

The full Python test suite passed with 20 tests.

Ruff linting passed for `dashboard`, `scripts`, `src`, and `tests`.

The dashboard homepage returned HTTP 200.

The dashboard metrics API returned JSON successfully.

The dashboard calibration API returned calibration rows successfully.

The CSV upload endpoint was tested with an alternate schema.

The alternate schema test used fields such as `status`, `issue_month`, `loan_amount`, `annual_income`, `debt_to_income`, and `revolUtil`.

The upload endpoint successfully normalized that CSV, ran the baseline, and wrote report artifacts.

Git ignore rules were verified for raw datasets, uploaded CSVs, processed datasets, report artifacts, model artifacts, and `down.py`.

## Current Limitations

The pipeline is generalized for Lending Club-like credit risk datasets, not arbitrary CSVs.

A CSV still needs a loan outcome column, issue date column, loan amount, annual income, DTI, and revolving utilization or a compatible alias.

Very different datasets will need aliases added to `configs/run.yaml`.

The dashboard upload currently runs the logistic regression baseline only.

The dashboard upload sends CSV text through a local JSON request, which is fine for local use but not ideal for very large production uploads.

The model artifacts are overwritten on each run.

The latest dashboard always reflects the most recent successful run.

The pipeline currently stores only aggregate metrics, calibration data, the plot, and the fitted model.

It does not yet store a persistent run history.

## Recommended Next Improvements

Add a run-history directory so each pipeline execution gets its own timestamped outputs.

Add schema validation that reports missing required fields before model training starts.

Add a dashboard preview step that shows detected columns, mapped columns, dropped columns, and missing fields before the user runs the model.

Add support for chunked upload or direct filesystem selection for very large local CSV files.

Add model comparison views for logistic regression and XGBoost.

Add threshold confusion matrix outputs.

Add feature importance reporting.

Add SHAP summary artifacts for XGBoost.

Add a model card artifact that summarizes data vintage, target definition, leakage assumptions, validation split, and intended use.

## Resume Bullets

- Built a leakage-aware credit default risk modeling pipeline using Python, pandas, scikit-learn, and XGBoost-ready infrastructure.
- Designed a schema normalization layer that lets Lending Club-like CSV datasets with different column names map into one canonical modeling schema.
- Implemented terminal-outcome filtering and binary default target creation for supervised credit risk modeling.
- Added origination-time feature selection to prevent post-origination leakage from payment history, recoveries, servicing fields, IDs, URLs, and raw free-text columns.
- Engineered borrower and loan risk features including cleaned DTI, revolving utilization, FICO bands when available, and loan-to-income ratio.
- Used time-based validation to train on earlier loan vintages and test on later vintages, better approximating future deployment performance.
- Trained and evaluated a regularized logistic regression baseline with AUC ROC, average precision, KS statistic, calibration outputs, and cost-aware threshold selection.
- Added a local dashboard for visualizing model metrics, calibration behavior, run details, and generated artifacts.
- Built a browser-based CSV upload workflow that lets users import their own Lending Club-like CSV, select train/test dates, and run the baseline locally.
- Added automated tests covering alternate dataset schemas, optional FICO fields, leakage-safe feature selection, and end-to-end pipeline execution.
- Verified the project with 20 passing tests and Ruff lint checks.
- Achieved a logistic regression full-dataset baseline of 0.7007 AUC ROC, 0.3611 average precision, and 0.2925 KS statistic on the larger Lending Club-style dataset.
