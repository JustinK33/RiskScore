# Credit Default Risk Modeling

## Project Overview

This project is a portfolio-grade data science and quantitative modeling project focused on predicting credit default risk using the Lending Club loan dataset.

The goal is to build an interpretable, leakage-aware credit risk pipeline that moves from raw loan data through feature engineering, model training, evaluation, calibration, and threshold selection.

## Problem Statement

Given borrower and loan information available at origination time, estimate the probability that a loan will default.

The modeling objective is not only to maximize predictive performance, but also to produce a defensible risk model with clear leakage controls, time-based validation, calibration analysis, and business-aware threshold selection.

## How This Project Works

1. Load Lending Club loan records and filter to loans with closed outcomes only.
2. Remove post-origination fields that would leak future information into training.
3. Engineer borrower and loan risk features such as DTI, utilization, FICO bands, and loan-to-income ratio.
4. Split data by loan origination date to simulate a realistic future scoring workflow.
5. Train a logistic regression baseline and a gradient boosted tree model.
6. Evaluate discrimination, ranking quality, calibration, and business cost tradeoffs.
7. Explain model behavior with SHAP and summarize results in reproducible reports.

## Technologies

- Python
- pandas and numpy for data manipulation
- scikit-learn for preprocessing, metrics, calibration, and logistic regression baseline
- XGBoost for the main gradient boosted tree model
- SHAP for model explainability
- matplotlib and seaborn for visualization
- pytest for automated testing
- Optional simple experiment tracking through structured output files or MLflow

## Key Design Decisions

### Leakage Handling

Credit default modeling is highly sensitive to data leakage.

Many Lending Club columns are only known after origination, after payment history has developed, or after the loan has reached a terminal status.

This project separates origination-time features from post-origination fields in `src/risk_score/leakage_check.py`.

Post-origination fields are flagged and excluded before model training so that the model reflects information that would have been available at the actual underwriting decision point.

### Time-Based Split

Random train/test splitting can overstate real-world performance for credit risk models because lending behavior, borrower mix, macroeconomic conditions, and underwriting policies change over time.

This project uses a time-based split on loan issue date to train on earlier vintages and test on later vintages.

That design better approximates how a model would perform when deployed on future loan applications.

### Baseline Before Complexity

The first model should be a regularized logistic regression baseline.

This creates a transparent benchmark before adding a more flexible model such as XGBoost.

The baseline makes it easier to catch data issues, leakage, unstable features, and calibration problems before optimizing a more complex estimator.

### Business-Aware Thresholding

Model scores are probabilities, but lending decisions require thresholds.

The threshold selection module is designed around configurable false negative and false positive costs so model evaluation can reflect business tradeoffs rather than accuracy alone.

## Project Structure

```text
.
├── configs/
│   ├── feature_config.yaml
│   └── model_config.yaml
├── data/
│   ├── README.md
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── notebooks/
│   ├── README.md
│   └── .gitkeep
├── reports/
│   ├── figures/
│   │   └── .gitkeep
│   ├── metrics/
│   │   └── .gitkeep
│   └── models/
│       └── .gitkeep
├── src/
│   └── risk_score/
│       ├── __init__.py
│       ├── calibration.py
│       ├── data_loading.py
│       ├── evaluation.py
│       ├── feature_engineering.py
│       ├── leakage_check.py
│       └── modeling.py
├── tests/
│   ├── __init__.py
│   └── test_feature_engineering.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Getting Started

Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project in editable mode with development dependencies.

```bash
pip install -e ".[dev]"
```

Run the test suite.

```bash
pytest
```

Run the MVP logistic regression baseline after placing Lending Club data under `data/raw/`.

```bash
python scripts/run_baseline.py \
  --raw-data-path data/raw/lending_club_loans.csv \
  --train-end-date 2016-12-31 \
  --test-start-date 2017-01-01
```

The baseline writes metrics to `reports/metrics/`, a calibration plot to `reports/figures/`, and a fitted model artifact to `reports/models/`.

To run the XGBoost model instead, pass `--model-type xgboost`.

## Data

Download Lending Club loan data from the original source or a trusted mirror and place raw files under `data/raw/`.

Raw data is intentionally ignored by git because the files can be large and may have redistribution restrictions.

Processed feature matrices and model-ready datasets should be written to `data/processed/`.

## Results

Results will be added after implementation.

Planned reporting artifacts include:

- AUC-ROC and precision-recall metrics
- KS statistic
- Calibration plot
- Threshold cost analysis
- Feature importance and SHAP summaries
- Model card style summary of assumptions, limitations, and intended use

## Resume Talking Points

- Built a leakage-aware credit default risk pipeline using Lending Club loan data.
- Used time-based validation to approximate future deployment performance.
- Compared interpretable logistic regression against XGBoost.
- Evaluated discrimination, calibration, KS statistic, and threshold cost tradeoffs.
- Designed the project as a reproducible Python package with tests, configs, and report artifacts.
