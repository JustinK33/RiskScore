# Credit Default Risk Modeling

An interpretable, leakage-aware machine learning pipeline that predicts the probability a loan will default, using the Lending Club dataset.

![RiskScore dashboard demo](riskscore.gif)

*The local dashboard showing model metrics, calibration plot, and threshold cost analysis.*

## What It Does

Given a borrower's information at the time they apply for a loan, this project estimates the probability that the loan will default.
It moves from raw loan data through feature engineering, model training, calibration, and business-aware threshold selection, comparing a logistic regression baseline against XGBoost.
The focus is on a defensible risk model, not just a high-scoring one: no data leakage, time-based validation, and clear tradeoffs between false positives and false negatives.

## Tech Stack

- Python
- pandas / numpy
- scikit-learn
- XGBoost
- SHAP
- matplotlib / seaborn
- pytest

## Install and Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

No download is needed to try it.
`make-sample-data` writes a synthetic extract shaped like the real one, including the survivorship bias the pipeline has to correct for:

```bash
riskscore make-sample-data data/sample/loans.csv
riskscore train data/sample/loans.csv
riskscore runs
```

For the real thing, place the Lending Club extract under `data/raw/` and point `train` at it.
Pass `--model xgboost` to fit XGBoost instead, and `--include-lender-priced` to admit the interest rate and grade - features that are the lender's own price, so they are excluded by default.

Each run publishes an immutable directory under `reports/runs/<run_id>/` holding the model, its calibration, the threshold it selected, and every metric and figure.
`riskscore runs` lists them and marks the active one; `riskscore activate <run_id>` rolls back to an earlier one without retraining.
`riskscore --help` documents the rest.
