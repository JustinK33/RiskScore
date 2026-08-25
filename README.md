# Credit Default Risk Modeling

An interpretable, leakage-aware machine learning pipeline that predicts the probability a loan will default, using the Lending Club dataset - and a scoring service that serves it.

![RiskScore dashboard demo](riskscore.gif)

*The local dashboard showing model metrics, calibration plot, and threshold cost analysis.*

## What It Does

Given a borrower's information at the time they apply for a loan, this project estimates the probability that the loan will default, and returns the reasons behind the number.

Training moves from the raw extract through feature engineering, calibration, and business-aware threshold selection, comparing a logistic regression baseline against XGBoost on one identical split.
Each run publishes an immutable, self-describing directory: the model, the calibrator, the threshold it chose, every metric and figure, and a generated model card.

Serving loads one of those directories and answers `POST /predict` in single-digit milliseconds with a calibrated probability, an approve/decline decision at the run's own threshold, and exact SHAP reason codes that sum back to the model's log-odds.

The focus is a defensible risk model rather than a high-scoring one.
That means post-origination columns excluded by name, the lender's own pricing held out by default and its cost measured rather than asserted, an outcome-maturity embargo that removes the survivorship bias in a naive closed-loans filter, and a calibrator and threshold fitted on validation so the test set is scored exactly once.

[docs/README.md](docs/README.md) is the index: [architecture](docs/architecture.md), a [page per source file](docs/code/), the [data dictionary](docs/data-dictionary.md), the [runbook](docs/runbook.md), and the [decision records](docs/decisions/).

## Tech Stack

- Python 3.12+
- pandas / numpy / scipy
- scikit-learn
- XGBoost
- FastAPI / uvicorn / pydantic
- matplotlib
- Vanilla ES modules for the dashboard - no npm, no bundler
- pytest, hypothesis, ruff, mypy strict

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

Then serve the run you just published - the API and the dashboard come from the same process, bound to localhost:

```bash
riskscore serve            # http://127.0.0.1:8000
```

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "loan_amnt": 12000, "term": " 36 months", "annual_inc": 62000,
  "purpose": "debt_consolidation", "home_ownership": "RENT",
  "dti": 18.4, "revol_util": "62.5%", "issue_d": "Jun-2015",
  "earliest_cr_line": "Aug-2003", "open_acc": 9, "total_acc": 22
}'
```

The request takes the raw extract's dialect or the parsed forms interchangeably, because the parsing is a step inside the pickled pipeline rather than a copy of the training code.
`GET /api/schema` is the authoritative field list for whichever bundle is loaded.

For the real thing, place the Lending Club extract under `data/raw/` and point `train` at it.
Pass `--model xgboost` to fit XGBoost instead, and `--include-lender-priced` to admit the interest rate and grade - features that are the lender's own price, so they are excluded by default.
`riskscore compare` fits several variants on one split and publishes the comparison.

`riskscore runs` lists every published run and marks the active one; `riskscore activate <run_id>` rolls back to an earlier one without retraining.
`riskscore --help` documents the rest.
