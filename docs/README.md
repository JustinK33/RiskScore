# RiskScore documentation

Credit default risk scoring on Lending Club data: a leakage-aware training pipeline, versioned model bundles, and a FastAPI scoring service.

## Start here

| If you want to | Read |
| --- | --- |
| Understand how the pieces fit together | [architecture.md](architecture.md) |
| Know what a specific source file is for | [code/](code/) - one page per code file |
| Know what a column means and where it comes from | [data-dictionary.md](data-dictionary.md) |
| Run, deploy, retrain, or roll back | [runbook.md](runbook.md) |
| Know why something was built this way | [decisions/](decisions/) |
| See what a run actually reports | [examples/model_card.example.md](examples/model_card.example.md) |
| Write code that matches the codebase | [conventions.md](conventions.md) |

## The short version

Lending Club publishes loan-level outcomes, which makes it a convenient default-risk dataset and a minefield of subtle mistakes.
This project is organized around four of them.

**Post-origination leakage.**
Roughly 20 columns in the extract describe what happened *after* the loan was funded - payments received, recoveries, updated FICO.
A model that sees them scores near-perfectly and predicts nothing.
They are excluded by name, with a documented reason per column.

**Lender-priced features.**
`int_rate`, `grade`, `sub_grade`, and `installment` are available at origination, so a naive leakage check passes them.
But they *are* the lender's own risk estimate, so a model built on them mostly reproduces an existing underwriting decision rather than predicting default from borrower attributes.
They live in a separate `LENDER_PRICED` tier, excluded by default, and both variants are reported so the difference is measured rather than asserted.
See [decisions/0005-lender-priced-feature-tier.md](decisions/0005-lender-priced-feature-tier.md).

**Outcome-maturity survivorship bias.**
Filtering to closed loans looks obviously correct and is quietly wrong.
A 36-month loan issued in 2016 has only closed by a 2018 snapshot if it *defaulted early*; the ones still paying read as `Current` and get dropped.
The measured default rate by vintage therefore climbs from 15.6% (2013) to 24.3% (2016) and then falls to 14.7% (2018) - a shape driven entirely by the snapshot date.
An outcome-maturity embargo keeps only loans whose full term has elapsed, which flattens it to a steady ~14.9%.
See [decisions/0004-outcome-maturity-embargo.md](decisions/0004-outcome-maturity-embargo.md).

**Fitting the decision on the test set.**
A decision threshold and a probability calibrator are fitted objects.
Choosing them on test data makes the reported cost and Brier score optimistic in a way no amount of cross-validation catches.
The split is train / validation / test: the estimator sees train, the calibrator and threshold see validation, and test is scored exactly once for reporting.
See [decisions/0003-train-validation-test-split.md](decisions/0003-train-validation-test-split.md).

## Documentation scope

`docs/code/` covers every file that contains executable logic: `src/risk_score/*.py`, `src/risk_score/api/*.py`, `dashboard/index.html`, `dashboard/styles/*.css`, `dashboard/js/*.js`, `tests/conftest.py`, and both files in `scripts/`.
Configuration files, `.gitignore`, and top-level READMEs are intentionally out of scope - they are either self-describing or documented where they are used.
The `*.test.js` files have no page of their own; each module's page names its tests.
The two `__init__.py` files are exempt with a stated reason.

[`scripts/check_docs.py`](code/check_docs.md) enforces this in CI: every in-scope code file must have a page, every page must correspond to a file that still exists, every page must carry all seven headings, and every page must be linked from the index.
It is in scope itself, because a rule that exempts its own enforcement is a rule with a hole in it.
Documentation that can rot silently does.
