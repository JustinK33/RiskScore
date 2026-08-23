# 0005 - Separate lender-priced features into their own tier, excluded by default

Status: accepted.
Affects `src/risk_score/features.py`, `src/risk_score/leakage_check.py`.

## Context

The original leakage control was a single binary: a column is either available at origination or it is not.
`int_rate`, `installment`, `grade`, and `sub_grade` are all set at origination, before the first payment, so they were listed in `ORIGINATION_TIME_COLUMNS` and fed to the model without comment.

Timing is the wrong test for these four.

Lending Club assigns a grade and a rate by running its own underwriting model on the same borrower attributes the model here is trying to use.
`sub_grade` is a 35-level ordinal that is, to a first approximation, the lender's predicted probability of default discretized into buckets.
`int_rate` is a monotone function of `sub_grade`.
`installment` is determined by `loan_amnt`, `term`, and `int_rate`, so it carries the rate as well.

A model given `sub_grade` therefore mostly learns to reproduce an existing decision.
It scores well, it looks like a good result, and it answers a question nobody asked: given that Lending Club has already assessed this borrower, what did they conclude?

There are two concrete problems beyond the philosophical one.

**It cannot be served.**
The scoring API takes an applicant who has not been priced yet.
There is no `int_rate` to send, because producing one is what the lender does *after* a risk assessment.
A model that requires `sub_grade` at inference time can only score loans that already exist.

**It hides the borrower-attribute signal.**
Because `sub_grade` dominates, the coefficients on `dti`, `credit_utilization`, and `credit_history_months` shrink toward zero, and a feature-importance chart reports that debt-to-income barely matters.
That is a statement about collinearity with the lender's grade, not about credit risk, and it reads as the latter.

## Decision

A third tier, `LENDER_PRICED`, holding exactly `int_rate`, `grade`, `sub_grade`, and `installment`.

- Excluded from the feature set by default.
- Included when `include_lender_priced=True`.
- Always *read* from the extract, so the comparison between both variants comes from one pass over the data rather than two runs that could differ in other ways.
- The setting is recorded in the run manifest, the leakage audit summary, and the model card.

Both variants are reported so the difference is a measured number.
Claiming that lender-priced features "would be leakage" without quantifying it is an assertion; reporting that including them moves AUC from one value to another, on the same split, is a result.

## Consequences

**The default AUC is lower.**
This is the point.
The default configuration answers "can borrower attributes predict default", which is the question a credit risk model is for.

**"Leakage" now means three different things in this codebase, and they are named separately.**
Post-origination columns are refused unconditionally.
Lender-priced columns are refused by default and available on request.
Unclassified columns are refused and flagged for a decision.
Collapsing these into one deny list is what let `int_rate` through.

**Anyone reading a metrics file needs to know which variant produced it.**
Hence the tier in the run id and in `LeakageAudit.summary()`, rather than only in a config file next to it.

## Consequences for the API

`/api/schema` derives the request model from the active bundle's `FeatureSpec`, so a default-tier bundle does not accept `int_rate` at all - sending it is a validation error rather than a silently ignored field.
A lender-priced bundle does accept it, and the dashboard's scoring form grows the field.
The service does not have to know which variant it loaded.

## Alternatives considered

**Keep them in and add a caveat to the README.**
The original state. A caveat in prose does not survive someone reading only the metrics JSON, and it does not make the model servable.

**Drop them from the registry entirely.**
Simpler, and it forecloses the comparison. The leakage cost stops being a number and goes back to being an assertion. It also loses `int_rate` as a *reporting* variable - relating predicted risk to the price actually charged is a genuinely useful chart.

**Keep `int_rate` and drop `grade`/`sub_grade`, on the theory that a continuous rate is less of a direct label proxy.**
It is not: `int_rate` is a deterministic function of `sub_grade` in this dataset. Splitting the four would create a tier boundary with no meaning behind it.
