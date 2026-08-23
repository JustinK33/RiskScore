# `src/risk_score/leakage_check.py`

## Purpose

Produce a per-run, checkable record of which columns reached the model and why every other column did not.

The point is not to drop columns - that is one line.
The point is that "leakage was controlled" should be a record in the run manifest naming each admitted and refused column with a reason, rather than a claim in a README that nothing verifies.

## Public API

| Name | What it does |
| --- | --- |
| `LeakageAudit` | Frozen record: `admitted`, `refused` (column to reason), `unclassified`, `include_lender_priced`. Has `is_clean` and `summary()`. |
| `audit_columns(columns, ...)` | Classify names. Takes an iterable, so it is testable without a frame. |
| `select_model_features(loans, ...)` | Reduce a frame to admitted features plus `keep_columns`. Returns `(frame, audit)`. |
| `REFUSAL_REASONS` | Tier to the reason a reader needs. |

## Inputs and outputs

In: a frame or a list of column names, the tier setting, and the columns to pass through untouched.

Out: the reduced frame and the audit.

`keep_columns` defaults to `("default_flag", "issue_d")`.
The target has to survive the pass and so does the split column, but neither is a feature, and neither is counted as one in `audit.admitted`.
The split drops `issue_d` before the model sees it.

## Invariants and failure modes

### Allow-list first, which is the whole point

The previous version was a deny list: a hard-coded set of post-origination column names, dropped by name, everything else kept.

A deny list can only reject what someone thought to name.
A future extract adding `settlement_amount` - recorded after charge-off, and therefore a near-perfect predictor of default - would have sailed through as an ordinary feature, and the resulting AUC would have looked like a triumph.

An unrecognized column is now refused *and* reported in `audit.unclassified`.
It is not proof of leakage; it is proof that nobody has decided, which is the state this module exists to surface.
`strict=True` refuses to run at all, which is the training pipeline's setting: onboarding a new extract should be a deliberate act rather than something a run does quietly.

### Audit B26: one pass, not two

The pipeline called `exclude_leaky_columns` and then immediately `select_origination_time_columns`.
The first copied the entire frame to drop columns the second was about to drop anyway.
`select_model_features` is a single allow-list pass.

### Refusals carry reasons, not just counts

`REFUSAL_REASONS` is a mapping rather than a set of tiers because "dropped 17 columns" is not reviewable and "known only after the loan was funded, so it encodes the outcome" is.

### The tier setting travels with the audit

`summary()` reports `lender_priced=on|off`.
A metrics file showing a suspiciously good AUC needs to be able to say, on its own, whether the lender's own price was one of the inputs.
See [decisions/0005-lender-priced-feature-tier.md](../decisions/0005-lender-priced-feature-tier.md).

### Three kinds of refusal, named separately

Collapsing these into one list is what let `int_rate` through in the first place:

- **Post-origination, identifier, target-source, timeline** - refused unconditionally, at any setting.
- **Lender-priced** - refused by default, admitted on request, both variants reported.
- **Unclassified** - refused and flagged for a human decision.

## What must NOT live here

- **The column lists themselves.** They are in `features.py`. This module holds policy about tiers, not membership of them.
- **Row filtering.** It only ever selects columns.
- **Anything statistical.** A correlation-based leakage detector is a different tool with a different failure mode (it flags genuinely predictive features), and mixing the two would make both harder to trust.

## Related tests

`tests/test_leakage_check.py`.

Named audit regression: `test_b26_select_model_features_is_one_pass_not_two`.

The most important test is `test_an_unrecognized_column_is_blocked_and_flagged_not_admitted`, which is the difference between an allow list and a deny list stated as an assertion.

## Known limits

- The audit checks *names*, not content. A column correctly registered as `BORROWER` that an extract has silently repurposed to hold post-origination data would pass. Nothing short of reading the data dictionary catches that.
- `is_clean` is about classification coverage, not about safety. A run can be clean and still be built on a column whose tier was assigned wrongly.
- `keep_columns` is trusted. A caller who passes a post-origination column there bypasses the audit entirely, which is deliberate (the target itself is one) but means the default value is load-bearing.
