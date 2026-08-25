# Data dictionary

Every column this project knows about, what it means, where it comes from, and - for the ones the model never sees - why not.

The authority is `src/risk_score/features.py`, not this file.
Fifty `ColumnSpec` entries and six `EngineeredFeature` entries live there as code, validated at import, and the tables below are transcribed from them.
If the two disagree, the code is right and this page is stale; see [code/features.md](code/features.md).

## How to read a row

| Field | Meaning |
| --- | --- |
| **Column** | The canonical name. Every alias is renamed to this before anything else happens. |
| **Parse** | The declared parse rule. Never inferred from the values - see below. |
| **Req** | Whether a run aborts when the column is absent. |
| **Aliases** | Other spellings seen in real extracts, renamed on load. |

**The parse rule is declared, not inferred.**
This is the single most load-bearing decision in the file.
`int_rate` arrives as `13.56%` in one published extract and as `13.56` in another; `revol_util` does the same.
Inferring the scale from the values means a batch of low-utilization applicants reads as already-fractional and gets scored a hundred times too low, silently, with a model that still fits.
So the registry declares `PERCENT` and the parser divides by exactly 100 every time, with a sanity check that *guards* the declaration rather than replacing it.

The seven parse kinds: `NUMERIC` (strips `$`, `,`, `%`, maps sentinels and infinities to NaN), `PERCENT` (`NUMERIC` then / 100), `TERM_MONTHS` (`' 36 months'` -> 36.0), `EMP_LENGTH_YEARS` (`'10+ years'` -> 10, `'< 1 year'` -> 0), `MONTH_DATE` (one whole-column format from an ordered list), `CATEGORY`, and `TEXT`.

**Three numeric sentinels are missingness, not values:** `-1.0`, `9999.0`, `999999.0`.
They are real numbers in the CSV, and left alone a model treats them as genuine extreme observations.

## The seven tiers

Every column has exactly one tier, and the tier decides whether the model may see it.

| Tier | Count | Reaches the model | Why |
| --- | --- | --- | --- |
| `LOAN_REQUEST` | 3 | **Yes** | What the applicant asked for. Known before any decision. |
| `BORROWER` | 18 | **Yes** | Who the applicant is, as of origination. |
| `LENDER_PRICED` | 4 | **Opt-in** | Available at origination, but it *is* the lender's own risk estimate. |
| `TIMELINE` | 1 | No | `issue_d` defines the split and the vintage breakdown. Using it as a feature is fitting the calendar. |
| `TARGET_SOURCE` | 1 | No | `loan_status` is the label's source. |
| `IDENTIFIER` | 6 | No | Row ids, URLs, free text, and one fair-lending exclusion. |
| `POST_ORIGINATION` | 17 | **Never** | Describes what happened *after* funding. |

An allow-list, not a deny-list.
A new column in a future extract is excluded by default and reported by the leakage audit, which is the opposite of the failure mode where a deny-list is one release out of date and something leaks through the gap.

### The tiers the model sees

#### `LOAN_REQUEST` - what was asked for

| Column | Parse | Req | Aliases | Meaning |
| --- | --- | --- | --- | --- |
| `loan_amnt` | NUMERIC | yes | `loan_amount`, `loanamnt`, `funded_amnt`, `fundedamnt` | Requested principal in dollars. |
| `term` | TERM_MONTHS | yes | `term_months` | Loan term in months. Also drives the outcome-maturity embargo. |
| `purpose` | CATEGORY | no | `loan_purpose` | Borrower-stated use of funds. |

`funded_amnt` as an alias of `loan_amnt` is where the duplicate-column bug lived (audit B02).
The standard extract contains **both**, so first-wins renaming produced a frame with two columns named `loan_amnt`, and `loans["loan_amnt"]` returned a DataFrame that died several functions later.
Aliases are now priority-ordered, the loser is dropped, and the drop is recorded in the schema report.

#### `BORROWER` - who is asking

| Column | Parse | Req | Aliases | Meaning |
| --- | --- | --- | --- | --- |
| `annual_inc` | NUMERIC | yes | `annual_income`, `annualinc`, `annualincome` | Self-reported annual income. |
| `emp_length` | EMP_LENGTH_YEARS | no | `emp_length_years`, `employment_length`, `emplength` | Years at current employer, capped at 10 by the source encoding. |
| `home_ownership` | CATEGORY | no | `homeownership` | RENT, MORTGAGE, OWN, or OTHER. |
| `verification_status` | CATEGORY | no | `verified_income`, `isincv` | Whether the platform verified stated income. |
| `addr_state` | CATEGORY | no | `state`, `addrstate` | Two-letter state of residence. |
| `dti` | NUMERIC | no | `debt_to_income`, `debttoincome` | Debt-to-income ratio excluding the requested loan, as a percentage. |
| `revol_util` | PERCENT | no | `revolutil`, `revol_utilization` | Revolving line utilization. Exceeds 100% in the real data. |
| `revol_bal` | NUMERIC | no | `revolbal` | Total revolving balance. |
| `delinq_2yrs` | NUMERIC | no | `delinq_2y`, `delinq2yrs` | 30+ day delinquencies in the last two years. |
| `inq_last_6mths` | NUMERIC | no | `inqlast6mths` | Credit inquiries in the last six months, excluding auto and mortgage. |
| `open_acc` | NUMERIC | no | `open_credit_lines`, `openacc` | Open credit lines. |
| `total_acc` | NUMERIC | no | `total_credit_lines`, `totalacc` | Total credit lines ever opened. |
| `pub_rec` | NUMERIC | no | `pubrec` | Derogatory public records. |
| `earliest_cr_line` | MONTH_DATE | no | `earliest_credit_line`, `earliestcrline` | First credit line opened. Sets `credit_history_months`. |
| `total_credit_utilized` | NUMERIC | no | `totalcreditutilized` | Bureau-style total balance; a fallback source for utilization. |
| `total_credit_limit` | NUMERIC | no | `totalcreditlimit` | Bureau-style total limit; the denominator for the utilization fallback. |
| `fico_range_low` | NUMERIC | no | `ficorangelow` | Lower bound of the origination FICO band. |
| `fico_range_high` | NUMERIC | no | `ficorangehigh` | Upper bound of the origination FICO band. |

**`emp_length` caps at 10 in the source, not in this code.**
`'10+ years'` means "ten or more", and nothing here can recover the true value.
Worth knowing before reading a reason code that names it.

**The two `fico_range_*` columns are absent from both published extracts.**
They are registered because the field name is documented and appears in some derived datasets, and because the alternative - an `EngineeredFeature` that references a column the registry has never heard of - is worse.
In practice `fico_midpoint` and `fico_band` are dead on real data, and the README used to imply otherwise.

`revol_util` is the reason the utilization fallback exists.
When it is missing, `credit_utilization` is computed from `total_credit_utilized / total_credit_limit` instead, and the fallback is declared in the engineered feature's `requires_any` rather than discovered at runtime.

#### `LENDER_PRICED` - excluded by default

| Column | Parse | Req | Aliases | Meaning |
| --- | --- | --- | --- | --- |
| `int_rate` | PERCENT | no | `interest_rate`, `intrate` | Assigned interest rate: the lender's own risk estimate, priced. |
| `grade` | CATEGORY | no | - | Assigned letter grade. A deterministic function of `int_rate`. |
| `sub_grade` | CATEGORY | no | `subgrade` | Assigned sub-grade, a finer slice of the same pricing decision. |
| `installment` | NUMERIC | no | - | Monthly payment: an exact function of amount, rate, and term. |

These four pass every leakage check that asks "was this known at origination?", because they were.
They fail the question that matters: **is this the model's own job, already done by somebody else?**

A model built on `int_rate` mostly reproduces Lending Club's underwriting decision.
It scores well and it cannot score an applicant nobody has priced yet, which is every applicant a scoring service is asked about.

Admit them with `--include-lender-priced`, and `riskscore compare --tiers both` fits both variants on one identical split so the difference is *measured*.
Expect the AUC to jump.
That jump is the finding, not the result.
See [decisions/0005-lender-priced-feature-tier.md](decisions/0005-lender-priced-feature-tier.md).

### The tiers the model never sees

#### `POST_ORIGINATION` - the leakage tier

Seventeen columns that describe what happened after the money moved.
A model that sees them scores near-perfectly and predicts nothing, which is the most common way a Lending Club notebook goes wrong.

| Column | Why it leaks |
| --- | --- |
| `last_fico_range_high` | **The leakiest column in the extract.** Updated FICO, which falls roughly 100 points at charge-off. |
| `last_fico_range_low` | Same, lower bound. |
| `recoveries` | Post-charge-off recoveries. A non-zero value *implies* the label. |
| `collection_recovery_fee` | Charged only after delinquency. |
| `total_rec_late_fee` | Late fees directly reveal repayment behaviour. |
| `out_prncp`, `out_prncp_inv` | Outstanding principal: known only once payments occur. |
| `total_pymnt`, `total_pymnt_inv` | Payments received to date. |
| `total_rec_prncp`, `total_rec_int` | Principal and interest received to date. |
| `last_pymnt_d`, `last_pymnt_amnt` | Last payment date and amount. |
| `next_pymnt_d` | Next scheduled payment: servicing state. |
| `last_credit_pull_d` | May postdate origination. |
| `collections_12_mths_ex_med` | The reporting window may postdate origination. |
| `pymnt_plan` | Set during servicing. |

#### `IDENTIFIER` - not features

| Column | Why not |
| --- | --- |
| `id` | Row identifier. **Correlates with time**, so it leaks the vintage and lets a model learn the calendar. |
| `member_id` | Fully null in every public extract. |
| `url` | Listing URL. Contains the row id. |
| `desc` | Free-text borrower description. Out of scope for this model. |
| `policy_code` | Constant `1.0` everywhere. Encodes nothing. |
| `zip_code` | **A fair-lending exclusion, not a technical one.** A truncated ZIP proxies for protected characteristics far more tightly than `addr_state` does. |

`zip_code` is the one row in this file excluded for a reason that is not about accuracy.
It is in the registry, with that sentence, rather than simply left out - a column silently absent is indistinguishable from a column nobody thought about.

#### `TIMELINE` and `TARGET_SOURCE`

| Column | Parse | Req | Aliases | Role |
| --- | --- | --- | --- | --- |
| `issue_d` | MONTH_DATE | yes | `issue_month`, `issuedate`, `issuemonth`, `issue_d_month` | Origination month. Defines the train/validation/test windows, the embargo, and the vintage breakdown. |
| `loan_status` | CATEGORY | yes | `loanstatus`, `status` | Terminal or in-flight outcome. The label is derived from it. |

`loan_status` used to carry `n` as an alias, which is how a one-letter column in an unrelated extract silently became the target.
Aliases now have a three-character minimum, and `status`/`addr_state`-style collisions with a real canonical column are resolved by priority rather than by whichever came first.

Both are required, both are used everywhere, and neither is a feature.
`issue_d` as a feature is fitting the calendar, which generalizes to exactly zero future applicants.

## Engineered features

Six, declared with their dependencies so a missing source column is a named error rather than a `KeyError` in a transformer.

| Feature | Kind | Requires | Consumes | Meaning |
| --- | --- | --- | --- | --- |
| `dti_clean` | numeric | `dti` | `dti` | Debt-to-income parsed to a float and bounded to a plausible range. |
| `credit_utilization` | numeric | `revol_util` **or** (`total_credit_utilized` + `total_credit_limit`) | `revol_util` | Revolving utilization as a fraction. Winsorized above 200%. |
| `loan_to_income_ratio` | numeric | `loan_amnt`, `annual_inc` | - | Requested principal divided by annual income. |
| `credit_history_months` | numeric | `earliest_cr_line`, `issue_d` | `earliest_cr_line` | Months between first credit line and origination. |
| `fico_midpoint` | numeric | `fico_range_low`, `fico_range_high` | both | Midpoint of the origination FICO band. Dead on both real extracts. |
| `fico_band` | category | `fico_midpoint` | - | Interpretable FICO bucket, for reporting rather than accuracy. |

**`consumes` is the column that gets dropped.**
This is the fix for the audit's largest bug (B01): the old code added `dti_clean` beside `dti` and left the raw string in the frame, where a dtype-based `ColumnTransformer` routed it to `OneHotEncoder`.
`earliest_cr_line` has 655 distinct values in the real extract, and `int_rate` as a percent string has about 600 - roughly 2400 dense one-hot columns over 1.8M rows, or about 35 GB.
The pipeline was not slow on real data; it was non-functional.

`credit_utilization` is winsorized rather than dropped above 200%.
Real `revol_util` runs past 800% for genuinely over-limit borrowers, and that is signal, but a single 8.9 in a scaled feature dominates the coefficient and the risk signal saturates long before then.

`credit_history_months` is derived rather than read.
The extract has no such column, only `earliest_cr_line`, and the number a reader wants is the distance from it to origination.

## The label

`loan_status`, lowercased, matched against two frozen sets.

| Set | Members | Label |
| --- | --- | --- |
| Default | `charged off`, `default`, `does not meet the credit policy. status:charged off` | **1** |
| Paid | `fully paid`, `does not meet the credit policy. status:fully paid` | **0** |

Anything else - `Current`, `Late (31-120 days)`, `In Grace Period`, `Issued` - is **dropped**, because the outcome is not known yet.

The `Does not meet the credit policy` variants are included in both directions.
They are ordinary funded loans with a prefixed status string, and dropping them silently discards a population with a materially different default rate.

The default set is immutable and is the only way to change what a default *is*.
Passing a different set is possible; it is a deliberate, greppable argument rather than a config value nobody notices.

## The two filters, in order

Both run **before** the split, so all three partitions share one definition of the outcome.

**1. Closed statuses.**
Keep only rows whose status is in one of the two sets above.

**2. The outcome-maturity embargo.**
Keep only rows where `issue_d + term <= snapshot`.

Filtering to closed loans and stopping there looks obviously correct and is quietly wrong.
A 36-month loan issued in 2016 has only closed by a 2018 snapshot **if it defaulted early**; the ones still paying read as `Current` and get dropped by filter 1.
So the surviving 2016 rows are disproportionately defaults, and the measured default rate by vintage climbs and then falls:

| Vintage | Closed only | With the embargo |
| --- | --- | --- |
| 2013 | 15.6% | 15.6% |
| 2014 | 18.5% | 13.7% |
| 2015 | 20.2% | 14.9% |
| 2016 | **24.3%** | - |
| 2018 | **14.7%** | - |

The left column is a shape driven entirely by the snapshot date.
The right column is a steady rate.
Every run's manifest records how many rows the embargo removed and the per-vintage counts before and after, so the correction is visible rather than asserted.
See [decisions/0004-outcome-maturity-embargo.md](decisions/0004-outcome-maturity-embargo.md).

## Dates

`MONTH_DATE` columns are parsed with **one whole-column format**, chosen from an ordered list by trying each against the whole column:

```
%b-%Y   %Y-%m-%d   %Y-%m-%d %H:%M:%S   %Y-%m-%dT%H:%M:%S   %Y-%m   %d-%b-%Y   %b-%y   %B %Y
```

Not `format="mixed"`, which infers per row.
`03/04/2016` is March in one row's inference and April in another's, and pandas will happily produce a column containing both readings with no warning.
An unparseable column raises and names the formats it tried; individual unparseable *rows* are quarantined and counted rather than aborting the run.

## What a scoring request looks like

`GET /api/schema` is the authoritative field list for the **loaded bundle** - generated from its own `FeatureSpec`, so it changes when the model does.
It accepts either dialect: the raw extract's strings (`" 36 months"`, `"62.5%"`, `"Jun-2015"`) or the parsed forms (`36`, `62.5`, `"2015-06"`), because the same `CanonicalizeFrame` transformer runs at fit time and at request time.

Unknown fields are refused with a 422 rather than ignored.
A silently dropped typo returns a score computed from an imputed value, with nothing anywhere to show it happened.
