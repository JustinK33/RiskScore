# `src/risk_score/explain.py`

## Purpose

Answer "why this score", exactly, for one applicant.

A credit decision that cannot be explained is, under ECOA and FCRA, a decision that cannot lawfully be made: an adverse action notice has to state *reasons*, not a probability.
So reason codes are not decoration here, they are the difference between a score and a decision.

What this module computes is SHAP values: each feature's contribution to this row's log-odds, measured against a background population, such that

```
log_odds(applicant) = baseline + sum(contributions)
```

That identity is the whole reason to use SHAP rather than coefficient magnitudes or a permutation ranking.
It is additive and exact, so a reason code is a number a reviewer can recompute, and the tests assert it against the model's own `decision_function`.

**There is no `shap` dependency, and that is deliberate.**
For the two model families this project fits, the exact values are already available without it.
A linear model's interventional SHAP value is exactly `coef_i * (x_i - E[x_i])` - one multiply, which is what `shap.LinearExplainer` computes.
XGBoost implements TreeSHAP inside the booster: `booster.predict(dmatrix, pred_contribs=True)` returns the exact contributions plus a bias column, and `shap.TreeExplainer` dispatches to that same C++ code.
The package would add a numba/llvmlite toolchain to the serving image, and a wheel-availability risk on every new Python release, to reach code paths this module reaches in a dozen lines.

## Public API

| Name | What it is |
| --- | --- |
| `Explainer` | Built once from a bundle, called per applicant. All the setup cost is in `__init__`. |
| `Explainer.explain` | A raw frame in, one `Explanation` per row out. |
| `Explainer.global_summary` | Mean absolute contribution per feature over a population, ranked. |
| `Explanation` | `baseline_log_odds`, every contribution sorted by magnitude, `total_log_odds`, `top(k)`, `adverse_reasons(k)`. |
| `Contribution` | `feature`, `label`, `value`, `log_odds`, `direction`, `sentence()`. |
| `build_background` | The transformed training sample a bundle carries so an explainer needs no training data. |
| `feature_label` | The human wording for a feature, read from the registry that declares it. |
| `transformed_feature_names`, `feature_sources` | Design-matrix column names, and which feature each came from. |
| `BACKGROUND_ROWS`, `DEFAULT_TOP_K` | 200 and 5. |

## Inputs and outputs

In: a `ScoringBundle`, and raw applicant frames in exactly the shape `POST /predict` receives.
Out: `Explanation` objects, and one `shap_summary.csv` per run.

Nothing here reads a file or a run directory.
An `Explainer` is constructed from the bundle alone, which is what lets the service build one at boot and keeps the training extract off the serving machine entirely.

```python
explainer = Explainer(load_active_bundle("reports"))
for reason in explainer.explain(applicant_frame)[0].adverse_reasons(k=3):
    print(reason.sentence())
# credit_utilization=0.87 increases risk (+0.412 log-odds)
# loan_to_income_ratio=0.41 increases risk (+0.288 log-odds)
# emp_length=1.0 increases risk (+0.130 log-odds)
```

`shap_summary.csv` carries `feature`, `label`, `mean_abs_log_odds`, `mean_log_odds`, `columns`, `rank`.
`metrics.json` carries only `top_features` (five names) and `explainer` (`linear` or `tree`), because a metrics file that grows a row per feature stops being readable.

## Invariants and failure modes

**The contributions reconstruct the model's own margin.**
`baseline + sum(contributions)` equals `decision_function` for the linear path and `predict(output_margin=True)` for the tree path, to 1e-9 and 1e-6 respectively.
This is the property that says the explanation is exact rather than indicative, and it is the first test in the file.

**Explanations are complete, never truncated.**
`explain` returns every feature; `top(k)` is a slice.
Truncating inside `explain` would leave a baseline that no longer sums with its parts, which quietly destroys the property above.

**Log-odds, not probability.**
Contributions are additive only on the log-odds scale, and they describe the *uncalibrated* pipeline.
Calibration is a monotone map applied afterwards, so it changes the probability but not the ranking or the sign of any contribution.

**One-hot families collapse to their source feature.**
The model sees `purpose_debt_consolidation`; the applicant has a `purpose`.
Forty near-zero one-hot columns are not forty reasons, and the collapsed name is also the field the request payload carries, so a reason code names something the caller sent.
`missingindicator_annual_inc` is attributed to `annual_inc` for the same reason: the fact that a value was absent is a fact *about that feature*.

**A column that cannot be attributed is reported under its own name and logged at `WARNING`.**
It means the preprocessor grew a step this module does not know about.
An odd-looking reason code is a far better failure than an explanation that silently no longer sums to the score.

**Ordering is by magnitude, in either direction.**
A feature that strongly reduces risk is as much of an explanation as one that raises it, so both are eligible for `top(k)`.
`adverse_reasons(k)` is the filtered view a declined applicant is owed - a rejection explained by the two things that helped them is not an explanation.

**The background is a uniform random sample, not a k-means summary.**
The linear path needs the background's column means to *be* the training population's column means, and the unweighted mean of k cluster centroids is not that: it over-weights sparse regions, which is exactly where a reason code matters.
Sampling is seeded and the selected indices are sorted, so two builds on the same data are byte-identical - a bundle that differed run to run could not be compared to itself.

**Refusals happen at construction, not at the first request.**
An estimator with neither `coef_` nor `get_booster` raises `TypeError`; a linear bundle with no background raises `ValueError`; a background whose width disagrees with the coefficient count raises rather than broadcasting into nonsense.
An explainer that builds and then fails per-applicant turns a deployment mistake into an outage.

**Global importance is measured on test, not on train.**
It is a claim about how the model behaves on rows it has not seen.
Measured on the fitted rows it over-weights whatever the model memorized.

**Mean absolute, with the signed mean beside it.**
A feature that pushes half the population up and half down is important and its signed mean is zero.
The `columns` count is published too, so a reader can see that `addr_state`'s importance is spread over fifty columns while `annual_inc`'s is one - the honest caveat on comparing a categorical's aggregate to a numeric's.

## What must NOT live here

- **Any fit.** An explainer reads a fitted model; it never trains, samples, or re-scales anything. The background comes from the bundle, which came from train.
- **Reading run directories or JSON.** The bundle is the only input. A second source of truth about which features the model uses is how an explanation starts disagreeing with the score.
- **Human-readable label text.** It comes from `ColumnSpec.description` and `EngineeredFeature.description`, so a new feature is declared in one place.
- **A `shap` import.** See Purpose. Adding one would also mean two code paths that can disagree about the same number.
- **Probability-space contributions.** They are not additive, so the sum would not reconstruct anything and the identity that makes this auditable would be gone.
- **Per-request pandas.** The one-hot grouping indices are computed once in `__init__` because a per-request `groupby` would put pandas on the hot path for arithmetic numpy does in microseconds.

## Related tests

`tests/test_explain.py`, 20 tests, built around the additivity property.

- `test_the_contributions_reconstruct_the_models_own_margin` and `test_tree_contributions_reconstruct_the_boosters_own_margin` are the reason this module can skip `shap`: both check against the model's own margin, not against a library agreeing with itself. The tree one is `requires_xgboost`.
- `test_the_baseline_is_the_score_of_an_average_applicant` pins the baseline: an applicant at the background mean must have no contributions at all.
- `test_every_declared_feature_appears_exactly_once_per_explanation` is what would catch a dropped feature silently folding into the baseline.
- `test_one_hot_columns_collapse_into_their_source_feature` and `test_a_missing_value_indicator_is_attributed_to_the_feature_it_is_about` cover the name mapping in both of its non-obvious cases.
- `test_a_reason_code_cites_the_applicants_own_value_not_a_scaled_one` is the readability claim, asserted against the engineered frame.
- `test_the_background_is_capped_and_deterministic` is the byte-identical requirement.
- `test_a_model_neither_path_can_explain_is_refused_at_construction` and `test_a_linear_bundle_without_a_background_is_refused` are the two construction-time refusals.
- `tests/test_pipeline.py::test_the_pipeline_writes_every_documented_artifact` includes `shap_summary.csv`, so the artifact cannot silently stop being written.

## Known limits

- **Interventional SHAP assumes independent features,** for both paths. With correlated inputs - `loan_amnt` and `installment`, `revol_bal` and `credit_utilization` - a shared effect is split between them in a way that depends on the model rather than on the data. Conditional SHAP would need the feature covariance and a much larger background, which is a real cost for a difference no reason code would show.
- **The explanation is of the uncalibrated model.** A caller wanting "how much did this feature move the *calibrated* probability" cannot get it additively, because the calibrator is not additive. The direction and the ranking transfer; the magnitudes are log-odds.
- **A categorical's aggregate is not comparable to a numeric's,** which is why `columns` is published rather than hidden. Fifty one-hot columns summing to a large magnitude is partly a statement about cardinality.
- **`global_summary` scores the whole frame it is given.** On the real extract's test partition that is fine; on 1.8M rows it would be a dense matrix multiply the caller should sample first. Nothing samples on the caller's behalf, because a silently sampled importance table is a lie about its own precision.
- **No interaction effects.** These are first-order attributions per feature. TreeSHAP can produce interaction values; nothing here asks for them, and no reason code has room for a pair.
- **The tree path builds a `DMatrix` per call.** A few hundred microseconds for one row, and it is why the latency budget for `/predict` with reasons is larger than without.
