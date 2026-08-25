# `src/risk_score/api/scoring.py`

## Purpose

Turn one applicant into a probability, a decision, and a reason, with no HTTP and no disk in sight.

Before this file the project could not score anything.
`joblib.dump` wrote a pickle that nothing ever loaded, and no `predict` function existed anywhere in `src/risk_score/`.
A credit risk project that cannot score a single applicant is a report generator.

It is separate from the routes because the interesting failures are not HTTP failures.
"A required field was absent" and "the calibrated probability crossed the threshold" are testable without a client, and a test that needs a client to assert them is a test nobody writes enough of.

**The whole latency budget lives here.**
For a warm process scoring one applicant against a logistic-regression bundle: p50 <= 12 ms and p99 <= 25 ms without reason codes, p50 <= 25 ms and p99 <= 50 ms with them.
`riskscore bench` prints the measured figures; on the development machine they are 7.6/8.1 ms and 14.9/15.7 ms, so the budget carries roughly 50% headroom for a slower CI runner.

## Public API

| Name | What it is |
| --- | --- |
| `ScoringService` | A loaded bundle, its explainer, and the two things you can ask them. Constructed once per process. |
| `ScoringService.score` | One applicant. Reasons on by default. |
| `ScoringService.score_batch` | Many applicants in one pass. Reasons off by default. |
| `ScoringService.build_frame` | Rows of declared inputs, correctly typed, in the declared order. |
| `ScoringService.can_explain`, `.explainer_error`, `.metadata` | What the service can do, why not, and which run it is. |
| `Score`, `ReasonCode` | The two frozen result shapes. Plain data, so `schemas.py` maps them to a response without reformatting. |

## Inputs and outputs

Takes a `ScoringBundle` at construction and mappings at call time.
Returns `Score` objects.
Opens no files, ever - that is the point of the class.

`latency_ms` is measured inside `score` with `perf_counter`, so it covers frame construction, preprocessing, the model, the calibrator and the explainer, and excludes HTTP framing.
That is the number an operator can act on: the part this project controls.
`perf_counter` and not `time.time`, because the latter is subject to NTP adjustment and can measure a negative duration, which is a confusing thing to find in a latency histogram.

For a batch, one elapsed time is divided across the rows.
Timing each row separately would report the cost of a vectorized call as though it were per row, and the per-row figure is exactly what a caller compares against `/predict`.

## Invariants and failure modes

**Everything expensive has already happened before the request arrives.**
The pickle is loaded once when the app is built.
The SHAP explainer is constructed once against the persisted background, so a request does no k-means and touches no training data.
The one-row frame is built from a dict with declared dtypes.
Nothing here opens a file.

**Declared dtypes, not `pd.DataFrame(rows)`.**
Inference on a single row is both slow and frequently wrong: a numeric column whose only value is `None` infers as `object`, and `StandardScaler` then raises about a string it was never given.
`_DTYPE_BY_PARSE_KIND` maps every parse kind whose source dialect may be a string - which is all of them except plain numerics - to `object`, because `CanonicalizeFrame` is what parses those and handing it a pre-coerced column would be a second parsing policy.
Declaring `float64` for `term` would reject `' 36 months'` here rather than in the parser written to read it.
The parsers already branch on dtype, so numbers arriving in an `object` column cost one `astype` and nothing else.

**An absent key is not an error.**
It becomes NA and is handled downstream exactly as an absent column in the training extract was: imputed, with the missing indicator set.
That is what an applicant with no credit history looks like.
Unknown keys are dropped here rather than passed through, because `CanonicalizeFrame` would drop them anyway and dropping them early keeps the frame narrow.

**A bundle that cannot explain still scores.**
The explainer is built at construction and a failure is recorded rather than raised.
Finding out per request would turn a missing SHAP background into a 500 on the hot path, and refusing to start would take a service that can still score offline over a feature that is not the point of it.
`can_explain` and `explainer_error` are how `/api/model`, `riskscore bench`, and the startup log report it.

**Building the explainer eagerly is deliberate.**
A lazy build makes the first request after every deploy the slow one, which is exactly the request a smoke test measures.

**Reason codes cost a second pass through the preprocessor**, and that is the entire 7 ms difference between the two budgets.
The calibrator wraps the whole pipeline, so a calibrated probability can only be had from the raw frame, while the explainer needs the transformed matrix.
The two passes cannot be shared without either reaching into `CalibratedClassifierCV` internals or changing what the bundle calibrates.

**Preprocessing *is* the request.**
Measured, not assumed: about 5.5 ms in the two pandas transformers, 1.5 ms in the `ColumnTransformer`, and under 0.1 ms in the model.
That is why the optimizations that mattered were pandas-pass-count reductions in `feature_engineering.py` and `transformers.py` rather than anything model-shaped - and why the same changes sped up training.

**A batch does not pay the per-pass cost per row.**
200 applicants cost about 8.8 ms in total, 0.04 ms each, because the cost is per pass.
Scoring 1000 rows as 1000 one-row frames is roughly two orders of magnitude slower than one 1000-row frame.

**`with_reasons` defaults to `False` on `score_batch` and `True` on `score`.**
Not an inconsistency: a batch is a scoring job, and a single call is a decision somebody has to justify.

**Reason values are JSON-serializable before they leave.**
`_reason` unwraps numpy scalars with `.item()`, turns a non-finite float into `None` (there is no JSON spelling of `NaN` that a strict parser accepts), and renders a `Timestamp` as ISO 8601.

## What must NOT live here

- **HTTP.** No `Request`, no `HTTPException`, no status codes. The routes translate.
- **Validation of field *names*.** The generated pydantic model does that at the boundary; this class takes a mapping and uses what it recognizes.
- **Loading a bundle from disk.** `artifacts.py` owns that, and `app.py` calls it.
- **The threshold or the calibrator.** They travel inside the bundle so the model and its decision rule cannot drift apart ([0007](../decisions/0007-the-scoring-bundle.md)); this file reads them.
- **Caching scores.** A memoized decision is a decision that survives a retrain, which is the opposite of what the run id in every response is for.

## Related tests

`tests/test_api.py` sections 1 and 2 cover this through the routes, which is where the interesting assertions land.

- `test_batch_matches_single_scoring` is the important one: the same applicant through `/predict` and `/predict/batch` gets the same probability, so the vectorized path is not a second implementation with its own rounding.
- `test_predict_accepts_both_extract_dialects` and `test_predict_accepts_a_source_alias` are the `object`-dtype policy paying off - `' 36 months'` and `36` both work, and so does a raw alias name.
- `test_predict_reasons_are_ordered_by_absolute_contribution` and `test_predict_without_explain_omits_reasons` pin the reason contract.
- `test_report_payloads_contain_no_nan_token` and `test_predict_scores_an_applicant` cover the serialization rules around `_reason`; `test_a_reason_value_is_json_safe_whatever_the_frame_held` covers the three conversions inside it directly. NaN is the one that matters: `json.dumps` writes a bare `NaN` token, which is not valid JSON, so a single missing value would take out the whole response rather than one cell.
- `test_a_bundle_that_cannot_explain_still_scores` is the degradation contract. A bundle with no persisted background scores normally, answers `explain=true` with an empty reason list rather than an error, and reports the reason on `/readyz` - so the condition is visible without being fatal.
- `tests/test_explain.py` holds the claim that the reason codes sum back to the model's own score, which is what makes them defensible rather than decorative.
- `tests/test_bench.py::test_scoring_stays_inside_the_regression_budget` is the latency tripwire; see [bench.md](bench.md).

## Known limits

- **The two-pass cost for reason codes is a bundle-format problem.** Calibrating the bare estimator rather than the pipeline would make one pass serve both, and it is the upgrade path if the budget ever binds. It changes what a bundle contains, so it is not made lightly.
- **Reason codes are per applicant and computed synchronously.** A 1000-row batch with `explain=true` is 1000 explanations in one request; the batch row cap is what bounds it.
- **No score caching and no request coalescing.** Two identical applicants cost twice, which is correct for an audited decision and wasteful for a load test.
- **`build_frame` copies.** One `pd.Series` per declared column per call. At one row that is microseconds; it is listed because it is the obvious thing to look at if the budget ever needs another millisecond.
- **The budget is stated for a logistic-regression bundle.** An XGBoost bundle's `TreeExplainer` path has a different cost curve, and the declared figures are not claimed for it.
