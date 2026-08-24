# 0002 - Feature engineering lives inside the scikit-learn Pipeline

Status: accepted.
Affects `src/risk_score/transformers.py`, `src/risk_score/modeling.py`, `src/risk_score/pipeline.py`.

## Context

Preprocessing used to happen in two places that had no way of agreeing with each other.

`build_feature_matrix` ran in `pipeline.py`, before the model existed: it parsed percent strings, derived `credit_utilization` and `credit_history_months`, and returned a frame.
`build_preprocessor` then decided what to do with each column by asking pandas what dtype it had ended up as:

```python
numeric_columns = features.select_dtypes(include=["number", "bool"]).columns
categorical_columns = features.select_dtypes(include=["object", "category"]).columns
```

That definition of "categorical" is *whatever failed to parse*.
`build_feature_matrix` added cleaned columns without dropping the raw ones, so `earliest_cr_line` - 655 distinct date strings in the real extract - went to `OneHotEncoder`, as did `int_rate` and `revol_util` in the extract where they arrive as percent strings, with roughly 600 and 1100 levels.
With `sparse_output=False` that is about 2400 dense float columns over 1.8M rows: roughly 35 GB.

The pipeline was not slow on the real dataset.
It could not run on it, and it destroyed the monotone rate information on the way (audit B01, B22, B23).

The second half of the problem only appears when you try to serve the model.
`joblib.dump` wrote the pipeline; nothing loaded it, and no `predict` function existed anywhere.
To score one applicant, something would have had to re-derive `credit_utilization` from a JSON body - which means a second implementation of every parse rule, in a different file, with no test comparing the two.
The failure mode of that arrangement is not an exception; it is a served probability that is quietly wrong because the request path standardized `revol_util` slightly differently from the fit path.

## Decision

Move all of it inside the `Pipeline`, as two stateless steps ahead of the fitted ones, and build the `ColumnTransformer` from **declared** column lists.

```
CanonicalizeFrame -> EngineerFeatures -> ColumnTransformer -> estimator
|------------- stateless -------------| |----- fitted on train only -----|
```

- `CanonicalizeFrame(spec)` resolves aliases, reindexes to `spec.raw_inputs`, adds absent optionals as all-NA, raises naming any missing *required* column, and casts each column to its declared dtype. Every messy parser lives here: `'13.56%'`, `' 36 months'`, `'10+ years'`, `'Aug-2003'`.
- `EngineerFeatures(spec)` calls the pure Series builders in `feature_engineering.py` and **drops the raw sources** its outputs replace, so the frame reaching the encoder is exactly `spec.model_features`.
- `build_preprocessor(spec)` takes a `FeatureSpec`, never a DataFrame. It cannot see the data, so the data cannot mislead it.
- `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=True)`.

The important property is not that the bug is fixed.
It is that a string column reaching `OneHotEncoder` unintentionally is no longer a state the code can represent: the numeric list is declared by name, and a column arriving as `object` at inference is coerced by declaration or raises with its own name in the message.

Nothing to the left of the bar is fitted - no means, no medians, no category vocabularies - which is what makes it safe to run those steps on any partition without carrying information across a split boundary.

## Consequences

**One code path, so there is nothing to keep in sync.**
Training and `POST /predict` share the same four steps from the same pickle.
`test_one_row_scores_identically_alone_and_in_the_batch` is the guarantee: row 5 transformed on its own must equal row 5 of the batch, which is the property that breaks first if anything in the frame steps becomes sample-dependent.

**SHAP gets real feature names for free.**
`get_feature_names_out` flows through the whole pipeline, so a reason code says `credit_utilization` rather than `column 7`.
With `verbose_feature_names_out=False` it says `loan_amnt` rather than `numeric__loan_amnt`.

**The pickle is coupled to this module path forever.**
A pickled bundle stores the import path of every object inside it, so renaming `transformers.py`, `CanonicalizeFrame`, or `EngineerFeatures` breaks every bundle ever written.
This is the real cost of the decision and it is not mitigable, only managed: `SPEC_VERSION` covers changes to the *meaning* of a spec field, a documented never-move rule covers the path, and the load path checks the version.

**XGBoost early stopping cannot go through `Pipeline.fit`.**
`eval_set` would arrive as raw string frames.
So `fit_with_validation_monitoring` fits the preprocessing prefix explicitly, `transform`s validation through it, fits the estimator with `eval_set=`, and reassembles a `Pipeline` from the already-fitted steps - relying on sklearn not cloning the steps it is handed.
That reliance is verified by asserting the assembled pipeline's predictions equal the manual two-step path's exactly.
It is the one place the design costs real complexity, which is why it is one small function with three tests rather than an inline block.

**mypy strict needs help at the `BaseEstimator` boundary.**
sklearn's base classes are untyped in places, so the transformers carry a few targeted annotations and one `fitted_` attribute that exists only because `check_is_fitted` treats a step with no trailing-underscore attribute as unfitted.

**Feature engineering is no longer inspectable as an intermediate artifact.**
Nothing writes the engineered frame to disk any more.
Debugging goes through `pipeline[:-1].transform(frame)`, which is a documented recipe in the runbook rather than a file on disk.

## Alternatives considered

**Keep the two-phase design and fix `select_dtypes` to check dtypes properly.**
The smallest possible diff, and it fixes the 35 GB symptom.
Rejected because it leaves the class of bug intact - the routing is still inferred, so the next unparsed column is the next incident - and it does nothing at all about serving, which was the larger gap.

**Keep the two-phase design and write a matching `preprocess_request` for the API.**
This is what most projects do.
It means every parse rule exists twice, and the tests that would catch a divergence are exactly the tests nobody writes.
The failure is silent and it is in production.

**A separate feature store or a `FunctionTransformer` chain.**
`FunctionTransformer` cannot express the required parts - naming the missing column, dropping replaced sources, holding a spec - without becoming a class anyway.
A feature store is a reasonable answer at a scale this project is not at, and it would reintroduce the two-code-path problem across a network boundary.

**Declare the columns, but keep the transformers outside the pipeline and apply them by hand at both ends.**
Gets the routing fix and avoids the pickle coupling.
Rejected because "apply them by hand at both ends" is the two-code-path problem with extra discipline required, and discipline is not a mechanism.
