# `src/risk_score/config.py`

## Purpose

Turn `configs/run.yaml` into typed, validated arguments - and reject anything it does not recognize.

There were three config files before this one, and they made the project worse than having no configuration at all.

**They duplicated the column registry, and the copies disagreed.**
`configs/feature_config.yaml` listed `grade` and `sub_grade` as ordinary features - they are lender-priced, and off by default now - and listed `fico_band` as engineered from `fico_range_*` columns that are absent from **both** real extracts.
None of that was noticed because nothing loaded the file.
A config that is not read cannot be wrong, and cannot be right either; it is documentation that looks like code.

**Nothing validated them.**
`load_yaml_config` returned a bare dict and callers reached into it with `.get`, so a misspelled key was not an error - it was a silently applied default (audit B30).
`class_wieght: balanced` trains a different model than the one you asked for and tells you nothing.
This is the failure mode that config files are uniquely good at: the typo is in data, so no linter, no type checker, and no test sees it.

**They had gone stale enough to break the run.**
`configs/dataset_schema.yaml` still declared `n` as an alias for `loan_status`.
Phase 1 added a minimum alias length precisely because a one-letter alias silently hijacks the target - so the shipped config *raised on load*, and every `scripts/run_baseline.py` invocation died on it.
Nothing tested the config files, because there was nothing to test: they were dicts.

So the three files are gone, replaced by one that is parsed into frozen dataclasses.
What remains configurable is what a run legitimately varies: split windows, model hyperparameters, the cost matrix, the feature tier, and an alias escape hatch for onboarding an unfamiliar extract without a code change.

## Public API

| Name | What it does |
| --- | --- |
| `RunConfig` | Everything one training run needs that is not a property of the data. `RunConfig()` is a complete, tested configuration. |
| `RunConfig.from_mapping(mapping)` | Parse a loaded YAML document, rejecting unrecognized keys. |
| `RunConfig.model_params(model_type)` | Hyperparameter overrides for one model, as a fresh dict. |
| `SplitConfig` | `date_column` plus the three `(start, end)` windows. |
| `SplitConfig.from_mapping(mapping)` | Same contract, for the `split` section. |
| `load_run_config(path)` | Read and validate a config file. This is what callers use. |
| `load_yaml_config(path)` | Read a YAML document as a mapping. No interpretation, no defaults. |
| `DEFAULT_SPLIT_WINDOWS` | The shipped windows, so the YAML only has to state what it changes. |

## Inputs and outputs

`load_run_config` takes a path and returns a `RunConfig`.
`run_baseline_pipeline` takes that object and nothing else configuration-shaped, so there is no second place a window or a cost can be specified and disagree.

Both dataclasses are `frozen=True, slots=True`: a run cannot mutate its own configuration halfway through and produce a manifest that describes a different run than the one that happened.

Windows stay **strings** all the way through this module.
Expanding `2014-12` to the last instant of December is `TimeWindow`'s job, and two places doing it differently is how a month goes missing from a split.

`model_params` returns `dict(...)` - a copy.
It is handed straight to an estimator constructor, and a caller mutating it must not edit the config that every later run reads.

## Invariants and failure modes

### Unknown keys raise, and the message lists the valid ones

`_reject_unknown_keys` runs on the document and on every section: `split`, `threshold`, `leakage`.
`spilt:` at the top level fails with `Unknown key(s) in 'config': ['spilt']. Valid keys: [...]`.

The valid-key list is in the message on purpose.
"Unknown key" alone sends the reader to the source to find out what they should have typed, and at that point the error has cost more than the typo.

### Hyperparameter names are deliberately *not* validated

`models.logistic_regression` accepts any keys.
They belong to sklearn and XGBoost, both of which reject an unknown parameter themselves and name it - and enumerating them here would go stale on every library upgrade, which is exactly the failure the rest of this file exists to prevent.

The boundary is: **this module validates its own vocabulary, and forwards someone else's.**

### `safe_load`, never `load`

A config file is input.
`!!python/object/apply:os.system ['echo pwned']` in one is arbitrary code execution, and there is a test that asserts it is refused rather than constructed.

### A window must be a two-item list

`train: '2013-01'` raises `TypeError` rather than iterating into characters, and `train: ['2013-01', '2013-06', '2013-12']` raises `ValueError`.
The string check comes first because a string *is* iterable, so a length check alone would accept `'2013-01'` as a seven-bound window and silently produce nonsense.

### Missing is not the same as wrong

An empty config file - or a file containing only comments - equals `RunConfig()`.
A section that sets one key keeps the defaults for the rest.
A run should not need a config file to be a correct run, and there is a test for each of those three statements.

### A single alias may be written as a bare string

`annual_inc: yearlyIncome` normalizes to `("yearlyIncome",)`, because everyone writes it that way at least once and failing on it teaches nothing.
This is the one place the module is lenient, and it is lenient about *shape*, never about *meaning*.

### The shipped file and the in-code defaults are tested against each other

`test_the_shipped_config_matches_the_in_code_defaults` compares `configs/run.yaml` field by field with `RunConfig()`.
Two sources of one default is one source too many; that test is what keeps them from drifting.

`models` is exempt: the YAML lists each model with an empty override block so a reader can see where hyperparameters go, and the test asserts every such block is empty rather than asserting the mapping equals `{}`.

## What must NOT live here

- **Column names, aliases, dtypes, parse rules, or feature tiers.** Those live in `features.py`, which is the single source of truth. This is the boundary whose violation produced the three deleted files, and `configs/run.yaml` carries a header comment saying so.
- **Anything derived from data.** Which columns an extract actually supplies is resolved by `build_feature_spec` at runtime, against the columns in front of it. A config file cannot know.
- **Date parsing or period expansion.** `schema.py` owns formats; `TimeWindow` owns expansion.
- **Secrets.** API keys, upload toggles, and bind addresses are environment variables read by the service's settings object, never fields in a tracked YAML file.
- **Defaults that are really constants.** `min_category_frequency`, the boosting budget, and the early-stopping patience are keyword arguments with documented defaults in `modeling.py`. Promoting a constant to a config key invites tuning it without measuring it.

## Related tests

`tests/test_config.py`.

Named audit regressions: `test_b30_an_unknown_top_level_key_is_rejected`, `test_b30_an_unknown_split_key_is_rejected`, `test_b30_an_unknown_threshold_key_is_rejected`.

The two worth reading for intent are `test_the_error_lists_the_keys_that_would_have_worked` - the message is part of the contract - and `test_load_yaml_config_refuses_to_construct_arbitrary_objects`.

`test_the_shipped_config_loads` exists because the shipped config *stopped* loading once a minimum alias length was enforced, and nothing noticed, because no test read it.

## Known limits

- **Values are coerced, not range-checked.** `false_negative_cost: -5` is accepted here; `CostMatrix` is where a nonsense cost would have to be refused.
- **Window ordering is not checked at load time.** A config whose test window precedes its train window passes validation and fails in `split_by_time`. The error there is clearer - it can name the actual dates and the overlap - so the check stays where the information is.
- **`date_column` is a free string.** Naming a column that does not exist is caught by `split_by_time` with a `KeyError` listing what was present, not here.
- **No environment-variable overlay and no config inheritance.** One file, read once. Both are easy to add and neither has been needed; a run that wants different windows copies the file and passes `--config`.
- **`column_aliases` is an escape hatch, and escape hatches drift.** An alias that belongs to every extract belongs in `features.py`. Nothing enforces that distinction, which is why the key is documented as onboarding-only.
