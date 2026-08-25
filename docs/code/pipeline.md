# `src/risk_score/pipeline.py`

## Purpose

Order the steps, and hold no logic of its own.

Everything this file does happens elsewhere: the reading in `data_loading.py`, the parsing in `transformers.py`, the fitting in `modeling.py`, the metrics in `evaluation.py`, the correction in `calibration.py`, the writing in `artifacts.py`.
What is here is the **sequence**, and the sequence is where the project's four hardest bugs lived.
None of them were in a formula.

**The threshold was selected on the test set.**
A threshold is a fitted parameter.
Choosing it on the same rows the headline metrics come from turns those metrics into a description of the selection procedure (audit B04).

**Calibration was measured and then thrown away.**
The old run drew a reliability curve, computed a Brier score from uncorrected probabilities, and dropped the correction on the floor - so the reported Brier described a model nobody would have shipped (audit B05).

**Immature loans were never removed.**
Filtering to closed statuses looks obviously right and is quietly wrong: a 36-month loan issued in 2016 has only closed by a 2018 snapshot if it defaulted *early*, while the ones still paying read as `Current` and get dropped.
The measured default rate by vintage therefore climbed 15.6% → 18.5% → 20.2% → 24.3% and then fell to 14.7%, a shape driven entirely by the snapshot date, and the time split reports on exactly the most contaminated vintages.

**Feature engineering happened here.**
So the persisted artifact could not preprocess a request the way the model was fitted, which is a class of bug that only appears in production.

Two more properties were added because the old layout could not support them: a run is one atomic directory rather than a tree of separately-overwritten files, and every number is traceable to a dataset digest, a commit, and a row count at each filter stage.

## Public API

| Name | What it is |
| --- | --- |
| `train_run` | The whole thing. One extract in, one published `RunResult` out. |
| `RunResult` | `run_id`, `run_dir`, `metrics`, `metadata`, `bundle`, and the full `metrics.json` payload. |
| `target_definition` | The label rule in words, built from the status constants so it cannot drift from the label used. |
| `METRICS_FILENAME`, `CALIBRATION_TEST_FILENAME`, `CALIBRATION_VALIDATION_FILENAME`, `THRESHOLD_COSTS_FILENAME`, `CALIBRATION_FIGURE`, `RUN_LOG_FILENAME` | The artifact contract: the API serves these from an allowlist and the dashboard fetches them by name, so there is one spelling of each. |

`train_run` takes the extract path and five keyword arguments: `config`, `output_dir`, `model_type`, `make_active`, `keep_runs`, `cache_dir`.
Everything a run *varies* - windows, cost matrix, hyperparameters, feature tier, column aliases - is inside the one `RunConfig`, so there is no second place a window can be specified and disagree.

## Inputs and outputs

Reads one CSV.
Writes one run directory, then appends to `registry.json` and optionally repoints `active_run.json`.

The order is the documentation:

```
raw -> closed statuses -> maturity embargo -> term filter -> label
    -> leakage audit -> feature spec -> tri-split
  TRAIN      fit the pipeline. Nothing else.
  VALIDATION fit the calibrator, then select the decision threshold.
  TEST       score once, report, never fit anything.
```

Every row filter sits **left of the split**, so all three partitions share one outcome definition.
An embargo applied per-partition would give train and test different labels for the same loan.

`make_active=False` publishes and registers without repointing serving, which is what a comparison run needs: it fits two models and only one should be served.
`cache_dir=None` (the default) skips the parquet cache; the CLI passes a directory.

## Invariants and failure modes

**Nothing is created before the arguments are checked.**
An unsupported `model_type` and a missing extract both raise before `output_dir` exists, so a typo leaves no empty report tree behind.

**A failed run publishes nothing at all.**
The fit happens inside `staged_run`, so an exception anywhere between reading and writing removes the scratch directory and leaves the registry and the active pointer untouched.
The previous run stays served.

**The run log is published by the same rename as the metrics it explains.**
`capture_run_log` opens the file *inside* the staging directory.
The context managers are entered left to right, so `staging` is already bound when the log handler needs it, and a failed run discards its log along with the artifacts the log describes.

**One instant, two spellings.**
`datetime.now(UTC)` is read once and formatted twice: basic ISO in the run id, because a colon is legal in a POSIX filename and fatal on Windows and in a URL, and extended ISO in the manifest, because the registry sorts runs by that string.
Reading the clock twice would let the two disagree.

**The threshold row is located by nearest value, not by float equality.**
`threshold_costs["threshold"] == threshold` happens to hold today because both come from the same `np.arange`, and would stop holding the moment a caller passes its own grid - raising `IndexError` on `.iloc[0]` of an empty selection (audit B11).

**The threshold is chosen on calibrated scores.**
That is what serving compares against, and a threshold picked on uncalibrated scores then applied to calibrated ones is a different policy than the one that was costed.

**The uncalibrated Brier score is reported beside the calibrated one.**
Otherwise the calibration step is unfalsifiable: there is no number that could show it made things worse.

**Approval rate is measured on test, not read out of the validation cost table.**
The rate a lender actually sees is a property of the population being scored, not of the population the rule was tuned on.

**Row counts are recorded at every stage.**
`raw`, `closed`, `mature`, `in_scope_terms`, `labelled`, `train`, `validation`, `test`.
A run that discarded 40% of its input must not look identical to one that discarded none.

**The per-vintage default rate is recorded before *and* after the embargo.**
That pair is the single most defensible thing this project reports, and reporting only the corrected number would make the correction invisible.

## What must NOT live here

- **Feature engineering.** It belongs inside the fitted `Pipeline`, so the artifact is self-contained and `POST /predict` cannot preprocess a request differently from how the model was fit (ADR 0002).
- **Metric definitions.** `evaluation.py`.
- **Any fit on test data.** Structurally prevented: `fit_calibrator` and `select_threshold_by_cost` accept only a partition-tagged wrapper, so passing test scores is both a type error and a runtime error.
- **File-layout decisions.** `artifacts.py` owns the tree; this module names the files it writes into it.
- **Comparison, drift, and the model card.** They read published runs rather than participating in one, so they can be re-run over history without refitting.
- **`print`.** Everything is logged, under a run id, and captured into the run's own log.

## Related tests

`tests/test_pipeline.py`, 27 tests, all against synthetic extracts from `sample_data.py`.
This is the suite that exercises the whole ordering, so it is where the audit regressions for the sequence live.

- `test_b04_the_threshold_is_selected_on_validation_not_on_test` monkeypatches the selector and asserts the test arrays never appear in its arguments - a value check, not a call-count check, so passing test scores in the right position would still fail.
- `test_b05_the_calibration_correction_is_applied_and_not_merely_drawn` recomputes the Brier score from the bare pipeline and shows it differs from the reported one.
- `test_b01_the_persisted_model_never_saw_a_post_origination_column` reads the column list back out of the pickle.
- `test_the_maturity_embargo_runs_and_flattens_the_vintage_default_rate` and `test_immature_loans_never_reach_any_partition` are the survivorship-bias pair.
- `test_a_failed_fit_publishes_nothing_at_all` monkeypatches `train_model` to raise and asserts the tree is untouched.
- `test_the_pipeline_writes_every_documented_artifact` is the artifact contract, file by file, and is the reason the filenames are constants.
- `test_the_run_log_is_published_inside_the_run_it_describes` checks that the log carries the run id and the embargo line.
- `test_a_second_run_supersedes_the_first_without_overwriting_it` is immutability from the caller's side.
- `test_the_persisted_model_scores_a_single_raw_applicant` is the closest thing here to a serving test: one raw row, straight from the pickle.
- `test_a_second_run_reuses_the_cached_extract_instead_of_reparsing_it` proves the cache is wired in, using each run's own published log as the evidence.

## Known limits

- **The validation cost table is mildly optimistic,** because the calibrator was fitted on those same rows. Reported anyway rather than splitting a fourth partition off a dataset this size; the honest number is the test one, which the calibrator never saw.
- **The default split windows are narrow** - train 2013-01 to 2014-09, validation to 2015-03, test to 2015-12 - because the embargo against a 2018-12 snapshot leaves nothing usable later. That is the snapshot's fault, not the split's, and the two are coupled: change the snapshot and the windows have to move with it.
- **`term_months_in: [36]` by default.** Post-embargo, 60-month loans are 28% of 2013 rows and 0% of 2015 rows, so admitting them makes the train and test populations structurally different. An empty `terms` admits all of them and the term-mix cliff gets reported.
- **One model per run.** `compare` publishes two runs rather than one run with two models, because a bundle holds one estimator and a run holds one bundle.
- **No resume.** A run that fails at the last write refits from scratch. The parquet cache removes the expensive part of that, which is the parse.
- **Single process, single machine.** No distributed training, no out-of-core fit. The sparse one-hot path is what keeps the real extract inside memory.
