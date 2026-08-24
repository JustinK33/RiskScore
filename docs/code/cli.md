# `src/risk_score/cli.py`

## Purpose

Be the supported way to drive the library, and get out of the way.

Every subcommand is a thin adapter over one library call.
Nothing is computed here that a caller importing `risk_score` could not compute, and nothing here is imported by the library - so the package stays usable as a library, and this file stays deletable.

It replaces two `scripts/` files that were removed, and the replacement is not cosmetic.
`scripts/run_baseline.py` wrapped every failure in `raise SystemExit(str(error)) from None`, which discards the traceback *and* the `__cause__`: a `KeyError` deep inside a transformer arrived as one unattributable line, and debugging it meant re-running the failure by hand in a REPL.
`scripts/serve_dashboard.py` retrained the model on unauthenticated uploaded CSV, on the request thread, overwriting the canonical `reports/` tree including the pickle.

## Public API

| Command | What it does |
| --- | --- |
| `riskscore train <csv>` | Fit one model and publish it as a run. |
| `riskscore runs` | List published runs, newest first, with the active one marked. |
| `riskscore activate <run_id>` | Point serving at a different published run. The rollback path. |
| `riskscore make-sample-data [path]` | Write a synthetic Lending Club-shaped extract. |

| Name | What it is |
| --- | --- |
| `main` | Parse and dispatch. Returns the exit code rather than calling `sys.exit`. |
| `build_parser` | The whole command surface, so `--help` is the documentation. |
| `EXIT_USER_ERROR` | `3`. |
| `TRAIN_SUMMARY_KEYS` | What `train` prints, in order. |

Notable flags: `--model`, `--config`, `--include-lender-priced`, `--no-activate`, `--keep`, `--cache-dir`, `--no-cache`, `--output-dir`, `-v`, and `runs --json` / `runs --rebuild`.

## Inputs and outputs

Reads an extract and a run config; writes a run tree.

**The summary goes to stdout and the log goes to stderr.**
So `riskscore runs --json | jq` works, and so does watching `train`'s progress while capturing its result.

```
$ riskscore train data/sample/loans.csv
run 20260824T183320563Z-logistic_regression-origination_only-02e3a94
  reports/runs/20260824T183320563Z-logistic_regression-origination_only-02e3a94
  rows                       raw=2000  closed=1551  mature=1193  in_scope_terms=1072  labelled=1072  train=629  validation=193  test=250
  AUC (test)                 0.6511
  average precision (test)   0.2115
  KS (test)                  0.2896
  Brier (calibrated)         0.1091
  Brier (uncalibrated)       0.1119
  ECE (test)                 0.0498
  threshold (validation)     0.1400
  approval rate (test)       0.7440
  default rate (test)        0.1280
  calibration                sigmoid on validation
```

Those are the real numbers from `make-sample-data --rows 2000`, which is a deliberately small extract: 250 test rows.
The row counts lead, before any metric, because a 0.65 AUC on 250 test rows and a 0.65 AUC on 250,000 are not the same claim and the metric alone does not say which one you are looking at.
Both Brier scores are printed together, so the calibration step is falsifiable from the terminal.

`riskscore runs` marks the active run with `*` and computes its column width from the data - a run id is a timestamp plus a model name plus a tier plus a sha, and `with_lender_priced` is twelve characters longer than `origination_only`, so a hardcoded width misaligns the moment somebody trains the other tier.

## Invariants and failure modes

**Tracebacks propagate.**
The only exceptions caught are the two that are genuinely *user* errors rather than bugs: a missing file, and a config that does not validate.
Anything else - a `KeyError` in a transformer, a shape mismatch, an `InvalidParameterError` from sklearn - reaches the terminal with its frames intact.

**Exit codes are meaningful.**
`0` success, `1` a bug (via the propagated traceback), `2` bad usage from argparse, `3` a user error this module recognized.
A caller can therefore tell "you typed it wrong" from "it is broken", which a single non-zero code cannot express.

**A successful run is never turned into a failure by the summary.**
The metric lines are read with `.get` and rendered through the same helper the run listing uses, so a metric this version of the pipeline no longer emits prints `-` rather than raising `KeyError` *after* the run was already published.
This is not hypothetical: the first version of `_train` indexed `payload["threshold"]`, the key is `selected_threshold`, and the result was a published run and a traceback.

**A missing metric prints `-`, never `0.000`.**
An absent number rendered as zero reads as a catastrophic model rather than as missing data.

**`activate` loads the bundle before repointing.**
Deliberately loaded and thrown away: activating a run whose pickle is unreadable would move the failure to the service's next boot, where it is an outage instead of a message on somebody's terminal.
A failed activation leaves serving exactly where it was.

**The cache defaults are opposite to the library's.**
On here, off there.
A command line writing under `data/cache` is expected; a library function doing it unasked is a surprise.

**`RunConfig` is copied, not mutated.**
`--include-lender-priced` produces a new frozen config, because the whole point of freezing it is that the configuration a run *reports* is the configuration it *used*.

**Only implemented commands exist.**
`compare`, `explain`, `card`, `serve`, and `bench` arrive with the modules that back them.
A subcommand that exists and fails is worse than one that does not, because `--help` stops being a reliable answer to what the tool can do.

## What must NOT live here

- **Any modelling, metric, or IO logic.** If a subcommand needs more than argument marshalling and printing, the missing piece belongs in a module and the CLI calls it. The test for this is whether the library can do everything the CLI can.
- **Being imported by the library.** The dependency runs one way. `pipeline.py` importing `cli.py` would make the training path depend on argparse.
- **`configure_logging` at import time.** It is called inside `main`, after parsing, so importing this module does not reconfigure a host process's logging.
- **Interactive prompts.** Every command has to be runnable from CI and from a `Dockerfile`. Anything destructive takes an explicit flag instead of a confirmation.
- **Its own defaults for anything the library already defaults.** Two default values for one setting is one place for them to disagree; the exception is `--cache-dir`, which is a deliberate, documented policy difference.

## Related tests

`tests/test_cli.py`, 25 tests, driven through `main(argv)` rather than through a subprocess: the parsing, the dispatch, and the exit code are the surface worth testing, and a subprocess would add an interpreter start per case for no extra coverage.
The one thing that genuinely needs the installed entry point - that `riskscore` resolves at all - is asserted against `pyproject.toml` instead.

Most of the suite is about behaviour under bad input, because that is where the old entry points were wrong.

- `test_a_bug_keeps_its_traceback` is the whole reason this file exists: `train_run` is monkeypatched to raise `KeyError`, and the test asserts the `KeyError` reaches the caller.
- `test_a_missing_extract_is_reported_without_a_traceback` and `test_a_broken_config_is_reported_without_a_traceback` are the other side: exit 3, no frames, and no partially created report tree.
- `test_train_publishes_a_run_and_reports_it_on_stdout` asserts every declared summary line rendered a *number* rather than the `-` that stands for a metric the pipeline no longer emits, which is what would have caught the `payload["threshold"]` mistake.
- `test_the_cache_is_on_by_default_and_off_on_request` pins the deliberate default asymmetry.
- `test_activate_refuses_a_run_whose_bundle_cannot_be_loaded` overwrites a pickle with `b"not a pickle"` and asserts serving did not move.
- `test_runs_lists_newest_first_and_marks_the_active_one` uses a tree whose active run is the *older* one, so the marker cannot be confused with "first row".
- `test_every_subcommand_is_reachable_and_documented` reads the rendered help text rather than argparse's private `_subparsers`, so it breaks when the output regresses and not when argparse rearranges its internals. It is what caught `runs` having no description.
- `test_make_sample_data_is_deterministic` is what lets the demo, the docs, and CI all quote the same numbers.

## Known limits

- **argparse, not click or typer.** No new dependency for four subcommands, and `--help` is adequate. The cost is manual `Namespace` handling and `Any`-typed handlers.
- **The handlers are typed `Any` through the `Namespace`.** argparse cannot express the shape of its own result, so mypy strict cannot check that `_train` reads flags `_add_train` defined. A typo there is a runtime `AttributeError`, which is why every flag appears in a test.
- **No shell completion.** Generated completions need a wrapper or a plugin, and the command list is four items long.
- **No progress bar on a long fit.** The log line per stage is the progress indicator, which is also what ends up in `run.log` and what CI can read.
- **`--config` is the only way to change windows or costs.** No `--train-end-date` style flags, on purpose: those were how the old script let a caller specify a window that disagreed with the config file.
- **Exit code 1 is a traceback rather than a message.** For a tool driven by its author and by CI that is the right trade; a tool driven by a scheduler that only logs the last line would want the opposite.
