# 0007 - One scoring bundle per immutable run directory

Status: accepted.
Affects `src/risk_score/artifacts.py`, `src/risk_score/pipeline.py`, `src/risk_score/cli.py`, and everything Phase 6 serves.

## Context

The project wrote a model file that nothing could use.

`joblib.dump(model, "reports/models/logistic_regression.joblib")` produced a fitted estimator, and no code anywhere loaded it.
That is not just dead weight, because scoring an applicant needs four things and the file held one.

The **calibrator** was in a second pickle, so a probability read out of the model file was not the number the cost matrix had been applied to.
The **threshold** was a float inside `metrics.json`, so the model and the decision rule were two files that a caller had to pair correctly, and a mismatched pair is indistinguishable at serving time from a correct one.
The **feature contract** was nowhere at all, so a request payload could not be checked against the columns the model was actually fitted on: a missing field became a silent `NaN`, and an extra one became a silent nothing.

The write pattern was worse than the contents.
Every run overwrote the same paths - `metrics.json`, `models/*.joblib`, `figures/*.png` - one file at a time, in place.
Three consequences followed from that single choice.

**A reader could observe a run in progress.**
The dashboard fetched `metrics.json` and `calibration.csv` as separate requests, so a poll landing mid-run returned new metrics beside the previous run's calibration curve, and nothing in either file said so.

**A failed run left a corrupted tree.**
An exception after the metrics were written but before the figures were regenerated produced a `reports/` directory that looked complete and described two different models.

**There was no history and no rollback.**
The previous model was gone the moment a new run started, so "the numbers got worse, put it back" had no answer other than re-running an older commit against a dataset that might no longer exist.

The audit's most embarrassing artifact is the direct consequence: a committed `metrics.json` reporting `auc_roc: 0.0696` next to `ks_statistic: 0.9304`, on 116 test rows with one positive, rendered by the dashboard with exactly the same confidence as a valid run.
Nothing recorded how many rows the run saw, which extract it read, or which commit produced it, so there was no way to tell that artifact from a good one without reading the code that wrote it.

## Decision

**One `ScoringBundle` pickle per run, inside an immutable run directory that is published atomically, indexed by an append-only registry, with a separate pointer naming the run that serves.**

```
reports/
  registry.json        every run: identity, row counts, headline metrics
  active_run.json      {"run_id": ..., "activated_at": ...} - the only thing the service reads at boot
  runs/<run_id>/
    model.joblib       ONE ScoringBundle
    manifest.json      the same metadata as plain JSON
    metrics.json  calibration_*.csv  threshold_costs.csv  figures/  run.log
```

Six parts, each closing one of the failures above.

**1. `ScoringBundle` is the only object this project pickles.**
A frozen dataclass holding `pipeline`, `calibrator`, `threshold`, `feature_spec`, `metadata`, and `shap_background`.
The threshold lives with the model because it is part of the decision rule and was selected against *these* calibrated scores; any arrangement that permits loading the model without it permits serving a model with somebody else's cut-off.
`predict_probability` is the single scoring method and it always goes through the calibrator, so there is no supported way to get an uncalibrated score by accident.

**2. `run_id = <UTC basic ISO with milliseconds>-<model>-<tier>-<short sha>`.**
Sortable first, because the most common question about a run directory is which one is newest and a lexical sort answers it with no parsing.
Basic ISO because a colon is legal in a POSIX filename and fatal on Windows and in a URL, and this id appears in both.
Milliseconds rather than seconds is not future-proofing: with the parquet cache warm a synthetic run finishes in well under a second, so two runs of the same model and tier inside one second is ordinary, and publishing refuses to overwrite an existing id.

**3. The run directory is published by one `os.replace`.**
Everything is written into `runs/.staging-<uuid>/`, every file and every directory is fsynced, and then the scratch directory is moved into place.
The fsync is not superstition: without it `os.replace` can make the *name* durable while the contents are still in the page cache, so a crash leaves a run that exists and is empty.
A run directory is therefore either absent or complete, which is what makes a concurrent reader unable to pair new metrics with an old curve.
On any exception the scratch directory is removed, so a failed run publishes nothing and the previous run keeps serving.

**4. `registry.json` and `active_run.json` are updated together, under an `O_CREAT | O_EXCL` lock.**
One atomic syscall, no new dependency, no daemon.
Both files are written inside one lock hold, so a reader never sees an active pointer naming a run the registry does not list.
The wait timeout is 10 seconds and the stale-lock timeout is 60, and that ordering is the design rather than two arbitrary numbers: were they equal, a lock held legitimately for slightly too long would be *stolen* instead of waited out and the timeout error would be unreachable.
A stale timeout has to exist at all because the process that would have released the lock is the one that died, and `SIGKILL` leaves no chance to clean up.

**5. The registry is a cache; the manifests are the truth.**
A corrupt or absent `registry.json` reads as empty rather than raising, because the run directories are the data and refusing to list them over one truncated index turns a cosmetic problem into an outage.
`rebuild_registry` reconstructs the index from the manifests on disk, which also handles runs copied in from another machine.
`RunMetadata.from_dict` ignores unknown keys, so a manifest written by a newer build still lists in `riskscore runs`; the pickle is where a version mismatch must be fatal, and `load_bundle` checks `bundle_schema_version` *before* touching the object so the error names the two versions rather than surfacing as a missing attribute inside a request handler.

**6. Activation is a separate, reversible step.**
`train --no-activate` publishes and registers without repointing serving, which is what a comparison run needs.
`riskscore activate <run_id>` is the rollback path, and it loads the bundle and throws it away before repointing - deliberately, because activating a run whose pickle is unreadable would move the failure to the service's next boot, where it is an outage instead of a message on somebody's terminal.
`prune_runs` never deletes the active run regardless of age, because the alternative is retention policy taking the service down.

The manifest carries what the old tree had nowhere to put: dataset path, SHA-256 prefix and byte count, git commit, library versions, the target definition in words, the split windows, the embargo rule and the rows it removed, the cost matrix, the feature tier and leakage summary, and **row counts at every filter stage** - `raw`, `closed`, `mature`, `in_scope_terms`, `labelled`, `train`, `validation`, `test`.
That last one is what makes the 116-row artifact self-describing instead of merely wrong.

## Consequences

**Every number the project reports is traceable to an extract digest, a commit, and a row count.**
This is the property the whole phase exists for, and it is what lets the model card, the run history, and the dashboard header be generated rather than written.

**Pickle couples every bundle to the module paths inside it.**
`risk_score.transformers` cannot be renamed or moved without breaking every bundle ever written, and it breaks at import time, silently, in whatever process loads it.
This is the standing cost of putting feature engineering inside the pipeline (ADR 0002) and the mitigation is a rule rather than code: these module paths do not move.
`BUNDLE_SCHEMA_VERSION` covers field changes, not module moves.

**A bundle is only loadable by code that can import this package.**
So `manifest.json` duplicates the metadata as plain JSON on purpose: identity questions are answerable by `cat`, from any language, without unpickling and therefore without executing anything the file chose.

**Loading is trusting.**
`joblib.load` executes pickle opcodes, so a bundle is as trusted as the code that wrote it.
The service loads only what `active_run.json` names inside its own reports root, uploads never become bundles, and this is the reason `/artifacts/` serves an explicit filename allowlist instead of anything under a run directory.

**Disk grows one directory per run.**
A few hundred kilobytes plus figures each, capped by `prune_runs` at 20 by default, and `reports/` is gitignored.
The cap is about keeping the tree readable rather than about disk.

**The lock is correct on a local filesystem and not on NFS.**
`O_CREAT | O_EXCL` over NFS is not reliably atomic.
Documented rather than papered over, because the alternative is a lock service this project has no use for.

**Two extra writes per run.**
An fsync of the whole staging tree and two small JSON files under a lock, which is milliseconds against a fit measured in seconds.

## Alternatives considered

**Keep overwriting `reports/`, and just add the metadata.**
Metadata makes a corrupt tree self-describing without making it correct.
The mid-run read, the half-failed run, and the absent rollback are all properties of overwriting in place, and no amount of recorded provenance addresses any of them.

**MLflow, or DVC, or any tracking server.**
The right answer for a team running many experiments, and the wrong one here.
It adds a service, a database, and a client library to a project whose entire artifact story is "a directory you can `ls`", and it moves the run history somewhere a reader cannot inspect with `cat`.
The registry is 40 lines and the file layout is the documentation.

**Three files instead of one - model, calibrator, threshold - kept in a directory and loaded together.**
This is what the project had, minus the overwriting.
Rejected because "loaded together" is a convention, and a convention that must hold across a CLI, a service, a benchmark, and a test suite will not.
One object cannot be assembled wrongly.

**ONNX or PMML instead of pickle, to escape the module-path coupling.**
Genuinely attractive: a portable artifact with no code execution on load.
Rejected because the pipeline's first step is a custom transformer full of Lending Club parsing rules (`' 36 months'`, `'10+ years'`, `'Aug-2003'`), and exporting that means either reimplementing it in the target runtime or dropping it from the artifact - which reintroduces the exact train/serve skew ADR 0002 exists to remove.
Revisit when the preprocessing is expressible in the standard operator set, not before.

**Copy-on-write publication - write into the real run directory and mark it complete with a sentinel file.**
Cheaper than staging plus fsync, and it makes completeness a fact a reader has to remember to check.
Every reader that forgets sees a half-written run under its real name, which is the failure being designed out.
`os.replace` makes the check unnecessary rather than optional.

**A symlink for the active run instead of a JSON pointer.**
`ln -sfn` is not atomic (it unlinks, then links) and symlinks behave differently in Docker images, in tarballs, and on Windows.
A JSON file replaced atomically also carries `activated_at`, which a symlink cannot.

**Timestamp-only run ids, with the model and tier only in the manifest.**
Shorter ids, but `ls reports/runs` stops answering the two questions actually asked of it - which model, which feature tier - and the id is the thing that appears in log lines, in the dashboard header, and in every model card.
The cost is a 66-character directory name and one CLI column that measures its own width.
