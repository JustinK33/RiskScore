# `src/risk_score/reporting.py`

## Purpose

Turn a run's measurements into the three things a person reads: the model card, the variant comparison, and the payload shapes the dashboard fetches.

All three live in one file because they answer one question - *what does a human see* - and because all three are held to the same rule: **nothing here computes a number**.
Every value on a model card is read out of `metrics.json` or `manifest.json`.
If a figure is not in one of those files it does not belong on the card, because a card that does its own arithmetic is a second implementation of the run, and the second implementation is the one nobody tests.

### Why the card is generated in code and not filled into a template

A template plus a substitution pass has one failure mode, and it is the worst one available to a document whose entire purpose is to be trusted: a placeholder nobody filled ships as `$auc_roc` inside a compliance artifact.

The plan's own acceptance check for this feature was *"no `$placeholder` survives in the card"*.
That check only needs to exist because templates leak.
Building the markdown from typed accessors makes the failure **impossible** rather than **detectable**: a missing key becomes a `-` placed deliberately by `format_metric`, and a missing *section* is a diff in this file.

The static prose - intended use, out of scope, ethical considerations, the standing limitations - is here as module constants.
It is genuinely static: it describes the modelling decisions, not one run's numbers.
A per-run copy of it would drift between runs of the same code, which is how model cards end up describing a model that no longer exists.

### Why the comparison is quantified rather than asserted

"Excluding the lender's own price costs a little accuracy" is the kind of claim that gets repeated until it is believed.
`comparison_payload` reports the AUC difference between the same model fitted under both tiers, so [ADR 0005](../decisions/0005-lender-priced-feature-tier.md) is defended with a number a reader can disagree with.

The same two functions do the LR-versus-XGBoost comparison, because "several variants of one split, one row each" is the same table either way.

### Where `comparison.json` lives

At the **report root**, beside `registry.json` - not in `runs/<run_id>/`.

A run directory is published by one atomic rename and is immutable afterwards, and a comparison is only complete once every variant has been published.
Writing it inside one variant's directory would therefore mean either reopening a published run or declaring one variant the owner of a document about all of them.
Root level is also what the file *is*: a statement about several runs, like the registry.

## Public API

| Name | What it does |
| --- | --- |
| `format_metric(value, digits=4)` | One metric for one table cell. A missing one reads as `-`, never `0.0000`. |
| `sanity_warnings(payload)` | Everything about a run a reader must not miss, as sentences with the numbers in them. Empty means it passed these checks, not that it is good. |
| `render_model_card(payload, metadata, *, vintages=None)` | The run's `model_card.md`, as markdown. |
| `variant_label(payload)` | `logistic_regression / origination_only`. The row name in a comparison. |
| `comparison_table(payloads)` | One row per variant, baseline first, with signed deltas against the baseline. |
| `comparison_payload(payloads)` | `comparison.json`: the table, the winner by AUC, and the leakage cost per model type. |
| `columnar(frame)` | A table as one list per column, JSON-safe. The shape a chart wants. |
| `MODEL_CARD_FILENAME`, `COMPARISON_FILENAME` | The two filenames, so no caller spells them. |
| `COMPARISON_METRICS` | What is compared, each with the direction that counts as an improvement. |
| `SANITY_MIN_TEST_ROWS`, `SANITY_MIN_TEST_POSITIVES` | The thresholds below which a metric describes a handful of rows. |

## Inputs and outputs

Everything takes **mappings**, not objects from this project's own pipeline.

`render_model_card`, `sanity_warnings`, `comparison_table` and `comparison_payload` accept the `metrics.json` payload as a plain `Mapping[str, Any]`; the card additionally takes a `RunMetadata` and, optionally, the vintage table as a DataFrame.
Nothing here imports `pipeline.py`, so the dependency direction is one way: `pipeline -> reporting`.

That is not a style preference.
The payload mapping already carries `run_id`, `model_type`, `include_lender_priced`, the row counts and every metric, which is the whole input a comparison needs - so accepting a richer type would buy nothing and would let a card be rendered only from a live run object, never from a published artifact.
Because it takes the published mapping, `riskscore card` can re-render a card for a run trained by an older build.

Returns are strings and plain dicts.
Nothing here opens a file, writes a file, or logs.
`pipeline.py` writes `model_card.md` inside the staging directory and `comparison.json` through `write_json_atomic`; this module has no idea where either goes.

## Invariants and failure modes

### A missing metric renders as `-`, and there is exactly one implementation of that rule

`format_metric` is the only place that decides.
It was two places for one commit - a private `_number` here and another in `cli.py` - and mypy caught it as a signature mismatch the moment one of them grew a `digits` argument.
The card, the CLI tables and the dashboard formatter all now go through this one function, because a metric that reads as `-` in one surface and `0.0000` in another is worse than either.

An absent number rendered as zero is read as a *catastrophic model* rather than as *missing data*.
Two exclusions in the implementation are deliberate:

- **`bool` is excluded before the numeric check.** `True` is an `int` in Python, so a truthy flag reaching a metric column would print `1.0000` - a perfect score.
- **`nan` renders as `-`.** A single-class vintage legitimately has no AUC, and that is missing data, not a zero.

`0.0` stays `0.0000`. It was measured.

### Nothing on the card is computed here

The one apparent exception is the cost *ratio* (`5.0:1`), which is a division of two numbers already on the card and exists so a reader does not have to do it.
Every other value is a `payload.get(...)` or a `metadata.` attribute.

The rule is what makes the card auditable: any number on it can be found by grepping the two JSON files in the same directory.

### The baseline of a comparison is the first variant, never the best one

A comparison answers *"what does moving away from the default buy"*.
If the winner defined the baseline, every delta would flip sign the moment a different variant won, and someone comparing two comparisons a month apart would see improvements where nothing had changed.

So `comparison_table` computes `frame[name] - frame[name].iloc[0]`, and the caller controls the order - which is why `DEFAULT_COMPARISON_MODELS` puts `logistic_regression` first and why the variant list in `compare_runs` is built model-major.

### A neutral metric gets no delta column

`approval_rate` is in `COMPARISON_METRICS` marked `neutral` and is the only one.
It is a policy consequence, not a score: a variant approving 4% more of the book is neither better nor worse until someone says what the book is for.
A `approval_rate_delta` column would invite reading that difference as an improvement, so it is not generated.

The direction of every other metric is published in the payload's own `metrics` object, so the dashboard does not carry a second hardcoded copy of "lower Brier is better".

### The leakage cost is absent, not zero, when only one tier was fitted

`lender_priced_delta` has an entry per model type fitted under **both** tiers, and no entry otherwise.
Reporting a gain of `0.0` for a single-tier comparison would be a positive claim - that admitting the lender's price changes nothing - which is the opposite of what one tier measures.

### Every value in a JSON payload survives `json.loads`

`json.dumps` emits a bare `NaN` token by default.
That is not valid JSON, `JSON.parse` throws on it, and the consequence is concrete: one missing metric in one variant would take out a whole dashboard panel.

`_json_safe_value` maps `None`, `pd.NA`, `pd.NaT`, `nan` and `inf` to `null`, unwraps numpy scalars (which are not serializable at all), and ISO-formats timestamps.
Both `comparison_payload` and `columnar` go through it, and the tests dump with `allow_nan=False` so a regression fails at the writer rather than in a browser.

### The manifest is written in insertion order, and the card depends on it

Not an invariant of this file, but the bug that proved the coupling: `save_bundle` wrote `manifest.json` with `sort_keys=True`, so the round trip returned `rows` as `closed, mature, raw` instead of the order the filters actually ran in, and `riskscore card` printed a stage table that read as nonsense while the card published by the run itself was correct.

`rows` and `split_windows` are **ordered data**. `artifacts.py` no longer sorts them; `test_the_manifest_keeps_the_order_its_dicts_were_built_in` is the proof.

### The warning block sits directly under the headline numbers

`## Read this first` is rendered third, above `## Intended use`, and only when `sanity_warnings` returns something.
Placement is the point: a caveat below the metrics is a caveat somebody reads *after* quoting them.

The thresholds are 1000 test rows and 50 test defaults, which is roughly where an AUC's standard error drops under 0.04 - smaller than the differences this project reports between variants.
The committed artifact that prompted this work had **116 rows and one positive**, and the dashboard rendered its 0.07 AUC as a result.

### An empty comparison is an error; a one-variant comparison is not

`comparison_table([])` raises `ValueError`, because a table with no baseline has no meaning.
A single variant is legitimate - it is what `riskscore compare --model logistic_regression` produces - and yields one row with zero deltas.

### `columnar` exists for the byte count and for the chart

The threshold cost table is 99 rows of six numbers.
Serving it as `[{"threshold": ..., "total_cost": ...}, ...]` repeats every key 99 times, which is most of the payload.
One array per column is also the shape a canvas chart wants, so the client does no reshaping.

## What must NOT live here

- **Computing a metric.** `evaluation.py`, `calibration.py` and `drift.py` measure; this file formats. A number that is on the card and in no artifact is a bug in whichever file put it there.
- **Deciding what is compared.** `compare_runs` in `pipeline.py` owns the variant list and its order, because ordering is what the baseline rule depends on and `pipeline.py` is the module that fits things.
- **Writing files.** No `open`, no `Path.write_text`, no `write_json_atomic` call. The card is returned as a string so the pipeline can write it inside a staging directory and publish it by rename.
- **Any import from `pipeline.py`.** The direction is one way. Taking payload mappings rather than `RunResult` objects is what keeps it that way.
- **Serving.** ETags, gzip and caching belong to the API. This file produces the payload; the transport is `api/`'s problem.
- **HTML.** The card is markdown and the payloads are JSON. The dashboard renders.

## Related tests

`tests/test_reporting.py`, in four groups matching the file:

- **`format_metric`** - parametrized over `0.0`, `None`, `nan`, `inf`, a numpy `nan`, a string and `True`. `test_zero_is_a_measurement_and_stays_a_number` is the one that states the distinction the function exists for.
- **`sanity_warnings`** - one test per check, each breaking exactly one thing against an otherwise healthy payload, so a failure names the check it broke. `test_a_moderate_psi_band_is_not_a_warning` guards the other direction: a banner nobody can clear is a banner nobody reads.
- **The card** - asserted by *claim*, not by layout. That the measured numbers reach the document, that every documented heading is present, that no `$` or unclosed brace survives, and that an almost-empty payload renders without `0.0000` anywhere. Asserting the exact markdown would make every prose edit a test failure, which trains people to update the fixture without reading it.
- **The comparison** - `test_the_baseline_is_the_first_variant_and_not_the_best_one` and `test_a_delta_is_signed_in_the_metrics_own_units` are the two that cover a refactor silently inverting a sign.

`tests/test_cli.py` covers the three commands this module backs: that `compare` publishes every variant and activates **none**, that `comparison.json` lands at the root and parses with `json.loads`, that `card` re-renders byte-identically to the card the run published, and that `explain`'s reason codes come out largest-magnitude first.

`tests/test_pipeline.py::test_the_pipeline_writes_every_documented_artifact` includes `model_card.md` in the artifact contract.

## Known limits

- **The card has no fairness section, because none can be computed.** The extract contains no protected attribute, so there is nothing to test disparate impact against. `_ETHICAL` says so explicitly rather than substituting a proxy and calling the question answered - but a deployment subject to ECOA needs that test on data that does have those columns.
- **The static prose is not versioned separately from the code.** Re-rendering an old run's card with a newer build produces the newer prose over the older numbers. That is deliberate (the layout should follow the build) and it means `model_card.md` in a published run directory is the only record of what the card said *at the time*.
- **`comparison.json` is overwritten by every `compare`.** There is no history of comparisons, only of runs. Recovering an old comparison means re-reading the variants' `metrics.json` files, which is possible precisely because these functions take mappings.
- **No PDF, no HTML rendering of the card.** Markdown only. Anything that needs a rendered document pipes it through `pandoc`.
- **The vintage table is passed in whole and rendered whole.** A run with fifty vintages produces a fifty-row table on the card. There is no truncation because no run this project produces has more than a handful.
- **`format_metric` has one digit default for every metric.** Four decimals is right for an AUC and unnecessary for an approval rate; callers that care pass `digits`, and the CLI's `COMPARE_COLUMNS` is the only place that does.
