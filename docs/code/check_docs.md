# `scripts/check_docs.py`

## Purpose

Make the per-file documentation impossible to let rot.

Nothing about a stale `docs/code/` page fails a test, appears in a diff, or looks wrong at a glance. A file gets renamed, its page keeps describing the old one, and a year later the page is worse than no page - because a reader trusts it. Every documentation convention in this repository that is not enforced here is an aspiration.

Three checks, in the order a reader notices the failure:

1. Every in-scope code file has a page.
2. Every page corresponds to a file that still exists.
3. Every page carries all seven required headings, spelled exactly.

Plus a fourth that is really about the index: every page has to be linked from `docs/code/README.md`, because an index nobody is forced to update is an index that goes stale in the same silent way.

This file is itself in scope, which is why this page exists. A rule that exempts its own enforcement is a rule with a hole in it.

## Public API

A script. No exports worth importing, though the three module constants are the actual specification and are meant to be read:

| Name | What it is |
| --- | --- |
| `IN_SCOPE` | Eight globs, relative to the repository root. |
| `EXEMPT` | Files matching a glob that deliberately have no page, **each with its reason as the value**. |
| `PAGE_NAME_OVERRIDES` | The two stylesheets whose stems would collide. |
| `REQUIRED_HEADINGS` | The seven headings, spelled exactly. |
| `page_for(relative_path)` | The page a given code file must have. |
| `in_scope_files()` | Every repo-relative path that needs one. |

```
python scripts/check_docs.py
```

Exits `0` with a count, `1` with one line per problem.

## Inputs and outputs

Reads the filesystem under the repository root, located from `__file__` rather than from the working directory, so it gives the same answer from anywhere.

Reads every `docs/code/*.md`, including `README.md` for the link check.

Writes to stdout. Modifies nothing.

Stdlib only - `pathlib` and `sys`. No dependency, so it runs in a bare checkout before `pip install`.

## Invariants and failure modes

**A page maps to a code file by stem, with two named overrides.**
`features.py` → `features.md`. The overrides exist because `dashboard/styles/app.css` and `src/risk_score/api/app.py` share a stem, and `docs/code/app.md` was already the Python one. Both stylesheets are renamed - `styles-tokens.md` and `styles-app.md` - rather than only the one that clashed, because an asymmetric pair reads as an accident.

**Both directions are checked.**
A missing page and an orphaned page are different failures with different fixes: the first means somebody added a file, the second means somebody renamed or deleted one and the page survived. Checking only the first is the common form of this script and it is the half that goes stale.

**`EXEMPT` is a dict, not a set, and the values are the reasons.**
An unexplained exemption is indistinguishable from an oversight. Two entries: both `__init__.py` files, which re-export names and state a dependency direction. `src/risk_score/api/__init__.py`'s one load-bearing consequence - that importing it pulls fastapi in transitively - is documented where it constrains code, in [cli.md](cli.md) and [bench.md](bench.md).

**`*.test.js` is excluded by rule, not by exemption.**
Six files today and one more with every new module. Listing them individually would make `EXEMPT` grow with the suite, and the rule is a real rule: a test file's page is its module's **Related tests** section.

**Headings are matched with surrounding newlines.**
`f"\n{heading}\n" not in f"\n{text}"` - so `## Purpose` does not match a line reading `## Purpose and scope`, and a heading that happens to appear inside a fenced code block on its own line does match, which is a false negative nobody has hit and would notice immediately.

**The heading order is not enforced.**
Presence is. Every page happens to use the canonical order and a reordered page is harder to skim, which is the whole reason for fixed headings - but a check on order would fail on a page that added a legitimate extra `##` section, and none of them do.

**`check_docs.py` is in scope; `pyproject.toml` and `.gitignore` are not.**
The scope line is `docs/README.md`'s and it is a decision, not an omission: configs are either self-describing or documented where they constrain code. `scripts/*.py` and `scripts/*.mjs` are in scope because both files there contain real logic - this one and the dashboard probe.

**The link check is on `(page.name)` appearing anywhere in the index.**
A substring match on the Markdown link target, which is loose enough to accept any link text and tight enough that a page nobody linked fails. It does not check that the row says anything true.

**Every problem is printed, then the count.**
No early return. Adding four files should produce four lines in one run, not four runs.

## What must NOT live here

- **Any dependency.** It has to run in a bare checkout, and a docs checker that needs an install is one that gets skipped in the job where it matters.
- **A check on a page's *content* beyond its headings.** Whether the Purpose section is true is a review question. A script that tried would either be trivially satisfiable or wrong.
- **Writing.** No generating a stub page, no fixing the index. A missing page is a decision for a person; a generated one would satisfy this script and tell a reader nothing.
- **A `pytest` test wrapper.** This is a repository invariant, not a property of the library. Its output wants to be a list of paths, not an assertion diff, and it must not need the package importable.
- **Coverage of anything outside the repository.** No network, no git.

## Related tests

None, and that is a deliberate asymmetry worth naming: this script *is* a test, and a test for a test is where the regress has to stop.

What stands in for one is that it is run in CI on every push, and that it fails loudly by construction - the first time it was run it reported exactly one problem, its own missing page, which is the shortest available proof that the scope glob and the page lookup both work.

The conventions it enforces are stated in [../README.md](../README.md#documentation-scope) and [README.md](README.md); the audit-ID convention it does *not* enforce is in [../conventions.md](../conventions.md#audit-id-tags).

## Known limits

- **It cannot tell a good page from a page with seven empty headings.** Every structural property this file can check is checkable; every interesting one is not. The mitigation is that the pages are reviewed as prose, once, and that a heading nobody filled is visible in a diff.
- **The audit-ID convention is unenforced.** `docs/conventions.md` says every ID appears at a fix and in a `test_<id>_*` test. A script asserting that would be twenty lines of `rg` and has not been written; today an ID with a comment and no test is caught only by reading.
- **`docs/architecture.md`, `docs/data-dictionary.md`, `docs/runbook.md`, and `docs/decisions/*.md` are not checked at all.** Only `docs/code/` is. A broken relative link anywhere in the tree passes. A link checker would be the addition and would need to handle anchors.
- **The link check does not verify the row's text.** A page linked with a one-line summary describing a different file passes.
- **The globs are not recursive.** `src/risk_score/*.py` and `src/risk_score/api/*.py` are spelled separately, so a new subpackage is silently out of scope until someone adds a glob. `**/*.py` would be the fix and would sweep in `__pycache__` unless filtered.
- **No `--fix` and no listing mode.** `python scripts/check_docs.py` is the whole interface. A `--list` printing the in-scope set would help when adding a glob.
