#!/usr/bin/env python3
"""Assert that every code file has a documentation page, and that every page is complete.

Documentation that can rot silently does. Nothing about a stale ``docs/code/``
page fails a test, shows up in a diff, or looks wrong at a glance - which is
exactly how a file gets renamed and its page keeps describing the old one for a
year. This script is the enforcement, and it runs in CI.

Three things are checked, in the order a reader would notice them:

1.  Every in-scope code file has a page.
2.  Every page corresponds to a file that exists.
3.  Every page carries all seven required headings, spelled exactly.

Deliberately stdlib only and deliberately not a pytest test: it is a repository
invariant rather than a property of the library, it needs to run from a checkout
with no dependencies installed, and its failure message wants to be a list of
missing files rather than an assertion diff.

    python scripts/check_docs.py

Exits 0 when clean, 1 with a list of problems otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "docs" / "code"

#: Glob patterns, relative to the repository root, of files that must have a page.
#:
#: Configs, `.gitignore`, and READMEs are out of scope as a deliberate decision:
#: they are either self-describing or documented where they constrain code.
#: `scripts/check_docs.py` - this file - is in scope, because a rule that exempts
#: its own enforcement is a rule with a hole in it.
IN_SCOPE = (
    "src/risk_score/*.py",
    "src/risk_score/api/*.py",
    "dashboard/*.html",
    "dashboard/styles/*.css",
    "dashboard/js/*.js",
    "tests/conftest.py",
    "scripts/*.py",
    "scripts/*.mjs",
)

#: Files matching an in-scope pattern that deliberately have no page, each with
#: the reason, because an unexplained exemption is indistinguishable from an
#: oversight. Both `__init__.py` files re-export names and state a dependency
#: direction; the JS test files are covered by their module's "Related tests".
EXEMPT = {
    "src/risk_score/__init__.py": "re-exports the public names; nothing to describe",
    "src/risk_score/api/__init__.py": "re-exports two names and states api -> risk_score",
}

#: Two stylesheets whose stems collide with pages that already exist for Python
#: files - `docs/code/app.md` is `src/risk_score/api/app.py`. Named symmetrically
#: rather than only disambiguating the one that clashes.
PAGE_NAME_OVERRIDES = {
    "dashboard/styles/tokens.css": "styles-tokens",
    "dashboard/styles/app.css": "styles-app",
}

#: Spelled exactly, in order. The order is not enforced - the presence is - but a
#: page that reorders them is harder to skim, which is the whole point of having
#: fixed headings.
REQUIRED_HEADINGS = (
    "## Purpose",
    "## Public API",
    "## Inputs and outputs",
    "## Invariants and failure modes",
    "## What must NOT live here",
    "## Related tests",
    "## Known limits",
)


def page_for(relative_path: str) -> Path:
    """The page a given code file must have."""
    override = PAGE_NAME_OVERRIDES.get(relative_path)
    stem = override if override else Path(relative_path).stem
    return PAGES / f"{stem}.md"


def in_scope_files() -> list[str]:
    """Every code file that must have a page, as repo-relative posix paths."""
    found: set[str] = set()
    for pattern in IN_SCOPE:
        for path in ROOT.glob(pattern):
            relative = path.relative_to(ROOT).as_posix()
            # `*.test.js` files are excluded by rule, not by exemption: they are
            # test files for a module whose own page names them, so listing six
            # of them individually in EXEMPT would be noise that grows with the
            # suite.
            if relative.endswith(".test.js"):
                continue
            if relative in EXEMPT:
                continue
            found.add(relative)
    return sorted(found)


def main() -> int:
    problems: list[str] = []
    expected_pages: set[Path] = set()

    for relative in in_scope_files():
        page = page_for(relative)
        expected_pages.add(page)
        if not page.exists():
            problems.append(f"{relative}: no page at docs/code/{page.name}")
            continue

        text = page.read_text(encoding="utf-8")
        missing = [heading for heading in REQUIRED_HEADINGS if f"\n{heading}\n" not in f"\n{text}"]
        if missing:
            problems.append(f"docs/code/{page.name}: missing {', '.join(missing)}")

    # The other direction. A page whose file was renamed or deleted keeps
    # describing something that no longer exists, and nothing else would notice.
    for page in sorted(PAGES.glob("*.md")):
        if page.name == "README.md" or page in expected_pages:
            continue
        problems.append(f"docs/code/{page.name}: no in-scope code file needs this page")

    # An index nobody has to update is an index that goes stale, so every page is
    # required to be linked from it.
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for page in sorted(expected_pages):
        if page.exists() and f"({page.name})" not in index:
            problems.append(f"docs/code/{page.name}: not linked from docs/code/README.md")

    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} documentation problem(s)")
        return 1
    print(f"{len(expected_pages)} pages, all present and complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
