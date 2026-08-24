"""Shared fixtures for the test suite.

Three near-identical 8-row dataframes used to be pasted into
``tests/test_pipeline.py``, which meant any schema change needed three edits and
none of them exercised realistic data. Everything now comes from
:mod:`risk_score.sample_data`, so tests and the ``riskscore make-sample-data``
demo path share one generator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from risk_score.api import Settings, create_app
from risk_score.pipeline import RunResult, train_run
from risk_score.sample_data import make_synthetic_loans

# Small enough to keep the default `pytest -q` run under a couple of seconds,
# large enough that a tri-split still leaves positives in every partition.
SMALL_ROWS = 900

#: The real dashboard, located from this file rather than from the working
#: directory. ``Settings.dashboard_dir`` defaults to a relative path, so a suite
#: that let it default would mount the dashboard when run from the repo root and
#: not when run from anywhere else - a difference nobody would look for.
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


def _xgboost_loads() -> bool:
    """Whether XGBoost can actually be used, not merely whether it is installed.

    ``pytest.importorskip("xgboost")`` is not enough: on macOS without the
    OpenMP runtime the package is present and importing it raises
    ``XGBoostError`` from a failed ``dlopen``, which is not an ``ImportError``,
    so importorskip lets it through and the test *fails* instead of skipping.
    """
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


#: Applied to every test that fits a real gradient-boosted model. The stub-based
#: tests deliberately carry no such marker: the fit-transform-reassemble logic
#: they cover is this project's, and it must be verified on every machine.
requires_xgboost = pytest.mark.skipif(
    not _xgboost_loads(),
    reason="xgboost cannot be loaded - on macOS run `brew install libomp`",
)


@pytest.fixture
def raw_loans() -> pd.DataFrame:
    """A raw Lending Club-shaped frame: percent strings, junk columns, NaNs."""
    return make_synthetic_loans(n_rows=SMALL_ROWS)


@pytest.fixture
def raw_loans_factory() -> Callable[..., pd.DataFrame]:
    """The generator itself, for tests that need non-default options."""
    return make_synthetic_loans


@pytest.fixture
def raw_csv(tmp_path: Path, raw_loans: pd.DataFrame) -> Path:
    """The same frame written to disk, for tests that go through the CSV reader."""
    destination = tmp_path / "loans.csv"
    raw_loans.to_csv(destination, index=False)
    return destination


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> RunResult:
    """One published run, fitted once for the whole session.

    Session-scoped on purpose. Explainability, drift, and reporting all need a
    *real* fitted bundle - a stub estimator would not have coefficients, a
    design matrix, or a background - and refitting per test would add a couple of
    seconds each to a suite that currently runs in twelve.

    Consumers must treat it as read-only. Nothing here mutates the run directory,
    and a test that needs to is expected to fit its own.
    """
    root = tmp_path_factory.mktemp("trained")
    dataset = root / "loans.csv"
    make_synthetic_loans(n_rows=SMALL_ROWS).to_csv(dataset, index=False)
    return train_run(dataset, output_dir=root / "reports")


@pytest.fixture
def api_settings(trained_run: RunResult) -> Settings:
    """Service settings pointing at the session's published run.

    Built explicitly rather than from the environment. ``Settings()`` reads
    ``RISKSCORE_*`` and ``.env``, so a test that relied on the default would pass
    or fail depending on the developer's shell.
    """
    return Settings(
        reports_dir=trained_run.run_dir.parent.parent,
        dashboard_dir=DASHBOARD_DIR,
        log_level="WARNING",
    )


@pytest.fixture
def client(api_settings: Settings) -> Iterator[TestClient]:
    """A client over a real app with the session's bundle loaded.

    ``raise_server_exceptions=False`` so an unhandled error is delivered as the
    500 a real client would receive, which is the thing worth asserting about -
    the default re-raises it into the test and the response body is never seen.
    """
    with TestClient(create_app(api_settings), raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def applicant() -> dict[str, object]:
    """One valid applicant, in the extract's own string dialect.

    Strings rather than numbers for ``term``, ``emp_length`` and ``revol_util``
    because that is the harder case and the one a client copying values out of a
    Lending Club CSV will send.
    """
    return {
        "issue_d": "Jun-2015",
        "loan_amnt": 15000,
        "term": " 36 months",
        "purpose": "debt_consolidation",
        "annual_inc": 62000,
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "addr_state": "CA",
        "dti": 18.2,
        "delinq_2yrs": 0,
        "earliest_cr_line": "Aug-2003",
        "inq_last_6mths": 1,
        "open_acc": 9,
        "pub_rec": 0,
        "revol_bal": 12000,
        "revol_util": "62.5%",
        "total_acc": 21,
    }


@pytest.fixture
def raw_csv_factory(tmp_path: Path) -> Callable[..., Path]:
    """Write a synthetic CSV with custom generator options and return its path."""
    counter = 0

    def _write(**kwargs: object) -> Path:
        nonlocal counter
        counter += 1
        destination = tmp_path / f"loans_{counter}.csv"
        make_synthetic_loans(**kwargs).to_csv(destination, index=False)  # type: ignore[arg-type]
        return destination

    return _write
