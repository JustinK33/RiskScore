"""Tests for run configuration loading and validation.

The old suite had one test, asserting that YAML parses into a dict. That is a
test of PyYAML. What needed testing was that a config file which says something
wrong is *rejected*, and there was nothing to reject with.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from risk_score.config import RunConfig, SplitConfig, load_run_config, load_yaml_config

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "run.yaml"


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# --- the shipped config --------------------------------------------------------


def test_the_shipped_config_loads(tmp_path: Path) -> None:
    """It stopped loading once a minimum alias length was enforced, and nothing
    noticed because no test read it."""
    config = load_run_config(PROJECT_CONFIG)

    assert config.split.date_column == "issue_d"
    assert config.split.train == ("2013-01", "2014-12")
    assert config.split.validation == ("2015-01", "2015-12")
    assert config.split.test == ("2016-01", "2016-12")
    assert config.cost_matrix.false_negative_cost == 5.0
    assert config.include_lender_priced is False


def test_the_shipped_config_matches_the_in_code_defaults() -> None:
    """Two sources of the same default is one source too many; this is the check
    that keeps the YAML and the dataclass from drifting apart.

    ``models`` is exempt: the YAML lists each model with an empty override block
    so a reader can see where hyperparameters go, and empty means "use the
    tested defaults".
    """
    shipped = load_run_config(PROJECT_CONFIG)
    defaults = RunConfig()

    assert shipped.split == defaults.split
    assert shipped.cost_matrix == defaults.cost_matrix
    assert shipped.include_lender_priced == defaults.include_lender_priced
    assert shipped.column_aliases == {}
    assert all(params == {} for params in shipped.models.values())


# --- validation ----------------------------------------------------------------


def test_b30_an_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    """`.get` on a bare dict turns a typo into a silently applied default."""
    path = write(tmp_path, "spilt:\n  train: ['2013-01', '2013-12']\n")
    with pytest.raises(ValueError, match="Unknown key\\(s\\) in `config`: \\['spilt'\\]"):
        load_run_config(path)


def test_b30_an_unknown_split_key_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "split:\n  train_end_date: '2014-12-31'\n")
    with pytest.raises(ValueError, match="train_end_date"):
        load_run_config(path)


def test_b30_an_unknown_threshold_key_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "threshold:\n  false_negatve_cost: 9.0\n")
    with pytest.raises(ValueError, match="false_negatve_cost"):
        load_run_config(path)


def test_the_error_lists_the_keys_that_would_have_worked(tmp_path: Path) -> None:
    path = write(tmp_path, "leakage:\n  include_lender_price: true\n")
    with pytest.raises(ValueError, match="Valid keys: \\['include_lender_priced'\\]"):
        load_run_config(path)


def test_hyperparameter_names_are_not_validated_here(tmp_path: Path) -> None:
    """sklearn and XGBoost reject their own unknown parameters, and name them.
    Duplicating that list here would go stale on every library upgrade."""
    path = write(tmp_path, "models:\n  logistic_regression:\n    C: 0.25\n    solver: saga\n")
    assert load_run_config(path).model_params("logistic_regression") == {
        "C": 0.25,
        "solver": "saga",
    }


def test_a_window_with_three_bounds_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "split:\n  train: ['2013-01', '2013-06', '2013-12']\n")
    with pytest.raises(ValueError, match="needs exactly \\[start, end\\]"):
        load_run_config(path)


def test_a_window_written_as_one_string_is_rejected(tmp_path: Path) -> None:
    """`train: 2013-01..2014-12` would otherwise iterate into characters."""
    path = write(tmp_path, "split:\n  train: '2013-01'\n")
    with pytest.raises(TypeError, match="two-item \\[start, end\\] list"):
        load_run_config(path)


def test_a_non_mapping_section_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "threshold: 5.0\n")
    with pytest.raises(TypeError, match="`threshold` must be a mapping"):
        load_run_config(path)


def test_a_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "- split\n- threshold\n")
    with pytest.raises(TypeError, match="must be a mapping"):
        load_run_config(path)


def test_a_missing_file_names_itself(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"absent\.yaml"):
        load_run_config(tmp_path / "absent.yaml")


# --- defaults and partial configs ----------------------------------------------


def test_an_empty_config_is_the_default_config(tmp_path: Path) -> None:
    """A run should not need a config file to be a correct run."""
    path = write(tmp_path, "# nothing but a comment\n")
    assert load_run_config(path) == RunConfig()


def test_a_partial_section_keeps_the_other_defaults(tmp_path: Path) -> None:
    path = write(tmp_path, "threshold:\n  false_negative_cost: 12.0\n")
    config = load_run_config(path)

    assert config.cost_matrix.false_negative_cost == 12.0
    assert config.cost_matrix.false_positive_cost == 1.0
    assert config.split == SplitConfig()


def test_the_lender_priced_tier_can_be_turned_on_from_the_config(tmp_path: Path) -> None:
    path = write(tmp_path, "leakage:\n  include_lender_priced: true\n")
    assert load_run_config(path).include_lender_priced is True


def test_a_model_with_no_overrides_yields_an_empty_parameter_dict(tmp_path: Path) -> None:
    path = write(tmp_path, "models:\n  xgboost: {}\n")
    config = load_run_config(path)

    assert config.model_params("xgboost") == {}
    assert config.model_params("logistic_regression") == {}


def test_model_params_returns_a_copy(tmp_path: Path) -> None:
    """It is handed straight to an estimator constructor; a caller mutating it
    must not edit the config every later run reads."""
    path = write(tmp_path, "models:\n  logistic_regression:\n    C: 0.5\n")
    config = load_run_config(path)
    config.model_params("logistic_regression")["C"] = 99.0

    assert config.model_params("logistic_regression") == {"C": 0.5}


def test_column_aliases_are_normalized_to_tuples(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "column_aliases:\n  loan_amnt:\n    - loanAmountRequested\n  annual_inc: yearlyIncome\n",
    )
    config = load_run_config(path)

    assert config.column_aliases == {
        "loan_amnt": ("loanAmountRequested",),
        # A single alias may be written as a plain string rather than a one-item
        # list, because everyone writes it that way at least once.
        "annual_inc": ("yearlyIncome",),
    }


# --- the raw loader ------------------------------------------------------------


def test_load_yaml_config_reads_a_mapping(tmp_path: Path) -> None:
    path = write(tmp_path, "models:\n  logistic_regression:\n    C: 1.0\n")
    assert load_yaml_config(path) == {"models": {"logistic_regression": {"C": 1.0}}}


def test_load_yaml_config_refuses_to_construct_arbitrary_objects(tmp_path: Path) -> None:
    """A config file is input, and `!!python/object` in one is code execution."""
    path = write(tmp_path, "split: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(Exception, match="python/object"):
        load_yaml_config(path)
