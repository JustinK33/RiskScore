"""The run configuration: one file, typed, and validated on load.

Three things made the old configuration worse than no configuration.

**It duplicated the column registry, and the copies disagreed.**
``configs/feature_config.yaml`` listed ``grade`` and ``sub_grade`` as ordinary
features (they are lender-priced, and off by default now), and listed
``fico_band`` as engineered from ``fico_range_*`` columns that are absent from
both real extracts. Which columns exist, what they are called, how they parse,
and whether they may be features is settled in :mod:`risk_score.features` and
nowhere else - so that file is gone rather than loaded.

**Nothing validated it.** ``load_yaml_config`` returned a bare dict and callers
reached into it with ``.get``, so a misspelled key was not an error, it was a
silently applied default (audit B30). A config that says
``class_wieght: balanced`` should fail loudly, not train a different model than
the one you asked for. Every mapping here is parsed into a frozen dataclass that
rejects keys it does not recognize.

**It had gone stale enough to break the run.** ``configs/dataset_schema.yaml``
still declared ``n`` as an alias for ``loan_status``. Phase 1 added a minimum
alias length precisely because a one-letter alias silently hijacks the target, so
the shipped config raised on load - every ``scripts/run_baseline.py`` invocation
died on it. Nothing tested the config files, because there was nothing to test:
they were dicts.

What remains configurable is what a run legitimately varies: the split windows,
the model hyperparameters, the cost matrix, the feature tier, and an alias
escape hatch for onboarding an unfamiliar extract without a code change.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from risk_score.evaluation import CostMatrix

#: Shipped defaults, so ``RunConfig()`` is a usable configuration and the YAML
#: only has to state what it changes. The windows suit the synthetic extract and
#: the 2013-2016 slice of the real one; see ``docs/decisions``.
DEFAULT_SPLIT_WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2013-01", "2014-12"),
    "validation": ("2015-01", "2015-12"),
    "test": ("2016-01", "2016-12"),
}


def _reject_unknown_keys(mapping: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    """Fail on any key the reader would otherwise ignore.

    The entire point of the exercise: a typo in a config file must not become a
    silently applied default (audit B30). The message lists the valid keys,
    because "unknown key" without them sends the reader to the source.
    """
    permitted = set(allowed)
    unknown = sorted(set(mapping) - permitted)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in `{where}`: {unknown}. Valid keys: {sorted(permitted)}."
        )


def _as_mapping(value: object, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"`{where}` must be a mapping, got {type(value).__name__}.")
    return value


def _as_window(value: object, where: str) -> tuple[str, str]:
    """A window is ``[start, end]``, and both bounds may be partial dates.

    Kept as strings rather than parsed here: expanding ``2014-12`` to the last
    instant of December is :class:`risk_score.modeling.TimeWindow`'s job, and two
    places doing it differently is how a month goes missing from a split.
    """
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise TypeError(f"`{where}` must be a two-item [start, end] list, got {value!r}.")
    bounds = [str(bound) for bound in value]
    if len(bounds) != 2:
        raise ValueError(f"`{where}` needs exactly [start, end]; got {bounds}.")
    return bounds[0], bounds[1]


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Which vintages train the model, tune the decision rule, and are reported."""

    date_column: str = "issue_d"
    train: tuple[str, str] = DEFAULT_SPLIT_WINDOWS["train"]
    validation: tuple[str, str] = DEFAULT_SPLIT_WINDOWS["validation"]
    test: tuple[str, str] = DEFAULT_SPLIT_WINDOWS["test"]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SplitConfig:
        _reject_unknown_keys(mapping, ("date_column", "train", "validation", "test"), "split")
        defaults = cls()
        return cls(
            date_column=str(mapping.get("date_column", defaults.date_column)),
            train=_as_window(mapping.get("train", defaults.train), "split.train"),
            validation=_as_window(
                mapping.get("validation", defaults.validation), "split.validation"
            ),
            test=_as_window(mapping.get("test", defaults.test), "split.test"),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything one training run needs that is not a property of the data."""

    split: SplitConfig = field(default_factory=SplitConfig)
    cost_matrix: CostMatrix = field(
        default_factory=lambda: CostMatrix(false_negative_cost=5.0, false_positive_cost=1.0)
    )
    include_lender_priced: bool = False
    models: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    column_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def model_params(self, model_type: str) -> dict[str, Any]:
        """Hyperparameter overrides for one model, or none.

        Absent is not an error: the defaults in :mod:`risk_score.modeling` are
        the tested ones, and a config that overrides nothing is a good config.
        """
        return dict(self.models.get(model_type, {}))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> RunConfig:
        """Parse a loaded YAML document, rejecting anything unrecognized."""
        _reject_unknown_keys(
            mapping, ("split", "threshold", "leakage", "models", "column_aliases"), "config"
        )

        threshold = _as_mapping(mapping.get("threshold"), "threshold")
        _reject_unknown_keys(threshold, ("false_negative_cost", "false_positive_cost"), "threshold")
        defaults = cls()
        costs = CostMatrix(
            false_negative_cost=float(
                threshold.get("false_negative_cost", defaults.cost_matrix.false_negative_cost)
            ),
            false_positive_cost=float(
                threshold.get("false_positive_cost", defaults.cost_matrix.false_positive_cost)
            ),
        )

        leakage = _as_mapping(mapping.get("leakage"), "leakage")
        _reject_unknown_keys(leakage, ("include_lender_priced",), "leakage")

        models = _as_mapping(mapping.get("models"), "models")
        # Hyperparameter names are not enumerated: they belong to sklearn and
        # XGBoost, both of which reject an unknown one themselves and name it.
        for name, params in models.items():
            _as_mapping(params, f"models.{name}")

        aliases = _as_mapping(mapping.get("column_aliases"), "column_aliases")
        column_aliases = {
            str(canonical): (
                (str(values),) if isinstance(values, str) else tuple(str(v) for v in values)
            )
            for canonical, values in aliases.items()
        }

        return cls(
            split=SplitConfig.from_mapping(_as_mapping(mapping.get("split"), "split")),
            cost_matrix=costs,
            include_lender_priced=bool(
                leakage.get("include_lender_priced", defaults.include_lender_priced)
            ),
            models={
                str(name): dict(_as_mapping(params, "models")) for name, params in models.items()
            },
            column_aliases=column_aliases,
        )


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML document as a mapping. No interpretation, no defaults."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        # safe_load, never load: a config file is input, and `!!python/object`
        # in one is arbitrary code execution.
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise TypeError(f"Top-level YAML config must be a mapping, got {type(config).__name__}.")
    return config


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate ``configs/run.yaml`` (or a copy of it)."""
    return RunConfig.from_mapping(load_yaml_config(path))
