"""Tests for configuration helpers."""

from pathlib import Path

from risk_score.config import load_yaml_config


def test_load_yaml_config_reads_mapping(tmp_path: Path) -> None:
    """YAML configs should load as dictionaries."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: logistic_regression\n", encoding="utf-8")

    result = load_yaml_config(config_path)

    assert result == {"model": {"name": "logistic_regression"}}
