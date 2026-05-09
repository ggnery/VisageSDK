"""Tests for BaseConfig __getattr__ behavior."""

from pathlib import Path

import pytest
import yaml

from config.base_config import BaseConfig


@pytest.fixture
def config_file(tmp_path):
    def _make(data: dict) -> Path:
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(data))
        return p

    return _make


class TestBaseConfig:
    def test_loads_yaml_keys_as_attrs(self, config_file):
        path = config_file({"margin": 0.2, "device": "cpu"})
        cfg = BaseConfig(str(path))
        assert cfg.margin == 0.2
        assert cfg.device == "cpu"

    def test_nested_dicts_returned_as_dicts(self, config_file):
        path = config_file({"normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}})
        cfg = BaseConfig(str(path))
        assert cfg.normalize["mean"] == [0.5, 0.5, 0.5]

    def test_missing_attr_raises(self, config_file):
        path = config_file({"foo": 1})
        cfg = BaseConfig(str(path))
        with pytest.raises(AttributeError, match="no attribute 'bar'"):
            _ = cfg.bar

    def test_underscore_attrs_skipped(self, config_file):
        path = config_file({"_private": "x"})
        cfg = BaseConfig(str(path))
        # _private is in YAML but underscored — __getattr__ refuses
        with pytest.raises(AttributeError):
            _ = cfg._private

    def test_explicit_attr_takes_precedence(self, config_file):
        path = config_file({"device": "cpu"})
        cfg = BaseConfig(str(path))
        # BaseConfig has no static `device` attr; subclasses populate concrete
        # attrs via __dict__ during their own __init__.
        cfg.__dict__["device"] = "cuda"
        assert cfg.device == "cuda"

    def test_get_config_string_includes_class_and_keys(self, config_file):
        path = config_file({"margin": 0.5, "device": "cpu"})
        cfg = BaseConfig(str(path))
        s = cfg.get_config_string()
        assert "BaseConfig CONFIGURATION" in s
        assert "margin: 0.5" in s
        assert "device: cpu" in s

    def test_get_config_string_dedups_when_attr_set_explicitly(self, config_file):
        path = config_file({"x": 1})
        cfg = BaseConfig(str(path))
        cfg.__dict__["x"] = 2  # overrides via __dict__
        s = cfg.get_config_string()
        # Only one "x: ..." line should appear (the explicit one)
        assert s.count("x: ") == 1
        assert "x: 2" in s

    def test_empty_yaml_file_handled(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = BaseConfig(str(p))
        with pytest.raises(AttributeError):
            _ = cfg.anything
