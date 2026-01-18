import pytest

from src.config.configuration import Configuration


def test_configuration_loads_valid_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset:\n  type: FashionMNIST\n", encoding="utf-8")

    config = Configuration(str(config_path))

    assert config.all() == {"dataset": {"type": "FashionMNIST"}}
    assert config.get("dataset.type") == "FashionMNIST"


def test_configuration_get_returns_default_for_missing_key(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset:\n  type: FashionMNIST\n", encoding="utf-8")

    config = Configuration(str(config_path))

    assert config.get("dataset.storage_path", "data") == "data"


def test_configuration_handles_missing_file(tmp_path):
    config_path = tmp_path / "missing.yaml"

    config = Configuration(str(config_path))

    with pytest.raises(ValueError, match="Configuration not loaded"):
        config.get("dataset.type")


def test_configuration_handles_invalid_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset: [", encoding="utf-8")

    config = Configuration(str(config_path))

    with pytest.raises(ValueError, match="Configuration not loaded"):
        config.all()


def test_configuration_handles_schema_error(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset:\n  type: 123\n", encoding="utf-8")

    config = Configuration(str(config_path))

    with pytest.raises(ValueError, match="Configuration not loaded"):
        config.get("dataset.type")
