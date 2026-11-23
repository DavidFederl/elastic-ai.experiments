import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _expand(value: Any) -> Any:
    """Expand environment variables like ``$SCRATCH`` in all strings."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_config(path: Path, config_type: type[T] | None = None) -> T | dict[str, Any]:
    """Load a YAML config file.

    Environment variables in string values are expanded, so a config can
    refer to cluster paths such as ``$SCRATCH/datasets/cifar10.h5``.

    Args:
        path (Path): Path to the YAML file.
        config_type (type | None): Optional dataclass. If given, the config
            is validated against it and returned as an instance.

    Returns:
        T | dict[str, Any]: Config as dataclass instance, or as dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a YAML mapping, or if keys do not
            match the dataclass fields.
    """
    with Path(path).open("r") as config_file:
        raw = yaml.safe_load(config_file)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a YAML mapping.")

    config = _expand(raw)

    if config_type is None:
        return config
    if not is_dataclass(config_type):
        raise TypeError(f"{config_type!r} is not a dataclass.")

    names = {field.name for field in fields(config_type)}
    unknown = set(config) - names
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")

    return config_type(**config)
