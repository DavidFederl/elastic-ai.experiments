import logging
from pathlib import Path
from typing import TypeVar

import yaml
from schema import SchemaError

from .config_schema import get_config_schema

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Configuration:
    """Configuration for the experiment.

    Args:
        config_path (str): Path to the configuration file.

    Raises:
        FILE_NOT_FOUND: If the configuration file is not found.
        YAML_ERROR: If the configuration file is not valid YAML.
        SCHEMA_ERROR: If the configuration file is not valid according to the schema.

    Returns:
        None
    """

    def __init__(self, config: Path):
        self.config_file = config
        try:
            with open(config, "r") as cf:
                self.configuration: dict | None = yaml.safe_load(cf)
                get_config_schema().validate(self.configuration)
        except FileNotFoundError as exc:
            logger.error(f"File not found: {exc}")
            self.configuration = None
        except yaml.YAMLError as exc:
            logger.error(f"Error while parsing configuration file: {exc}")
            self.configuration = None
        except SchemaError as exc:
            logger.error(f"Error while validating configuration file: {exc}")
            self.configuration = None

    def get(self, key: str, default: T = None) -> T:
        """Get a value from the configuration.

        Args:
            key (str): Key to the value.
            default (Any, optional): Default value if the key is not found. Defaults to None.

        Raises:
            VALUE_ERROR: If the cofiguration is not loaded.

        Returns:
            value: Value of the key.
        """
        if self.configuration is None:
            raise ValueError("Configuration not loaded.")

        keys = key.split(".")
        value = self.configuration
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_all(self) -> dict:
        """Get the entire configuration.

        Raises:
            VALUE_ERROR: If the coinfiguration is not loaded.

        Returns:
            config: The entire configuration.
        """

        if self.configuration is None:
            raise ValueError("Configuration not loaded.")

        return self.configuration

    def save(self, path: Path) -> None:
        """Writes config as YAML to given Path.

        Args:
            path (str): Path to write configuration to.

        Raises:
            FILE_NOT_FOUND: If the configuration file is not found.
            PERMISSION_DENIED: if the configuration can't be written.
            YAML_ERROR: If the configuration file is not valid YAML.

        Returns:
            None
        """
        try:
            with open(self.config_file, "w") as cf:
                yaml.dump(self.configuration, cf, default_flow_style=False)
        except FileNotFoundError as exc:
            logger.error(f"File not found: {exc}")
        except PermissionError as exc:
            logger.error(f"Permission denied: {exc}")
        except yaml.YAMLError as exc:
            logger.error(f"Error while parsing configuration file: {exc}")
