import logging
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

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._load()

    def _load(self) -> None:
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            get_config_schema().validate(self.config)
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found.")
            self.config = None
        except yaml.YAMLError as exc:
            logger.error(
                f"Error while parsing configuration file {self.config_path}: {exc}"
            )
            self.config = None
        except SchemaError as exc:
            logger.error(
                f"Error while validating configuration file {self.config_path}: {exc}"
            )
            self.config = None

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
        if self.config is None:
            raise ValueError("Configuration not loaded.")

        keys = key.split(".")
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def all(self) -> dict:
        """Get the entire configuration.

        Raises:
            VALUE_ERROR: If the coinfiguration is not loaded.

        Returns:
            config: The entire configuration.
        """

        if self.config is None:
            raise ValueError("Configuration not loaded.")

        return self.config
