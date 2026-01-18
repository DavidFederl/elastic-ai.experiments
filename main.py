import logging
from datetime import datetime
from os import makedirs

import click

from src.config.configuration import Configuration
from src.nn.data import Dataset, FashionMNIST

logger: logging.Logger
configuration: Configuration


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Configuration file defining the experiment.",
)
@click.option("--debug/--no-debug", default=False)
@click.option("--verbose/--no-verbose", default=False)
@click.option("--log-dir", default=f"logs/{int(datetime.now().timestamp() * 1000)}")
def main(debug: bool, verbose: bool, log_dir: str, config: str):
    logger = setup_logging(log_dir, debug, verbose)
    configuration = load_config(logger, config)

    dataset: Dataset = prepare_dataset(logger, configuration)


def setup_logging(log_dir: str, debug: bool, verbose: bool) -> logging.Logger:
    """Setup Logging Facility.

    Args:
        log_dir (str): Log directory.
        debug (bool): Debug mode.
        verbose (bool): Verbose mode.

    Returns:
        Logger: Logger instance.
    """

    def set_log_level(debug: bool, verbose: bool):
        if debug:
            return logging.DEBUG
        elif verbose:
            return logging.INFO
        else:
            return logging.ERROR

    makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=set_log_level(debug, verbose),
        filename=f"{log_dir}/experiment.log",
        filemode="a",
        format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(logger: logging.Logger, config_path: str) -> Configuration:
    """Load the configuration.

    IMPORTANT: Exit program with exit code '1' if loading confiugration fails.
               See log for further details!

    Args:
        logger (logging.Logger): Logger instance.
        config_path (str): Path to the configuration file.

    Returns:
        config: Configuration instance.
    """
    config = Configuration(config_path)
    if config is None:
        exit(1)
    logger.info("Configuration loaded!")
    logger.debug(f"Configuration: {config.all()}")
    return config


def prepare_dataset(logger: logging.Logger, configuration: Configuration) -> Dataset:
    """Load the dataset.

    IMPORTANT: Exit program with exit code '1' if loading confiugration fails.
               See log for further details!

    Args:
        logger (logging.Logger): Logger instance.
        configuration (Configuration): Configuration instance.

    Returns:
        Dataset: Dataset instance.
    """
    dataset: Dataset
    if configuration.get("dataset.type") == "fashionmnist":
        dataset = FashionMNIST(
            storage_path=configuration.get("dataset.storage_path", "data"),
            transform=None,
            batch_size=configuration.get("dataset.batch_size", 512),
        )
    else:
        logger.error(f"Unknown dataset type: {configuration.get('dataset.type')}")
        exit(1)

    logger.info(f"Dataset '{dataset.name}' loaded")
    return dataset


if __name__ == "__main__":
    main()
