import logging
from datetime import datetime
from os import makedirs

import click
from torch import optim
from torch.nn import CrossEntropyLoss

from src.config.configuration import Configuration
from src.nn.data import Dataset, FashionMNIST
from src.nn.model import Sequential, linear_v1
from src.nn.training import Training, TrainingBuilder

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
    model: Sequential = prepare_model(logger, configuration, dataset)
    training: Training = prepare_training(
        logger, configuration, log_dir, dataset, model
    )
    training.train(
        epochs=configuration.get("training.epochs", 100),
        store_only_last_model=configuration.get("training.store_only_last", False),
    )


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

    return dataset


def prepare_model(
    logger: logging.Logger, configuration: Configuration, dataset: Dataset
) -> Sequential:
    """Initialize a model.

    IMPORTANT: Exit program with exit code '1' if loading confiugration fails.
               See log for further details!
    Args:
        logger (logging.Logger): Logger instance.
        configuration (Configuration): Configuration instance.
        dataset (Dataset): Dataset instance.

    Returns:
        Sequential: Sequential model.
    """
    model: Sequential
    if configuration.get("model.type") == "linear_v1":
        ...
        _, model = linear_v1(
            in_features=dataset.element_shape.numel(),
            out_features=len(dataset.classes),
            total_bits=configuration.get("model.fixed-point.total_bits", 16),
            fraction_bits=configuration.get("model.fixed-point.fraction_bits", 8),
        )
    else:
        logger.error(f"Unknown model type: {configuration.get('model.type')}")
        exit(1)

    return model


def prepare_training(
    logger: logging.Logger,
    configuration: Configuration,
    log_dir: str,
    dataset: Dataset,
    model: Sequential,
) -> Training:
    builder = TrainingBuilder().model(model).dataset(dataset).log_dir(log_dir)
    builder.device(configuration.get("training.device", "cpu"))
    match configuration.get("training.loss", ""):
        case "cse":
            builder.loss_fn(CrossEntropyLoss())
    match configuration.get("training.optimizer", ""):
        case "adam":
            builder.optimizer(optim.Adam(model.parameters()))

    try:
        training = builder.build()
        return training
    except ValueError as exc:
        logger.error(f"Error while building Training: {exc}")
        exit(1)


if __name__ == "__main__":
    main()
