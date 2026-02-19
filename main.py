import logging
from datetime import datetime
from os import makedirs
from pathlib import Path
from types import FunctionType
from typing import Annotated

import typer
from torch import optim
from torch.nn import CrossEntropyLoss

import src.experiments
import src.nn.data
import src.nn.model
from src.config import Configuration
from src.experiments import Experiment
from src.nn.data import Dataset
from src.nn.model import Sequential
from src.nn.training import Training, TrainingBuilder

logger: logging.Logger
configuration: Configuration


def main(
    config: Annotated[
        Path,
        typer.Option(
            help="Configuration file defining the experiment.",
            exists=True,
            file_okay=True,
            writable=True,
            readable=True,
        ),
    ],
    log_dir: Annotated[Path, typer.Option(help="Log directory.")] = Path(
        f"logs/{int(datetime.now().timestamp() * 1000)}"
    ),
    debug: Annotated[bool, typer.Option(help="Enable debug mode.")] = False,
    verbose: Annotated[bool, typer.Option(help="Enable verbose mode.")] = False,
):
    logger: logging.Logger = setup_logging(log_dir, debug, verbose)
    configuration: Configuration = load_config(logger, config)
    configuration.save(log_dir.joinpath("config.yaml"))

    dataset: Dataset = prepare_dataset(logger, configuration)
    model: Sequential = prepare_model(logger, configuration, dataset)
    training: Training = prepare_training(
        logger, configuration, log_dir, dataset, model
    )
    training.train(
        epochs=configuration.get("training.epochs", 100),
        store_only_last_model=configuration.get("training.store_only_last", False),
    )

    experiment: Experiment = prepare_experiemt(logger, configuration, log_dir)
    experiment.run(model=model, dataset=dataset)


def setup_logging(log_dir: Path, debug: bool, verbose: bool) -> logging.Logger:
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
        filename=log_dir.joinpath("experiment.log"),
        filemode="a",
        format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(logger: logging.Logger, config_file: Path) -> Configuration:
    """Load the configuration.

    IMPORTANT: Exit program with exit code '1' if loading confiugration fails.
               See log for further details!

    Args:
        logger (logging.Logger): Logger instance.
        config_path (str): Path to the configuration file.

    Returns:
        config: Configuration instance.
    """
    config = Configuration(config_file)
    if config.configuration is None:
        logger.error("Configuration not loaded!")
        exit(1)
    logger.info("Configuration loaded!")
    logger.debug(f"Configuration: {config.get_all()}")
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
    dataset_name = configuration.get("dataset.type", "")
    if dataset_name == "":
        logger.error("Dataset type not specified!")
        exit(1)

    dataset = getattr(src.nn.data, dataset_name, None)
    dataset_params = configuration.get("dataset.parameter", {})

    if dataset is None or not isinstance(dataset, type(Dataset)):
        logger.error(f"Unknown dataset type: {dataset_name}")
        exit(1)

    return dataset(**dataset_params)


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
    model_name = configuration.get("model.type", "")
    if model_name == "":
        logger.error("Model not specified!")
        exit(1)

    model_builder = getattr(src.nn.model, model_name, None)
    model_params = configuration.get("model.parameter", {})

    if model_builder is None or not isinstance(model_builder, FunctionType):
        logger.error(f"Unknown model type: {model_name}")
        exit(1)

    _, model = model_builder(
        in_features=dataset.element_shape.numel(),
        out_features=len(dataset.classes),
        **model_params,
    )

    return model


def prepare_training(
    logger: logging.Logger,
    configuration: Configuration,
    log_dir: Path,
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


def prepare_experiemt(
    logger: logging.Logger, configuration: Configuration, log_dir: Path
) -> Experiment:
    """Instantiate Experiment.

    IMPORTANT: Exit program with exit code '1' if loading confiugration fails.
               See log for further details!

    Args:
        logger (logging.Logger): Logger instance.
        configuration (Configuration): Configuration instance.
        log_dir (str): Log directory.

    Returns:
        Dataset: Dataset instance.
    """
    experiment_name = configuration.get("experiment.type", "")
    if experiment_name == "":
        logger.error("Experiment not specified!")
        exit(1)

    experiment_params = configuration.get("experiment.parameter", {})

    experiment = getattr(src.experiments, experiment_name, None)

    if experiment is None or not isinstance(experiment, type(Experiment)):
        logger.error(f"Unknown Experiment type: {experiment_name}")
        exit(1)

    return experiment(log_dir=log_dir.joinpath("experiment"), config=configuration)


if __name__ == "__main__":
    typer.run(main)
