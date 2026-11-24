import logging
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated

import torch
import torchvision.transforms as transforms
import typer
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.nn.data import get_dataloader
from src.nn.data.cifar10 import CIFAR10_H5, CIFAR10_MEAN, CIFAR10_STD
from src.nn.model.resnet8 import FPModel as ResNet8
from src.nn.training import Training, TrainingBuilder, set_initial_seed
from src.utils import load_config, setup_logging, write_run_info


@dataclass(frozen=True)
class Config:
    """Settings of one training run, loaded from a YAML file."""

    dataset: str
    log_dir: str
    seed: int = 42
    epochs: int = 200
    batch_size: int = 512
    num_workers: int = 0


logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    config_path: Annotated[
        Path, typer.Argument(help="YAML config file", exists=True, dir_okay=False)
    ] = Path(__file__).with_name("config.yaml"),
    seed: Annotated[
        int | None, typer.Option(help="Seed, overrides the seed of the config")
    ] = None,
) -> None:
    """Train ResNet8 on CIFAR10 (FP32)."""
    config = load_config(config_path, Config)
    assert isinstance(config, Config)
    if seed is not None:
        config = replace(config, seed=seed)

    setup_logging(log_dir=Path(config.log_dir), verbose=True)
    write_run_info(Path(config.log_dir), asdict(config))

    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus is not None and int(cpus) != config.num_workers:
        message = (
            f"num_workers is {config.num_workers}, but the job has {cpus} CPUs "
            "(--cpus-per-task). Set both to the same value."
        )
        logger.warning(message)
        typer.secho(f"Warning: {message}", fg=typer.colors.YELLOW, err=True)

    set_initial_seed(seed=config.seed, make_determenistic=True)

    training_set = CIFAR10_H5(
        path=Path(config.dataset),
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        ),
    )
    training_dl: DataLoader = get_dataloader(
        dataset=training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_set = CIFAR10_H5(
        path=Path(config.dataset),
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        ),
    )
    validation_dl: DataLoader = get_dataloader(
        dataset=validation_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model: Module = ResNet8()

    training: Training = (
        TrainingBuilder()
        .model(model=model)
        .dataset(
            classes=training_set.classes,
            training_dl=training_dl,
            validation_dl=validation_dl,
        )
        .loss_fn(loss_fn=CrossEntropyLoss())
        .optimizer(optimizer=Adam(params=model.parameters()))
        .device(device="cuda" if torch.cuda.is_available() else "cpu")
        .log_dir(log_dir=Path(config.log_dir))
        .build()
    )
    training.train(epochs=config.epochs, store_only_last_model=False, skip=False)


if __name__ == "__main__":
    app()
