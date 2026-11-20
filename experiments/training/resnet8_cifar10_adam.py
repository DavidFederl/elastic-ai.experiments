import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from random import randint
from typing import Annotated

import typer
from torch import optim
from torch.nn import Module
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.nn.data import (
    CIFAR10,
    cifar10_trainingset,
    cifar10_validationset,
    get_dataloader,
)
from src.nn.model.resnet8 import FPModel
from src.nn.training import TrainingBuilder, set_initial_seed
from src.tools.generate_graphs import generate_graphs
from src.utils import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer()

training_ds: CIFAR10
training_dl: DataLoader
validation_ds: CIFAR10
validation_dl: DataLoader


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def __setup_dataloader(batch_size: int) -> None:
    global training_ds, training_dl, validation_ds, validation_dl
    training_ds = cifar10_trainingset()
    validation_ds = cifar10_validationset()
    training_dl = get_dataloader(training_ds, batch_size=batch_size, shuffle=True)
    validation_dl = get_dataloader(validation_ds, batch_size=batch_size)


def __train(
    model: Module,
    model_log_dir: Path,
    epochs: int,
    learning_rate: float,
    device: Device = Device.CPU,
    early_stopping: tuple[int, float] | None = None,
) -> None:
    global training_ds, training_dl, validation_dl
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
    )
    loss_fn = CrossEntropyLoss()
    training_builder = (
        TrainingBuilder()
        .dataset(training_ds.classes, training_dl, validation_dl)
        .model(model)  # type: ignore
        .device(device.value)
        .optimizer(optimizer)
        .loss_fn(loss_fn)
        .log_dir(model_log_dir)
    )
    if early_stopping is not None:
        training_builder.early_stopping(
            patience=early_stopping[0], threshold=early_stopping[1]
        )
    training = training_builder.build()
    training.train(epochs=epochs, skip=True)


def __generate_graphs(model_log_dir: Path):
    generate_graphs(
        input_dir=model_log_dir.joinpath("metrics", "validation"),
        output_dir=model_log_dir.joinpath("graphs", "validation"),
    )
    generate_graphs(
        input_dir=model_log_dir.joinpath("metrics", "training"),
        output_dir=model_log_dir.joinpath("graphs", "training"),
    )


@app.callback()
def main() -> None:
    """Train ResNet-8 on CIFAR-10."""


@app.command()
def floating_point(
    log_dir: Annotated[Path, typer.Option()] = Path(
        f"logs/{datetime.now(tz=timezone.utc).timestamp() * 10000}"
    ),
    verbose: Annotated[bool, typer.Option(is_flag=True)] = False,
    epochs: Annotated[int, typer.Option(min=1)] = 100,
    batch_size: Annotated[int, typer.Option(min=1)] = 128,
    learning_rate: Annotated[float, typer.Option(min=0.0)] = 0.001,
    seed: Annotated[int, typer.Option(min=0, max=(2**32) - 1)] = randint(
        0, (2**32) - 1
    ),
    device: Annotated[Device, typer.Option()] = Device.CPU,
    early_stopping: Annotated[tuple[int, float], typer.Option()] | None = None,
) -> None:
    setup_logging(log_dir, verbose)

    set_initial_seed(seed, make_determenistic=True)
    __setup_dataloader(batch_size=batch_size)
    model = FPModel(
        in_channels=training_ds[0][0].shape[0],
        num_classes=len(training_ds.classes),
    )
    __train(
        model=model,
        model_log_dir=log_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        early_stopping=early_stopping,
    )
    __generate_graphs(model_log_dir=log_dir)


if __name__ == "__main__":
    app()
