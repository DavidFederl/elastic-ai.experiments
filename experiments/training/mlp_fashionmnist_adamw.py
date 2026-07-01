import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from random import randint
from typing import Annotated

import typer
from elasticai.creator.experimental.delta_compression import DeltaCompBuilder
from elasticai.creator.nn.sequential import Sequential as Sequential_creator
from torch import optim
from torch.nn import Module
from torch.nn import Sequential as Sequential_torch
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data import DataLoader

from nn.model.evaluation import DeltaModel, FPModel, FxPModel
from src.nn.data import (
    FashionMNIST,
    fashionmnist_trainingset_flattened,
    fashionmnist_validationset_flattened,
    get_dataloader,
)
from src.nn.training import TrainingBuilder, set_initial_seed
from src.tools.generate_graphs import generate_graphs
from src.utils import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer()

training_ds: FashionMNIST = fashionmnist_trainingset_flattened()
training_dl: DataLoader
validation_ds: FashionMNIST = fashionmnist_validationset_flattened()
validation_dl: DataLoader

type Sequential = Sequential_torch | Sequential_creator | Module


class DeltaType(Enum):
    CONSECUTIVE = "consecutive"
    FIXED_REFERENCE = "fixed_reference"

    def __call__(self, delta_compression_builder: DeltaCompBuilder):
        match self.value:
            case self.CONSECUTIVE.value:
                return delta_compression_builder.consecutive_delta()
            case self.FIXED_REFERENCE.value:
                return delta_compression_builder.fixed_reference_delta()
            case _:
                raise ValueError("Delta Type not found!")


def __setup_dataloader(batch_size: int) -> None:
    global training_dl, validation_dl
    training_dl = get_dataloader(
        training_ds, batch_size=len(training_ds) if batch_size == 0 else batch_size
    )
    validation_dl = get_dataloader(
        validation_ds, batch_size=len(validation_ds) if batch_size == 0 else batch_size
    )


def __train(model: Sequential, model_log_dir: Path, epochs: int, weight_decay: float):
    global training_ds, training_dl, validation_dl
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=0.001,
        weight_decay=weight_decay,
    )
    loss_fn = CrossEntropyLoss()
    training = (
        TrainingBuilder()
        .dataset(training_ds.classes, training_dl, validation_dl)
        .model(model)  # type: ignore
        .device("cpu")
        .optimizer(optimizer)
        .loss_fn(loss_fn)
        .log_dir(model_log_dir)
        .build()
    )
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


@app.command()
def floating_point(
    log_dir: Annotated[Path, typer.Option()] = Path(
        f"logs/{datetime.now(tz=timezone.utc).timestamp() * 10000}"
    ),
    verbose: Annotated[bool, typer.Option(is_flag=True)] = False,
    epochs: Annotated[int, typer.Option(min=1)] = 100,
    batch_size: Annotated[
        int,
        typer.Option(
            min=0, help="samples per batch for dataloader (0 == whole dataset)"
        ),
    ] = 0,
    seed: Annotated[int, typer.Option(min=0, max=(2**32) - 1)] = randint(
        0, (2**32) - 1
    ),
    weight_decay: Annotated[float, typer.Option()] = 0.01,
) -> None:
    setup_logging(log_dir, verbose)

    set_initial_seed(seed, make_determenistic=True)
    __setup_dataloader(batch_size=batch_size)
    model = FPModel(
        in_features=training_ds[0][0].shape.numel(),
        out_features=len(training_ds.classes),
        bias=False,
    )
    __train(
        model=model,
        model_log_dir=log_dir,
        epochs=epochs,
        weight_decay=weight_decay,
    )
    __generate_graphs(model_log_dir=log_dir)


@app.command()
def fixed_point(
    log_dir: Annotated[Path, typer.Option()] = Path(
        f"logs/{datetime.now(tz=timezone.utc).timestamp() * 10000}"
    ),
    verbose: Annotated[bool, typer.Option(is_flag=True)] = False,
    epochs: Annotated[int, typer.Option(min=1)] = 100,
    batch_size: Annotated[
        int,
        typer.Option(
            min=0, help="samples per batch for dataloader (0 == whole dataset)"
        ),
    ] = 0,
    seed: Annotated[int, typer.Option(min=0, max=(2**32) - 1)] = randint(
        0, (2**32) - 1
    ),
    weight_decay: Annotated[float, typer.Option()] = 0.01,
    total_fixed_point_bits: Annotated[int, typer.Option(min=1)] = 8,
    fraction_bits: Annotated[int, typer.Option(min=0)] = 4,
) -> None:
    setup_logging(log_dir, verbose)

    set_initial_seed(seed, make_determenistic=True)
    __setup_dataloader(batch_size=batch_size)
    model = FxPModel(
        in_features=training_ds[0][0].shape.numel(),
        out_features=len(training_ds.classes),
        total_bit_width=total_fixed_point_bits,
        fraction_bit_width=fraction_bits,
        bias=False,
    )
    __train(
        model=model,
        model_log_dir=log_dir,
        epochs=epochs,
        weight_decay=weight_decay,
    )
    __generate_graphs(model_log_dir=log_dir)


@app.command()
def delta(
    delta_type: Annotated[DeltaType, typer.Option()],
    log_dir: Annotated[Path, typer.Option()] = Path(
        f"logs/{datetime.now(tz=timezone.utc).timestamp() * 10000}"
    ),
    verbose: Annotated[bool, typer.Option(is_flag=True)] = False,
    epochs: Annotated[int, typer.Option(min=1)] = 100,
    batch_size: Annotated[
        int,
        typer.Option(
            min=0, help="samples per batch for dataloader (0 == whole dataset)"
        ),
    ] = 0,
    seed: Annotated[int, typer.Option(min=0, max=(2**32) - 1)] = randint(
        0, (2**32) - 1
    ),
    weight_decay: Annotated[float, typer.Option()] = 0.01,
    total_fixed_point_bits: Annotated[int, typer.Option(min=1)] = 8,
    fraction_bits: Annotated[int, typer.Option(min=0)] = 4,
    delta_bits: Annotated[int, typer.Option(min=1)] = 4,
    delta_offset: Annotated[int, typer.Option(min=0)] = 2,
) -> None:
    setup_logging(log_dir, verbose)

    set_initial_seed(seed, make_determenistic=True)
    __setup_dataloader(batch_size=batch_size)
    delta_compression_builder = DeltaCompBuilder().saturated_compression(
        delta_width=delta_bits, offset=delta_offset
    )
    delta_type(delta_compression_builder)
    model = DeltaModel(
        in_features=training_ds[0][0].shape.numel(),
        out_features=len(training_ds.classes),
        total_bit_width=total_fixed_point_bits,
        fraction_bit_width=fraction_bits,
        delta_compression=delta_compression_builder.build(),
        bias=False,
    )
    __train(
        model=model,
        model_log_dir=log_dir,
        epochs=epochs,
        weight_decay=weight_decay,
    )
    __generate_graphs(model_log_dir=log_dir)


if __name__ == "__main__":
    app()
