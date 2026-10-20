import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from random import randint
from typing import Annotated

import typer
from torch import optim
from torch.nn import Module
from torch.nn import Sequential as Sequential_torch
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.nn.data import (
    SmarTable,
    get_dataloader,
)

# from src.nn.model import linear_v1_eai, linear_v1_torch
from src.nn.model.smartable import FPModel, FxPModel, LQModel
from src.nn.training import TrainingBuilder, set_initial_seed
from src.tools.generate_graphs import generate_graphs
from src.utils import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer()

training_ds: SmarTable = SmarTable(
    root=Path("datasets"),
    train=True,
)
training_dl: DataLoader
validation_ds: SmarTable = SmarTable(root=Path("datasets"), train=False)
validation_dl: DataLoader

type Sequential = Sequential_torch | Module


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def __setup_dataloader(batch_size: int) -> None:
    global training_dl, validation_dl
    training_dl = get_dataloader(
        training_ds, batch_size=len(training_ds) if batch_size == 0 else batch_size
    )
    validation_dl = get_dataloader(
        validation_ds, batch_size=len(validation_ds) if batch_size == 0 else batch_size
    )


def __train(
    model: Sequential,
    model_log_dir: Path,
    epochs: int,
    device: Device = Device.CPU,
    early_stopping: tuple[int, float] | None = None,
) -> None:
    global training_ds, training_dl, validation_dl
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=0.001,
    )
    loss_fn = CrossEntropyLoss()
    training_builder = (
        TrainingBuilder()
        .dataset(training_ds.classes, training_dl, validation_dl)
        .model(model)
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
    device: Annotated[Device, typer.Option()] = Device.CPU,
    early_stopping: Annotated[tuple[int, float], typer.Option()] | None = None,
) -> None:
    setup_logging(log_dir, verbose)

    set_initial_seed(seed, make_determenistic=True)
    __setup_dataloader(batch_size=batch_size)
    model = FPModel(
        in_channels=training_ds[0][0].shape[0],
        num_classes=len(training_ds.classes),
        kernel_size=7,
        p_drop=0.2,
    )
    __train(
        model=model,
        model_log_dir=log_dir,
        epochs=epochs,
        device=device,
        early_stopping=early_stopping,
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
    total_bits: Annotated[int, typer.Option(min=1)] = 8,
    fraction_bits: Annotated[int, typer.Option(min=0)] = 4,
    device: Annotated[Device, typer.Option()] = Device.CPU,
    early_stopping: Annotated[tuple[int, float], typer.Option()] | None = None,
) -> None:
    setup_logging(log_dir, verbose)

    set_initial_seed(seed, make_determenistic=True)
    __setup_dataloader(batch_size=batch_size)
    model = FxPModel(
        in_channels=training_ds[0][0].shape[0],
        num_classes=len(training_ds.classes),
        kernel_size=7,
        p_drop=0.2,
        total_bits=total_bits,
        frac_bits=fraction_bits,
    )
    __train(
        model=model,
        model_log_dir=log_dir,
        epochs=epochs,
        device=device,
        early_stopping=early_stopping,
    )
    __generate_graphs(model_log_dir=log_dir)


@app.command()
def linear_quant(
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
    total_bits: Annotated[int, typer.Option(min=1)] = 8,
    early_stopping: Annotated[tuple[int, float], typer.Option()] | None = None,
) -> None:
    setup_logging(log_dir, verbose)

    set_initial_seed(seed, make_determenistic=True)
    __setup_dataloader(batch_size=batch_size)
    model = LQModel(
        in_channels=training_ds[0][0].shape[0],
        num_classes=len(training_ds.classes),
        kernel_size=7,
        p_drop=0.2,
        total_bits=total_bits,
    )
    print(model.state_dict())
    __train(
        model=model,
        model_log_dir=log_dir,
        epochs=epochs,
        early_stopping=early_stopping,
    )
    __generate_graphs(model_log_dir=log_dir)


# @app.command()
# def delta_fxp(
#     delta_type: Annotated[DeltaType, typer.Option()],
#     log_dir: Annotated[Path, typer.Option()] = Path(
#         f"logs/{datetime.now(tz=timezone.utc).timestamp() * 10000}"
#     ),
#     verbose: Annotated[bool, typer.Option(is_flag=True)] = False,
#     epochs: Annotated[int, typer.Option(min=1)] = 100,
#     batch_size: Annotated[
#         int,
#         typer.Option(
#             min=0, help="samples per batch for dataloader (0 == whole dataset)"
#         ),
#     ] = 0,
#     seed: Annotated[int, typer.Option(min=0, max=(2**32) - 1)] = randint(
#         0, (2**32) - 1
#     ),
#     total_fixed_point_bits: Annotated[int, typer.Option(min=1)] = 8,
#     fraction_bits: Annotated[int, typer.Option(min=0)] = 4,
#     delta_bits: Annotated[int, typer.Option(min=1)] = 4,
#     delta_offset: Annotated[int, typer.Option(min=0)] = 2,
# ) -> None:
#     setup_logging(log_dir, verbose)
#
#     set_initial_seed(seed, make_determenistic=True)
#     __setup_dataloader(batch_size=batch_size)
#     delta_compression_builder = DeltaCompBuilder().saturated_compression(
#         delta_width=delta_bits, offset=delta_offset
#     )
#     delta_type(delta_compression_builder)
#
#     model = DeltaModel(
#         in_features=training_ds[0][0].shape.numel(),
#         out_features=len(training_ds.classes),
#         total_bit_width=total_fixed_point_bits,
#         fraction_bit_width=fraction_bits,
#         bias=False,
#         delta_compression=delta_compression_builder.build(),
#     )
#     __train(
#         model=model,
#         model_log_dir=log_dir,
#         epochs=epochs,
#     )
#     __generate_graphs(model_log_dir=log_dir)
#     print(model.state_dict())
#

if __name__ == "__main__":
    app()
