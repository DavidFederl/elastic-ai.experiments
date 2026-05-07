import logging
import pprint
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import torch
import typer
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    PerAxis,
    quantize_,
)
from torchao.quantization.qat import QATConfig
from torchao.quantization.qat.api import QATStep

from src.nn.data import (
    fashionmnist_trainingset_flattened,
    fashionmnist_validationset_flattened,
    get_dataloader,
)
from src.nn.model import linear_v1_torch
from src.nn.training import TrainingBuilder, set_initial_seed
from src.tools.generate_graphs import generate_graphs
from src.utils import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    log_dir: Annotated[Path, typer.Option(help="Log directory.")] = Path(
        f"logs/{int(datetime.now(timezone.utc).timestamp() * 10000)}"
    ),
    verbose: Annotated[
        bool, typer.Option(help="Enable verbose output.", is_flag=True)
    ] = False,
    epochs: Annotated[int, typer.Option(help="Epochs for Training.", min=1)] = 100,
    seed: Annotated[int | None, typer.Option(min=0)] = None,
):
    setup_logging(log_dir, verbose)

    if seed is not None:
        set_initial_seed(seed, make_determenistic=False)

    # transformations: transforms.Compose = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,)),
    #     ]
    # )
    # training_ds = FashionMNIST(root="datasets", train=True, transform=transformations)
    training_ds = fashionmnist_trainingset_flattened()
    training_dl = get_dataloader(training_ds)

    # validation_ds = FashionMNIST(root="datasets", train=False, transform=transformations)
    validation_ds = fashionmnist_validationset_flattened()
    validation_dl = get_dataloader(validation_ds)

    _, model = linear_v1_torch(
        in_features=training_ds[0][0].shape.numel(),
        out_features=len(training_ds.classes),
        bias=False,
    )

    base_config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int8,
        weight_granularity=PerAxis(0),
    )
    quantize_(model, QATConfig(base_config, step=QATStep("prepare")))

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=0.001,
    )
    loss_fn = CrossEntropyLoss()
    training = (
        TrainingBuilder()
        .dataset(training_ds.classes, training_dl, validation_dl)
        .model(model)  # type: ignore
        .device("cpu")
        .optimizer(optimizer)
        .loss_fn(loss_fn)
        .log_dir(log_dir)
        .build()
    )
    training.train(epochs=epochs, skip=True)

    quantize_(model, QATConfig(base_config, step=QATStep("convert")))

    logger.info(f"QAT model parameter: {pprint.pformat(model.state_dict())}")

    generate_graphs(
        input_dir=log_dir.joinpath("metrics", "validation"),
        output_dir=log_dir.joinpath("graphs", "validation"),
    )


if __name__ == "__main__":
    app()
