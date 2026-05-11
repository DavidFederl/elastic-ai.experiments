import logging
from datetime import datetime, timezone
from pathlib import Path
from random import randint
from typing import Annotated

import typer
from elasticai.creator.nn import Sequential
from torch import optim
from torch.nn.modules import CrossEntropyLoss

from src.nn.data import (
    fashionmnist_trainingset_flattened,
    fashionmnist_validationset_flattened,
    get_dataloader,
)
from src.nn.model import linear_v1_eai
from src.nn.training import TrainingBuilder, set_initial_seed
from src.tools.generate_graphs import generate_combined_graphs, generate_graphs
from src.utils import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
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
    total_fixed_point_bits: Annotated[int, typer.Option(min=2)] = 8,
    weight_decay: Annotated[float, typer.Option()] = 0.01,
) -> None:
    setup_logging(log_dir, verbose)
    set_initial_seed(seed, make_determenistic=True)

    training_ds = fashionmnist_trainingset_flattened()
    training_dl = get_dataloader(
        training_ds, batch_size=len(training_ds) if batch_size == 0 else batch_size
    )

    validation_ds = fashionmnist_validationset_flattened()
    validation_dl = get_dataloader(
        validation_ds, batch_size=len(training_ds) if batch_size == 0 else batch_size
    )

    models: dict[str, Sequential] = {}
    for fraction_bits in range(1, total_fixed_point_bits):
        model_id, model = linear_v1_eai(
            in_features=training_ds[0][0].shape.numel(),
            out_features=len(training_ds.classes),
            fixed_point_total_bits=total_fixed_point_bits,
            fixed_point_fraction_bits=fraction_bits,
            bias=False,
        )
        models[model_id] = model

    model_log_dirs: list[Path] = []
    for model_id, model in models.items():
        model_log_dir = log_dir.joinpath(model_id)
        model_log_dirs.append(model_log_dir)

        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=0.001,
            weight_decay=weight_decay,
        )
        loss_fn = CrossEntropyLoss()
        training = (
            TrainingBuilder()
            .dataset(training_ds.classes, training_dl, validation_dl)
            .model(model)
            .device("cpu")
            .optimizer(optimizer)
            .loss_fn(loss_fn)
            .log_dir(model_log_dir)
            .build()
        )
        training.train(epochs=epochs, skip=True)

        generate_graphs(
            input_dir=model_log_dir.joinpath("metrics", "validation"),
            output_dir=model_log_dir.joinpath("graphs", "validation"),
        )
        generate_graphs(
            input_dir=model_log_dir.joinpath("metrics", "training"),
            output_dir=model_log_dir.joinpath("graphs", "training"),
        )

    generate_combined_graphs(
        list(map(lambda dir: dir.joinpath("metrics", "training"), model_log_dirs)),
        log_dir.joinpath("graphs", "training"),
    )
    generate_combined_graphs(
        list(map(lambda dir: dir.joinpath("metrics", "validation"), model_log_dirs)),
        log_dir.joinpath("graphs", "validation"),
    )


if __name__ == "__main__":
    app()
