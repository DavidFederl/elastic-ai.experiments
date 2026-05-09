import logging
from datetime import datetime, timezone
from pathlib import Path
from random import randint
from typing import Annotated

import torch
import typer
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from torch.nn.modules.loss import CrossEntropyLoss

from src.nn.data import (
    fashionmnist_validationset_flattened,
    get_dataloader,
)
from src.nn.delta import compress_consecutive, inflate_consecutive
from src.nn.model import linear_v1_eai
from src.nn.training import Metrics, MetricWriter, set_initial_seed
from src.utils import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    model_configuration: Annotated[Path, typer.Argument(exists=True, readable=True)],
    total_fixed_point_bits: Annotated[int, typer.Option(min=2)],
    fraction_fixed_point_bits: Annotated[int, typer.Option(min=1)],
    delta_bits: Annotated[int, typer.Option(min=1)],
    log_dir: Annotated[Path, typer.Option()] = Path(
        f"logs/{int(datetime.now(tz=timezone.utc).timestamp() * 10000)}"
    ),
    verbose: Annotated[bool, typer.Option(is_flag=True)] = False,
    seed: Annotated[int, typer.Option(min=0, max=(2**32) - 1)] = randint(
        0, (2**32) - 1
    ),
):
    setup_logging(log_dir, verbose)
    set_initial_seed(seed, make_determenistic=True)

    validation_ds = fashionmnist_validationset_flattened()
    validation_dl = get_dataloader(validation_ds, batch_size=len(validation_ds))

    _, model = linear_v1_eai(
        in_features=validation_ds[0][0].shape.numel(),
        out_features=len(validation_ds.classes),
        fixed_point_total_bits=total_fixed_point_bits,
        fixed_point_fraction_bits=fraction_fixed_point_bits,
        bias=False,
    )
    model.load_state_dict(torch.load(model_configuration, weights_only=True))
    logger.info(f"Model loaded from {model_configuration}")

    metrics_dir = log_dir.joinpath("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_writer = MetricWriter(metrics_dir)
    metrics = Metrics(CrossEntropyLoss(), len(validation_ds.classes))

    metrics.reset()
    with torch.no_grad():
        for data, labels in validation_dl:
            outputs = model(data)
            metrics.add(labels, outputs)
    metrics_writer.write(metrics, filename="original_model.json")

    original_state_dict = model.state_dict()

    fxp_params = FxpParams(total_fixed_point_bits, fraction_fixed_point_bits)
    fxp_arithemetic = FxpArithmetic(fxp_params)

    quantized_state_dict = {}
    for layer_id, layer_weights in original_state_dict.items():
        if "weight" in layer_id or "bias" in layer_id:
            quantized_state_dict[layer_id] = fxp_arithemetic.cut_as_integer(
                layer_weights
            )
        else:
            quantized_state_dict[layer_id] = layer_weights
    logger.info("Model quantized")

    compressed_state_dict = {}
    for layer_id, layer_weights in quantized_state_dict.items():
        if "weight" in layer_id or "bias" in layer_id:
            logger.info(f"{layer_id=}")
            compressed_state_dict[layer_id] = compress_consecutive(
                layer_weights, delta_bits
            )
        else:
            compressed_state_dict[layer_id] = layer_weights
    logger.info("Model compressed")

    inflated_state_dict = {}
    for layer_id, layer_weights in compressed_state_dict.items():
        if "weight" in layer_id or "bias" in layer_id:
            inflated_state_dict[layer_id] = inflate_consecutive(
                layer_weights, total_fixed_point_bits
            )
        else:
            inflated_state_dict[layer_id] = layer_weights
    logger.info("Model inflated")

    reconstructed_state_dict = {}
    for layer_id, layer_weights in inflated_state_dict.items():
        if "weight" in layer_id or "bias" in layer_id:
            reconstructed_state_dict[layer_id] = fxp_arithemetic.as_rational(
                layer_weights
            )
        else:
            reconstructed_state_dict[layer_id] = layer_weights
    logger.info("Model reconstructed")

    model.load_state_dict(reconstructed_state_dict)
    logger.info("Delta Compressed model loaded")

    metrics.reset()
    with torch.no_grad():
        for data, labels in validation_dl:
            outputs = model(data)
            metrics.add(labels, outputs)
    metrics_writer.write(metrics, filename="after_delta_compression.json")

    # TODO: show metrics


if __name__ == "__main__":
    app()
