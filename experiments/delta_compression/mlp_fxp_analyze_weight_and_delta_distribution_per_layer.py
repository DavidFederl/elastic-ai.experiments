import logging
from datetime import datetime, timezone
from pathlib import Path
from random import randint
from typing import Annotated

import torch
import typer
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.nn.delta_compression import DeltaOperations, DeltaType

from src.nn.data import (
    fashionmnist_trainingset_flattened,
)
from src.nn.model import linear_v1_eai
from src.nn.training import set_initial_seed
from src.utils import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    model_configuration: Annotated[Path, typer.Argument(exists=True, readable=True)],
    total_fixed_point_bits: Annotated[int, typer.Option(min=2)],
    fraction_fixed_point_bits: Annotated[int, typer.Option(min=1)],
    delta_bits: Annotated[int, typer.Option(min=1)],
    delta_offset: Annotated[int, typer.Option(min=0)] = 0,
    delta_type: DeltaType = DeltaType.CONSECUTIVE,
    log_dir: Annotated[Path, typer.Option()] = Path(
        f"logs/{int(datetime.now(tz=timezone.utc).timestamp() * 10000)}"
    ),
    verbose: Annotated[bool, typer.Option(is_flag=True)] = False,
    seed: Annotated[int, typer.Option(min=0, max=(2**32) - 1)] = randint(
        0, (2**32) - 1
    ),
) -> None:
    setup_logging(log_dir, verbose)
    set_initial_seed(seed, make_determenistic=True)

    fxp_params = FxpParams(total_fixed_point_bits, fraction_fixed_point_bits)
    fxp_arithmetic = FxpArithmetic(fxp_params)
    delta_operations = DeltaOperations(
        fxp_arithmetic=fxp_arithmetic,
        delta_bits=delta_bits,
        delta_offset=delta_offset,
        delta_type=delta_type,
        clamp=True,
    )

    log_dir.joinpath("graphs").mkdir(parents=True, exist_ok=True)
    log_dir.joinpath("states").mkdir(parents=True, exist_ok=True)

    training_ds = fashionmnist_trainingset_flattened()

    # load model from storage
    _, model = linear_v1_eai(
        in_features=training_ds[0][0].shape.numel(),
        out_features=len(training_ds.classes),
        fixed_point_total_bits=total_fixed_point_bits,
        fixed_point_fraction_bits=fraction_fixed_point_bits,
        bias=False,
    )
    logger.info(f"Try loading state_dict from '{model_configuration}'")
    model.load_state_dict(torch.load(model_configuration, weights_only=True))
    original_state_dict = model.state_dict()

    # simulate quantization as INT
    quantized_state_dict = {}
    for layer_id, layer_weights in original_state_dict.items():
        if "weight" in layer_id or "bias" in layer_id:
            quantized_state_dict[layer_id] = fxp_arithmetic.cut_as_integer(
                layer_weights
            )
        else:
            quantized_state_dict[layer_id] = layer_weights
    logger.info("Model quantized")
    get_weight_distribution(quantized_state_dict)

    # simulate compression
    compressed_state_dict = {}
    for layer_id, layer_weights in original_state_dict.items():
        if "weight" in layer_id or "bias" in layer_id:
            logger.info(f"{layer_id=}")
            compressed_state_dict[layer_id] = delta_operations.compress(layer_weights)
        else:
            compressed_state_dict[layer_id] = layer_weights
    logger.info("Model compressed")
    get_weight_distribution(compressed_state_dict)


def get_weight_distribution(state_dict: dict) -> None:
    # Compute statistics
    def get_stats(tensor, name):
        if tensor.numel() == 0:
            logger.error(f"{name}: No values")
            return
        logger.info(
            f"{name} Statisitcs:"
            f"\n  Shape: {tensor.shape}"
            f"\n  Reference Value: {tensor.flatten()[0]}"
            f"\n  Min: {tensor.min().item():.4f}"
            f"\n  Max: {tensor.max().item():.4f}"
            f"\n  Mean: {tensor.mean().item():.4f}"
            f"\n  Std: {tensor.std().item():.4f}"
            f"\n  Median: {tensor.median().item():.4f}"
            f"\n  25th Percentile: {tensor.quantile(0.25).item():.4f}"
            f"\n  75th Percentile: {tensor.quantile(0.75).item():.4f}"
        )

    for key, tensor in state_dict.items():
        if "weight" in key or "bias" in key:
            get_stats(tensor, key)


if __name__ == "__main__":
    app()
