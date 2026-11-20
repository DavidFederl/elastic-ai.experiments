import logging
from datetime import datetime, timezone
from pathlib import Path
from random import randint
from typing import Annotated

import torch
import typer
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams

from src.nn.delta_compression import (
    DeltaType,
    Quantizer,
    build_delta_compression,
    compress_state_dict,
    is_parameter,
)
from src.nn.training import set_initial_seed
from src.utils import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    model_configuration: Annotated[Path, typer.Argument(exists=True, readable=True)],
    total_fixed_point_bits: Annotated[int, typer.Option(min=2, max=32)],
    fraction_fixed_point_bits: Annotated[int, typer.Option(min=1)],
    delta_bits: Annotated[int, typer.Option(min=1)],
    delta_type: DeltaType = DeltaType.CONSECUTIVE,
    quantizer: Quantizer = Quantizer.CLAMP,
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
    delta_compression = build_delta_compression(delta_type, quantizer, delta_bits)

    log_dir.joinpath("graphs").mkdir(parents=True, exist_ok=True)
    log_dir.joinpath("states").mkdir(parents=True, exist_ok=True)

    # load state_dict from storage
    logger.info(f"Try loading state_dict from '{model_configuration}'")
    original_state_dict = torch.load(model_configuration, weights_only=True)

    # simulate quantization as INT
    quantized_state_dict = {}
    for layer_id, layer_weights in original_state_dict.items():
        if is_parameter(layer_id):
            quantized_state_dict[layer_id] = fxp_arithmetic.cut_as_integer(
                layer_weights
            ).to(torch.int32)
        else:
            quantized_state_dict[layer_id] = layer_weights
    logger.info("Model quantized")
    get_weight_distribution(quantized_state_dict)

    # simulate compression
    deltas, references = compress_state_dict(quantized_state_dict, delta_compression)
    logger.info("Model compressed")
    get_weight_distribution(deltas, references)


def get_weight_distribution(state_dict: dict, references: dict | None = None) -> None:
    # Compute statistics
    def get_stats(tensor, name):
        if tensor.numel() == 0:
            logger.error(f"{name}: No values")
            return
        reference = references[name] if references is not None else tensor.flatten()[0]
        tensor = tensor.float()
        logger.info(
            f"{name} Statisitcs:"
            f"\n  Shape: {tensor.shape}"
            f"\n  Reference Value: {reference}"
            f"\n  Min: {tensor.min().item():.4f}"
            f"\n  Max: {tensor.max().item():.4f}"
            f"\n  Mean: {tensor.mean().item():.4f}"
            f"\n  Std: {tensor.std().item():.4f}"
            f"\n  Median: {tensor.median().item():.4f}"
            f"\n  25th Percentile: {tensor.quantile(0.25).item():.4f}"
            f"\n  75th Percentile: {tensor.quantile(0.75).item():.4f}"
        )

    for key, tensor in state_dict.items():
        if is_parameter(key):
            get_stats(tensor, key)


if __name__ == "__main__":
    app()
