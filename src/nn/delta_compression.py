import logging
from enum import Enum

import torch
from elasticai.creator.experimental.dc import (
    DeltaCompression,
    DeltaCompressionBuilder,
)
from torch import Tensor

logger = logging.getLogger(__name__)


class DeltaType(str, Enum):
    CONSECUTIVE = "consecutive"
    FIXED_REFERENCE = "fixed-reference"


class Quantizer(str, Enum):
    CLAMP = "clamp"
    CUT = "cut"


def _first_value(data: Tensor) -> Tensor:
    return data[0]


def is_parameter(layer_id: str) -> bool:
    return "weight" in layer_id or "bias" in layer_id


def build_delta_compression(
    delta_type: DeltaType, quantizer: Quantizer, delta_bits: int
) -> DeltaCompression:
    builder = DeltaCompressionBuilder().delta_bit_width(delta_bits)
    match delta_type:
        case DeltaType.CONSECUTIVE:
            builder.consecutive()
        case DeltaType.FIXED_REFERENCE:
            builder.fixed_reference(_first_value)
    match quantizer:
        case Quantizer.CLAMP:
            builder.clamp()
        case Quantizer.CUT:
            builder.cut()
    return builder.build_delta()


def compress_state_dict(
    quantized_state_dict: dict[str, Tensor], delta_compression: DeltaCompression
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Compresses every weight and bias as one flat sequence of integers.

    Returns the deltas and the reference values, both keyed by layer id.
    """
    deltas: dict[str, Tensor] = {}
    references: dict[str, Tensor] = {}
    for layer_id, layer_values in quantized_state_dict.items():
        if is_parameter(layer_id):
            references[layer_id], deltas[layer_id] = delta_compression.compress(
                layer_values.flatten()
            )
    return deltas, references


def inflate_state_dict(
    deltas: dict[str, Tensor],
    references: dict[str, Tensor],
    shapes: dict[str, torch.Size],
    delta_compression: DeltaCompression,
    value_ranges: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Tensor]:
    """Restores the weights and biases from deltas and reference values.

    Deltas wrapped by the `cut` quantizer need the inclusive value range of each layer.
    A layer whose range spans `2**delta_width` values or more cannot be decoded
    unambiguously. It is inflated without a range and can differ from the original.
    """
    inflated: dict[str, Tensor] = {}
    for layer_id, delta in deltas.items():
        value_range = None if value_ranges is None else value_ranges[layer_id]
        if value_range is not None and (
            value_range[1] - value_range[0] >= 2**delta_compression.delta_width
        ):
            logger.warning(
                f"{layer_id}: value range {value_range} spans 2**"
                f"{delta_compression.delta_width} values or more, inflating without it"
            )
            value_range = None
        inflated[layer_id] = delta_compression.inflate(
            references[layer_id], delta, value_range=value_range
        ).reshape(shapes[layer_id])
    return inflated
