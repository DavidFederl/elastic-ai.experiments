import logging

from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import HardTanh as TanhEai
from elasticai.creator.nn.fixed_point import Linear as LinearEai
from torch.nn import Hardtanh as TanhTorch
from torch.nn import Linear as LinearTorch
from torch.nn import Sequential as SequentialTorch

from creator_plugins.delta_compression import (
    Linear as LinearDelta,
)

logger = logging.getLogger(__name__)


def linear_v1_torch(
    in_features: int,
    out_features: int,
) -> tuple[str, SequentialTorch]:
    """Model consisting of a linear layer followed by a tanh layer.

    IMPORTANT: FP32 based model!

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.

    Returns:
        Squence: Sequential model.
    """
    logger.info("Model: Linear v1 (PyTorch)")
    logger.debug(f"Model: Linear v1 configuration: {in_features=}, {out_features=}")
    return "linear_v1", SequentialTorch(
        LinearTorch(in_features=in_features, out_features=150),
        TanhTorch(),
        LinearTorch(in_features=150, out_features=16),
        TanhTorch(),
        LinearTorch(in_features=16, out_features=400),
        TanhTorch(),
        LinearTorch(in_features=400, out_features=120),
        TanhTorch(),
        LinearTorch(in_features=120, out_features=84),
        TanhTorch(),
        LinearTorch(in_features=84, out_features=out_features),
    )


def linear_v1_eai(
    in_features: int,
    out_features: int,
    fixed_point_total_bits: int,
    fixed_point_fraction_bits: int,
) -> tuple[str, Sequential]:
    """Model consisting of a linear layer followed by a tanh layer.

    IMPORTANT: Fixed Point based model!

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        fixed_point_total_bits (int): Total number of bits for fixed point representation.
                                      IMPORTANT: includes sign bit!
        fixed_point_fraction_bits (int): Number of fraction bits for fixed point representation.

    Returns:
        Squence: Sequential model.
    """
    logger.info("Model: Linear v1 (elastic-AI)")
    logger.debug(
        f"Model: Linear v1 configuration: {in_features=}, {out_features=}, {fixed_point_total_bits=}, {fixed_point_fraction_bits=}"
    )
    return "linear_v1", Sequential(
        LinearEai(
            in_features=in_features,
            out_features=150,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearEai(
            in_features=150,
            out_features=16,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearEai(
            in_features=16,
            out_features=400,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearEai(
            in_features=400,
            out_features=120,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearEai(
            in_features=120,
            out_features=84,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearEai(
            in_features=84,
            out_features=out_features,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
    )


def linear_v1_delta(
    in_features: int,
    out_features: int,
    fixed_point_total_bits: int,
    fixed_point_fraction_bits: int,
    delta_bit_width: int,
) -> tuple[str, Sequential]:
    """Model consisting of a linear layer followed by a tanh layer.

    IMPORTANT: Fixed Point based model!

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        fixed_point_total_bits (int): Total number of bits for fixed point representation.
                                      IMPORTANT: includes sign bit!
        fixed_point_fraction_bits (int): Number of fraction bits for fixed point representation.

    Returns:
        Squence: Sequential model.
    """
    logger.info("Model: Linear v1 (elastic-AI DELTA)")
    logger.debug(
        f"Model: Linear v1 configuration: {in_features=}, {out_features=}, {fixed_point_total_bits=}, {fixed_point_fraction_bits=}, {delta_bit_width=}"
    )
    return "linear_v1", Sequential(
        LinearDelta(
            in_features=in_features,
            out_features=150,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
            delta_bit_width=delta_bit_width,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearDelta(
            in_features=150,
            out_features=16,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
            delta_bit_width=delta_bit_width,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearDelta(
            in_features=16,
            out_features=400,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
            delta_bit_width=delta_bit_width,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearDelta(
            in_features=400,
            out_features=120,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
            delta_bit_width=delta_bit_width,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearDelta(
            in_features=120,
            out_features=84,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
            delta_bit_width=delta_bit_width,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearDelta(
            in_features=84,
            out_features=out_features,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
            delta_bit_width=delta_bit_width,
        ),
    )
