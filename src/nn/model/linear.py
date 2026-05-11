import logging

from elasticai.creator.nn import Sequential
from elasticai.creator.nn.delta_compression import BatchNormedLinear as LinearDelta
from elasticai.creator.nn.delta_compression.delta_operations import DeltaType
from elasticai.creator.nn.fixed_point import BatchNormedLinear as LinearEai
from elasticai.creator.nn.fixed_point import HardTanh as TanhEai
from torch.nn import BatchNorm1d as BatchNorm1dTorch
from torch.nn import Hardtanh as TanhTorch
from torch.nn import Linear as LinearTorch
from torch.nn import Sequential as SequentialTorch

logger = logging.getLogger(__name__)


def linear_v1_torch(
    in_features: int, out_features: int, bias: bool = True
) -> tuple[str, SequentialTorch]:
    """Model consisting of a linear layer followed by a tanh layer.

    IMPORTANT: FP32 based model!

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bias): if bias is used (default: True).

    Returns:
        Squence: Sequential model.
    """
    logger.info("Model: Linear v1 (PyTorch)")
    logger.debug(f"Model: Linear v1 configuration: {in_features=}, {out_features=}")
    return "linear_v1_torch", SequentialTorch(
        LinearTorch(in_features=in_features, out_features=150, bias=bias),
        BatchNorm1dTorch(150),
        TanhTorch(),
        LinearTorch(in_features=150, out_features=16, bias=bias),
        BatchNorm1dTorch(16),
        TanhTorch(),
        LinearTorch(in_features=16, out_features=400, bias=bias),
        BatchNorm1dTorch(400),
        TanhTorch(),
        LinearTorch(in_features=400, out_features=120, bias=bias),
        BatchNorm1dTorch(120),
        TanhTorch(),
        LinearTorch(in_features=120, out_features=84, bias=bias),
        BatchNorm1dTorch(84),
        TanhTorch(),
        LinearTorch(in_features=84, out_features=out_features, bias=bias),
    )


def linear_v1_eai(
    in_features: int,
    out_features: int,
    fixed_point_total_bits: int,
    fixed_point_fraction_bits: int,
    bias: bool = True,
) -> tuple[str, Sequential]:
    """Model consisting of a linear layer followed by a tanh layer.

    IMPORTANT: Fixed Point based model!

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        fixed_point_total_bits (int): Total number of bits for fixed point representation.
                                      IMPORTANT: includes sign bit!
        fixed_point_fraction_bits (int): Number of fraction bits for fixed point representation.
        bias (bias): if bias is used (default: True).

    Returns:
        Squence: Sequential model.
    """
    logger.info("Model: Linear v1 (elastic-AI)")
    logger.debug(
        f"Model: Linear v1 configuration: {in_features=}, {out_features=}, {fixed_point_total_bits=}, {fixed_point_fraction_bits=}"
    )
    return (
        f"linear_v1_q{fixed_point_total_bits - fixed_point_fraction_bits - 1}.{fixed_point_fraction_bits}",
        Sequential(
            LinearEai(
                in_features=in_features,
                out_features=150,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                bias=bias,
            ),
            TanhEai(
                total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits
            ),
            LinearEai(
                in_features=150,
                out_features=16,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                bias=bias,
            ),
            TanhEai(
                total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits
            ),
            LinearEai(
                in_features=16,
                out_features=400,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                bias=bias,
            ),
            TanhEai(
                total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits
            ),
            LinearEai(
                in_features=400,
                out_features=120,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                bias=bias,
            ),
            TanhEai(
                total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits
            ),
            LinearEai(
                in_features=120,
                out_features=84,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                bias=bias,
            ),
            TanhEai(
                total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits
            ),
            LinearEai(
                in_features=84,
                out_features=out_features,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                bias=bias,
            ),
        ),
    )


def linear_v1_delta(
    in_features: int,
    out_features: int,
    fixed_point_total_bits: int,
    fixed_point_fraction_bits: int,
    delta_bit_width: int,
    delta_offset: int,
    delta_type: DeltaType,
    bias: bool = True,
) -> tuple[str, Sequential]:
    """Model consisting of a linear layer followed by a tanh layer.

    IMPORTANT: Fixed Point based model!

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        fixed_point_total_bits (int): Total number of bits for fixed point representation.
                                      IMPORTANT: includes sign bit!
        fixed_point_fraction_bits (int): Number of fraction bits for fixed point representation.
        bias (bias): if bias is used (default: True).
        delta_bit_width (int): total delta width
        delta_offset (int): bits to move delta from LSB
        delta_type (DeltaType): delta calculationa algorithm to use

    Returns:
        Squence: Sequential model.
    """
    logger.info("Model: Linear v1 (elastic-AI DELTA)")
    logger.debug(
        f"Model: Linear v1 configuration: {in_features=}, {out_features=}, {fixed_point_total_bits=}, {fixed_point_fraction_bits=}, {delta_bit_width=}"
    )
    return (
        f"linear_v1_q{fixed_point_total_bits - fixed_point_fraction_bits - 1}.{fixed_point_fraction_bits}_d{delta_bit_width}o{delta_offset}",
        Sequential(
            LinearDelta(
                in_features=in_features,
                out_features=150,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                delta_bits=delta_bit_width,
                delta_offset=delta_offset,
                delta_type=delta_type,
                clamp=True,
                bias=bias,
            ),
            TanhEai(
                total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits
            ),
            LinearDelta(
                in_features=150,
                out_features=16,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                delta_bits=delta_bit_width,
                delta_offset=delta_offset,
                delta_type=delta_type,
                clamp=True,
                bias=bias,
            ),
            TanhEai(
                total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits
            ),
            LinearDelta(
                in_features=16,
                out_features=400,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                delta_bits=delta_bit_width,
                delta_offset=delta_offset,
                delta_type=delta_type,
                clamp=True,
                bias=bias,
            ),
            TanhEai(
                total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits
            ),
            LinearDelta(
                in_features=400,
                out_features=120,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                delta_bits=delta_bit_width,
                delta_offset=delta_offset,
                delta_type=delta_type,
                clamp=True,
                bias=bias,
            ),
            TanhEai(
                total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits
            ),
            LinearDelta(
                in_features=120,
                out_features=84,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                delta_bits=delta_bit_width,
                delta_offset=delta_offset,
                delta_type=delta_type,
                clamp=True,
                bias=bias,
            ),
            TanhEai(
                total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits
            ),
            LinearDelta(
                in_features=84,
                out_features=out_features,
                total_bits=fixed_point_total_bits,
                frac_bits=fixed_point_fraction_bits,
                delta_bits=delta_bit_width,
                delta_offset=delta_offset,
                delta_type=delta_type,
                clamp=True,
                bias=bias,
            ),
        ),
    )
