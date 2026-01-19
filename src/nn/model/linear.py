import logging

from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import HardTanh as Tanh
from elasticai.creator.nn.fixed_point import Linear

logger = logging.getLogger(__name__)


def linear_v1(
    in_features: int,
    out_features: int,
    fixed_point_total_bits: int,
    fixed_point_fraction_bits: int,
) -> tuple[str, Sequential]:
    """Model consisting of a linear layer followed by a tanh layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        fixed_point_total_bits (int): Total number of bits for fixed point representation.
                                      IMPORTANT: includes sign bit!
        fixed_point_fraction_bits (int): Number of fraction bits for fixed point representation.

    Returns:
        Squence: Sequential model.
    """
    logger.info("Model: Linear v1")
    logger.debug(
        f"Model: Linear v1 configuration: {in_features=}, {out_features=}, {fixed_point_total_bits=}, {fixed_point_fraction_bits=}"
    )
    return "linear_v1", Sequential(
        Linear(
            in_features=in_features,
            out_features=150,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        Tanh(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        Linear(
            in_features=150,
            out_features=16,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        Tanh(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        Linear(
            in_features=16,
            out_features=400,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        Tanh(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        Linear(
            in_features=16 * 5 * 5,
            out_features=120,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        Tanh(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        Linear(
            in_features=120,
            out_features=84,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        Tanh(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        Linear(
            in_features=84,
            out_features=out_features,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
    )
