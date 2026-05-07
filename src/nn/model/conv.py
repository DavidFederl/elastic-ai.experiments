import logging

from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import BatchNormedLinear as LinearEai
from elasticai.creator.nn.fixed_point import HardTanh as TanhEai
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, Softmax
from torch.nn import Sequential as torch_Sequential

from eai_creator.nn.fixed_point.conv2d.layer import BatchNormedSeparableConv2d

logger = logging.getLogger(__name__)


def conv_torch_gupta(
    in_features: int,
    out_features: int,
    bias: bool = True,
) -> tuple[str, torch_Sequential]:
    """Model from  Gupta et al. 2015
    DOI:10.48550/arXiv.1502.02551

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.

    Returns:
        Squence: Sequential model.
    """
    logger.info("Model: Convolutanl Gupta et al. 2015 (PyTorch)")
    logger.debug(f"Model configuration: {in_features=}, {out_features=}")
    return "conv_torch_gupta", torch_Sequential(
        Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), bias=bias),
        MaxPool2d(kernel_size=(2, 2)),
        Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), bias=bias),
        MaxPool2d(kernel_size=(2, 2)),
        Flatten(),
        Linear(in_features=1024, out_features=128, bias=bias),
        Linear(in_features=128, out_features=out_features, bias=bias),
        Softmax(dim=0),
    )


def conv_eai_gupta(
    in_features: int,
    out_features: int,
    fixed_point_total_bits: int,
    fixed_point_fraction_bits: int,
) -> tuple[str, Sequential]:
    """Model from  Gupta et al. 2015
    DOI:10.48550/arXiv.1502.02551

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
    logger.info("Model: Convolutanl Gupta et al. 2015 (elastic-AI.FxP")
    logger.debug(
        f"Model configuration: {in_features=}, {out_features=}, {fixed_point_total_bits=}, {fixed_point_fraction_bits=}"
    )
    return "conv_eai_gupta", Sequential(
        BatchNormedSeparableConv2d(
            in_channels=in_features,
            out_channels=24 * 24,
            signal_length=-1,
            kernel_size=(5, 5),
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        MaxPool2d(
            kernel_size=[2, 2],
        ),
        BatchNormedSeparableConv2d(
            in_channels=12 * 12,
            out_channels=8 * 8,
            signal_length=-1,
            kernel_size=(5, 5),
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        MaxPool2d(
            kernel_size=[2, 2],
        ),
        LinearEai(
            in_features=1024,
            out_features=128,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
        TanhEai(total_bits=fixed_point_total_bits, frac_bits=fixed_point_fraction_bits),
        LinearEai(
            in_features=128,
            out_features=out_features,
            total_bits=fixed_point_total_bits,
            frac_bits=fixed_point_fraction_bits,
        ),
    )
