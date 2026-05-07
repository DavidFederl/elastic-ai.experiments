import logging

from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, Softmax
from torch.nn import Sequential as torch_Sequential

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
