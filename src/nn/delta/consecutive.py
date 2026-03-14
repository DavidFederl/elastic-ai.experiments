import logging

import torch

logger = logging.getLogger(__name__)


def compress(data: torch.Tensor, bit_width: int) -> torch.Tensor:
    delta: torch.Tensor = data.detach().clone().flatten()

    for index in reversed(range(1, len(delta))):
        delta[index] = delta[index] - delta[index - 1]
    logger.debug(f"{delta.max().item()=}, {delta.min().item()=}")

    min_val = -(2 ** (bit_width - 1))
    max_val = (2 ** (bit_width - 1)) - 1
    logger.debug(f"clamp to [{min_val}, {max_val}]")
    delta[1:].clamp_(min=min_val, max=max_val)

    logger.debug(f"{delta=}")
    return delta.reshape(data.shape)


def inflate(delta: torch.Tensor, bit_width: int) -> torch.Tensor:
    data: torch.Tensor = delta.detach().clone().flatten()

    for index in range(1, len(data)):
        data[index] = data[index - 1] + data[index]
    logger.debug(f"{data.max().item()=}, {data.min().item()=}")

    min_val = -(2 ** (bit_width - 1))
    max_val = (2 ** (bit_width - 1)) - 1
    logger.debug(f"clamp to [{min_val}, {max_val}]")
    data[1:].clamp_(min=min_val, max=max_val)

    logger.debug(f"{data=}")
    return data.reshape(delta.shape)
