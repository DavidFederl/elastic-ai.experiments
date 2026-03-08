import logging

import torch

logger = logging.getLogger(__name__)


def compress(data: torch.Tensor, bit_width: int) -> torch.Tensor:
    delta: torch.Tensor = data.detach().clone().flatten()

    max_delta: float = 0
    min_delta: float = 0
    for index in reversed(range(1, len(delta))):
        diff = (delta[index] - delta[index - 1]).item()
        if diff > max_delta:
            max_delta = diff
        if diff < min_delta:
            min_delta = diff
        delta[index] = diff
    logger.debug(f"{max_delta=}, {min_delta=}")

    min_val = -(2 ** (bit_width - 1))
    max_val = (2 ** (bit_width - 1)) - 1
    delta[1:].clamp_(min=min_val, max=max_val)
    logger.debug(f"{delta=}")

    return delta.reshape(data.shape)


def inflate(delta: torch.Tensor, bit_width: int) -> torch.Tensor:
    data: torch.Tensor = delta.detach().clone().flatten()

    for index in range(1, len(data)):
        data[index] = data[index - 1] + data[index]

    min_val = -(2 ** (bit_width - 1))
    max_val = (2 ** (bit_width - 1)) - 1
    data[1:].clamp_(min=min_val, max=max_val)

    return data.reshape(delta.shape)
