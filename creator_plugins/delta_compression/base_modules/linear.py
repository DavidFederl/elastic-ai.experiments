from typing import Any

import torch
from elasticai.creator.nn.fixed_point import Linear as LinearEai
from elasticai.creator.nn.fixed_point.linear.design import LinearDesign

from creator_plugins.delta_compression.delta.consecutive import (
    ConsecutiveDeltaCompression,
)


class Linear(LinearEai):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        total_bits: int,
        frac_bits: int,
        delta_bit_width: int,
        bias: bool = True,
        device: Any = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            total_bits=total_bits,
            frac_bits=frac_bits,
            bias=bias,
            device=device,
        )
        self._delta = ConsecutiveDeltaCompression(delta_bit_width, self._params)

    def _simulate_delta_compression(self, data: torch.Tensor) -> torch.Tensor:
        # quantize to correct FXP values as float
        data = self._operations.quantize(data)

        # convert float to INT
        data = data / self._params.minimum_step_as_rational

        # simulate delta-compression
        data = self._delta.compress(data)
        data = self._delta.inflate(data)

        # convert INT to float
        data = data * self._params.minimum_step_as_rational

        return data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._simulate_delta_compression(self.weight)
        bias = (
            None if self.bias is None else self._simulate_delta_compression(self.bias)
        )

        if bias is None:
            return self._operations.matmul(x, weight.T)
        else:
            return self._operations.add(self._operations.matmul(x, weight.T), bias)

    def get_params_delta(self) -> tuple[list[list[int]], list[int]]:
        # TODO: implement!
        raise NotImplementedError("Coming soon!")

    def create_design(self, name: str) -> LinearDesign:
        # TODO: implement!
        raise NotImplementedError("Coming soon!")
