from typing import Any

from elasticai.creator.arithmetic.fxp_arithmetic import FxpArithmetic
from elasticai.creator.arithmetic.fxp_params import FxpParams
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.math_operations import MathOperations

from eai_creator.base_modules.separable_conv2d import (
    SeparableConv2d as SeparableConv2dBase,
)

type Conv2dDesign = Any
type Conv2dTestbench = Any


class SeparableConv2d(SeparableConv2dBase, DesignCreatorModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        in_channels: int,
        out_channels: int,
        signal_length: int,
        kernel_size: int | tuple[int, int],
        bias: bool = True,
        device: Any = None,
    ) -> None:
        self._params = FxpParams(total_bits=total_bits, frac_bits=frac_bits)
        self._config = FxpArithmetic(self._params)
        self._signal_length = signal_length
        super().__init__(
            operations=MathOperations(config=self._config),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            device=device,
        )

    def create_design(self, name: str) -> Conv2dDesign:
        raise NotImplementedError()

    def create_testbench(self, name: str, uut: Conv2dDesign) -> Conv2dTestbench:
        raise NotImplementedError()
