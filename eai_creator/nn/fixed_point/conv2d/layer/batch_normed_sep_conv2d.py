from typing import Any

from elasticai.creator.arithmetic.fxp_arithmetic import FxpArithmetic
from elasticai.creator.arithmetic.fxp_params import FxpParams
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.math_operations import MathOperations
from torch import Tensor
from torch.nn import BatchNorm2d

from eai_creator.base_modules.separable_conv2d import (
    SeparableConv2d as Conv2dBase,
)

type Conv2dDesign = Any
type Conv2dTestbench = Any


class BatchNormedSeparableConv2d(DesignCreatorModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        in_channels: int,
        out_channels: int,
        signal_length: int,
        kernel_size: int | tuple[int, int],
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        bn_affine: bool = True,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        device: Any = None,
    ) -> None:
        super().__init__()
        self._config = FxpArithmetic(
            FxpParams(total_bits=total_bits, frac_bits=frac_bits)
        )
        self._operations = MathOperations(config=self._config)
        self._signal_length = signal_length
        self._conv2d = Conv2dBase(
            operations=self._operations,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            device=device,
        )

        self._batch_norm = BatchNorm2d(
            num_features=out_channels,
            eps=bn_eps,
            momentum=bn_momentum,
            affine=bn_affine,
            track_running_stats=True,
            device=device,
        )

    @property
    def conv_weight(self) -> Tensor:
        return self._conv2d.weight

    @property
    def conv_bias(self) -> Tensor | None:
        return self._conv2d.bias

    @property
    def bn_weight(self) -> Tensor:
        return self._batch_norm.weight

    @property
    def bn_bias(self) -> Tensor:
        return self._batch_norm.bias

    def forward(self, input: Tensor) -> Tensor:
        has_batches = input.dim() == 3

        if not has_batches:
            input = input.view(1, *input.shape)

        input = self._conv2d(input)
        input = self._batch_norm(input)
        input = self._operations.quantize(input)

        if not has_batches:
            input = input.squeeze(dim=0)

        return input

    def create_design(self, name: str) -> Conv2dDesign:
        raise NotImplementedError()

    def create_testbench(self, name: str, uut: Conv2dDesign) -> Conv2dTestbench:
        raise NotImplementedError()
