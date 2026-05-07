from typing import Any, Protocol

from elasticai.creator.base_modules.math_operations import Quantize
from torch import Tensor
from torch.nn import Conv2d as torch_Conv2d
from torch.nn.functional import conv1d


class MathOperations(Quantize, Protocol): ...


class SeparableConv2d(torch_Conv2d):
    def __init__(
        self,
        operations: MathOperations,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )
        self._operations = operations

    def forward(self, input: Tensor) -> Tensor:
        quantized_weights = self._operations.quantize(self.weight)
        quantized_bias = (
            self._operations.quantize(self.bias) if self.bias is not None else None
        )

        # x shape: (batch_size, in_channels, height, width)
        batch_size, C, H, W = (
            input.shape
        )  # FIXME: ValueError: too many values to unpack (expected 4)

        # 1. Convolution über die Zeilen (horizontal)
        # Reshape: (batch_size * height, in_channels, width)
        input_row = input.permute(0, 2, 1, 3).reshape(batch_size * H, C, W)
        input_row = conv1d(
            input=input_row,
            weight=quantized_weights,
            bias=quantized_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Reshape back: (batch_size, height, out_channels, width)
        input_row = input_row.reshape(batch_size, H, -1, W).permute(0, 2, 1, 3)

        # 2. Convolution über die Spalten (vertical)
        # Reshape: (batch_size * width, out_channels, height)
        input_col = input_row.permute(0, 3, 2, 1).reshape(batch_size * W, -1, H)
        input_col = conv1d(
            input=input_col,
            weight=quantized_weights,
            bias=quantized_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Reshape back: (batch_size, out_channels, height, width)
        input_col = input_col.reshape(batch_size, W, -1, H).permute(0, 2, 3, 1)

        return self._operations.quantize(input_col)
