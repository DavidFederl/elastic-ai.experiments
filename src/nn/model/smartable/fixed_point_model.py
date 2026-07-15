from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.nn.fixed_point import (
    BatchNormedConv1d,
    Conv1d,
    Linear,
    MathOperations,
    ReLU,
)
from torch import Tensor, nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        p_drop: float = 0.2,
        total_bits: int = 8,
        frac_bits: int = 4,
    ):
        super().__init__()
        self.depthwise = Conv1d(
            total_bits=total_bits,
            frac_bits=frac_bits,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            signal_length=-1,
            bias=False,
        )
        self.pointwise = BatchNormedConv1d(
            total_bits=total_bits,
            frac_bits=frac_bits,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            signal_length=-1,
            bias=False,
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.act = ReLU(total_bits=total_bits)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.pool(x)
        x = self.act(x)
        return self.drop(x)


class AdaptiveAvgPool1d(nn.Module):
    """Fixed-point quantized global average pooling (output size 1 per channel).

    elasticai.creator.nn.fixed_point has no pooling layer, so this quantizes
    torch's AdaptiveAvgPool1d output the same way the other fixed_point
    layers quantize theirs (clamp + cut via FxpArithmetic).
    """

    def __init__(self, total_bits: int = 8, frac_bits: int = 4):
        super().__init__()
        config = FxpArithmetic(
            FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
        )
        self._ops = MathOperations(config=config)
        self._pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor) -> Tensor:
        return self._ops.quantize(self._pool(x))


class FxPModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 6,
        widths: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 7,
        p_drop: float = 0.2,
        total_bits: int = 8,
        frac_bits: int = 4,
    ):
        super().__init__()

        layers = []

        prev_channels = in_channels
        for width in widths:
            layers.append(
                DepthwiseSeparableConv(
                    prev_channels,
                    width,
                    kernel_size=kernel_size,
                    p_drop=p_drop,
                    total_bits=total_bits,
                    frac_bits=frac_bits,
                )
            )
            prev_channels = width
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            AdaptiveAvgPool1d(total_bits=total_bits, frac_bits=frac_bits),
            nn.Flatten(),
            Linear(prev_channels, 16, total_bits=total_bits, frac_bits=frac_bits),
            ReLU(total_bits=total_bits),
            nn.Dropout(p_drop),
            Linear(16, num_classes, total_bits=total_bits, frac_bits=frac_bits),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.head(x)
