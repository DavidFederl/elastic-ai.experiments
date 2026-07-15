from torch import Tensor, nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, p_drop=0.2):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding="same",
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.act(x)
        return self.drop(x)


class FPModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 6,
        widths: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 7,
        p_drop: float = 0.2,
    ):
        super().__init__()

        layers = []

        prev_channels = in_channels
        for width in widths:
            layers.append(
                DepthwiseSeparableConv(
                    prev_channels, width, kernel_size=kernel_size, p_drop=p_drop
                )
            )
            prev_channels = width
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(prev_channels, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.head(x)
