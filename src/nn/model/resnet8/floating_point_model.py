from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class FPModel(nn.Module):
    """ResNet-8 as used in the MLPerf Tiny image classification benchmark."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        widths: tuple[int, ...] = (16, 32, 64),
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        layers = []
        prev_channels = widths[0]
        for i, width in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers.append(ResidualBlock(prev_channels, width, stride=stride))
            prev_channels = width
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(prev_channels, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)
