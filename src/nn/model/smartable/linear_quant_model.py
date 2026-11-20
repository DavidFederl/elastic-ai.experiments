from elasticai.creator.nn.integer.avgpooling1dflatten import AVGPooling1dFlatten
from elasticai.creator.nn.integer.batchnorm1d import BatchNorm1d
from elasticai.creator.nn.integer.conv1d import Conv1d
from elasticai.creator.nn.integer.linear import Linear
from elasticai.creator.nn.integer.maxpooling import MaxPooling1d
from elasticai.creator.nn.integer.relu import ReLU
from torch import Tensor, nn

################################################################################
## Custom Layer

### DepthwiseSeparableConv


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_length: int,
        kernel_size: int = 7,
        quant_bits: int = 8,
        p_drop: float = 0.2,
    ):
        super().__init__()
        self.depthwise = Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding="same",
            seq_len=-1,
            name="1_depthwise_conv1d",
            quant_bits=quant_bits,
            device="cpu",
        )
        self.pointwise = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            seq_len=-1,
            name="2_pointwise_conv1d",
            quant_bits=quant_bits,
            device="cpu",
        )
        self.bn = BatchNorm1d(
            norm_dim=out_channels,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            num_dimensions=input_length,
            in_features=out_channels,
            out_features=out_channels,
            name="3_batchnorm1d",
            quant_bits=quant_bits,
            device="cpu",
        )
        self.pool = MaxPooling1d(
            in_features=input_length,
            out_features=input_length // 2,
            in_num_dimensions=out_channels,
            out_num_dimensions=out_channels,
            kernel_size=2,
            name="4_maxpooling1d",
            quant_bits=quant_bits,
            device="cpu",
        )
        self.act = ReLU(
            name="5_relu",
            quant_bits=quant_bits,
            device="cpu",
        )
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x.transpose(1, 2), None).transpose(1, 2)
        x = self.pool(x)
        x = self.act(x)
        return self.drop(x)


################################################################################
## Model


class LQModel(nn.Module):
    def __init__(
        self,
        total_bits: int,
        in_channels: int = 4,
        num_classes: int = 6,
        input_length: int = 1000,
        widths: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 7,
        p_drop: float = 0.2,
    ):
        super().__init__()

        layers = []

        prev_channels = in_channels
        prev_length = input_length
        for width in widths:
            layers.append(
                DepthwiseSeparableConv(
                    in_channels=prev_channels,
                    out_channels=width,
                    input_length=prev_length,
                    kernel_size=kernel_size,
                    quant_bits=total_bits,
                    p_drop=p_drop,
                )
            )
            prev_channels = width
            prev_length //= 2
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            AVGPooling1dFlatten(
                in_features=prev_length,
                out_features=prev_channels,
                in_num_dimensions=prev_channels,
                out_num_dimensions=1,
                name="avg_pool1d_flatten",
                quant_bits=total_bits,
                device="cpu",
            ),
            Linear(
                in_features=prev_channels,
                out_features=16,
                bias=True,
                num_dimensions=1,
                name="linear_1",
                quant_bits=total_bits,
                device="cpu",
            ),
            ReLU(
                name="relu_1",
                quant_bits=total_bits,
                device="cpu",
            ),
            nn.Dropout(p_drop),
            Linear(
                in_features=16,
                out_features=num_classes,
                bias=True,
                num_dimensions=1,
                name="linear_2",
                quant_bits=total_bits,
                device="cpu",
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.transpose(1, 2)
        return self.head(x)
