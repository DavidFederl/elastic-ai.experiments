import logging

from elasticai.creator.base_modules.linear import Linear as Linear_Base
from elasticai.creator.experimental.delta_compression import (
    DeltaCompression,
    delta_compressed_bias,
    delta_compressed_weights,
)
from elasticai.creator.nn.fixed_point import BatchNormedLinear as Linear_FxP
from elasticai.creator.nn.fixed_point import HardTanh as HardTanh_FxP
from torch import Tensor, no_grad
from torch.nn import BatchNorm1d as BatchNorm_FP
from torch.nn import Hardtanh as HardTanh_FP
from torch.nn import Linear as Linear_FP
from torch.nn import Module, Sequential

logger = logging.getLogger(__name__)


class FPModel(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias
        self._setup_model()

    def _setup_model(self) -> None:
        self.model = Sequential(
            Linear_FP(in_features=self._in_features, out_features=150, bias=self._bias),
            BatchNorm_FP(num_features=150),
            HardTanh_FP(),
            Linear_FP(in_features=150, out_features=16, bias=self._bias),
            BatchNorm_FP(num_features=16),
            HardTanh_FP(),
            Linear_FP(in_features=16, out_features=400, bias=self._bias),
            BatchNorm_FP(num_features=400),
            HardTanh_FP(),
            Linear_FP(in_features=400, out_features=120, bias=self._bias),
            BatchNorm_FP(num_features=120),
            HardTanh_FP(),
            Linear_FP(in_features=120, out_features=84, bias=self._bias),
            BatchNorm_FP(num_features=84),
            HardTanh_FP(),
            Linear_FP(in_features=84, out_features=self._out_features, bias=self._bias),
            BatchNorm_FP(num_features=self._out_features),
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)


class FxPModel(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        total_bit_width: int,
        fraction_bit_width: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._total_bits = total_bit_width
        self._fraction_bits = fraction_bit_width
        self._bias = bias
        self._setup_model()

    def _setup_model(self) -> None:
        self.model = Sequential(
            Linear_FxP(
                in_features=self._in_features,
                out_features=150,
                total_bits=self._total_bits,
                frac_bits=self._fraction_bits,
                bias=self._bias,
            ),
            HardTanh_FxP(total_bits=self._total_bits, frac_bits=self._fraction_bits),
            Linear_FxP(
                in_features=150,
                out_features=16,
                total_bits=self._total_bits,
                frac_bits=self._fraction_bits,
                bias=self._bias,
            ),
            HardTanh_FxP(total_bits=self._total_bits, frac_bits=self._fraction_bits),
            Linear_FxP(
                in_features=16,
                out_features=400,
                total_bits=self._total_bits,
                frac_bits=self._fraction_bits,
                bias=self._bias,
            ),
            HardTanh_FxP(total_bits=self._total_bits, frac_bits=self._fraction_bits),
            Linear_FxP(
                in_features=400,
                out_features=120,
                total_bits=self._total_bits,
                frac_bits=self._fraction_bits,
                bias=self._bias,
            ),
            HardTanh_FxP(total_bits=self._total_bits, frac_bits=self._fraction_bits),
            Linear_FxP(
                in_features=120,
                out_features=84,
                total_bits=self._total_bits,
                frac_bits=self._fraction_bits,
                bias=self._bias,
            ),
            HardTanh_FxP(total_bits=self._total_bits, frac_bits=self._fraction_bits),
            Linear_FxP(
                in_features=84,
                out_features=self._out_features,
                total_bits=self._total_bits,
                frac_bits=self._fraction_bits,
                bias=self._bias,
            ),
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)


class DeltaModel(FxPModel):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        total_bit_width: int,
        fraction_bit_width: int,
        delta_compression: DeltaCompression,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_features, out_features, total_bit_width, fraction_bit_width, bias
        )
        self._delta_compression = delta_compression

        @delta_compressed_weights(self._delta_compression)
        @delta_compressed_bias(self._delta_compression)
        class DeltaLinear(Linear_Base):
            pass

        def to_delta_linear(original: Linear_Base) -> Linear_Base:
            replacement = DeltaLinear(
                in_features=original.in_features,
                out_features=original.out_features,
                operations=original._operations,
                bias=(original.bias is not None),
            )
            with no_grad():
                replacement._original_weight.copy_(original.weight)
                if original.bias is not None:
                    replacement._original_bias.copy_(original.bias)
            return replacement

        # The delta compressed weights only take effect for the module that is
        # actually called during the forward pass, so the Linear_Base instances
        # have to be swapped out inside their owning parent module.
        replacements = 0
        for parent in list(self.model.modules()):
            for name, child in list(parent.named_children()):
                if not isinstance(child, Linear_Base) or isinstance(child, DeltaLinear):
                    continue
                logger.debug(
                    "Replacing Linear module '%s.%s' with DeltaLinear in model '%s'",
                    parent.__class__.__name__,
                    name,
                    self.__class__.__name__,
                )
                setattr(parent, name, to_delta_linear(child))
                replacements += 1

        if replacements == 0:
            raise RuntimeError(
                "No Linear layer was replaced by a delta compressed one - delta"
                " compression would not be simulated during inference."
            )
