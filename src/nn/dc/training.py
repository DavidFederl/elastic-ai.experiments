"""Simulate delta compression of the parameters during the training with torch.

`apply_delta_compression` registers a parametrization on the parameters of a model.
Each forward pass quantizes the parameter to fixed-point integers (floor), compresses it to
deltas, inflates it again and returns the restored values. The gradient passes
unchanged to the original parameter (straight-through estimator), so the model, the
optimizer and the `Training` class need no other change.

The compression runs along the first dimension of the parameter: the reference value
is the first row and the deltas are taken between rows. `consecutive` loops over the
rows in Python, so the cost of a forward pass grows with the number of rows.
"""

import logging
from typing import cast

import torch
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.experimental.dc import DeltaCompression
from torch import Tensor, nn
from torch.nn.utils import parametrize

from .delta_compression import (
    DeltaType,
    Quantizer,
    build_delta_compression,
    resolve_value_range,
)

logger = logging.getLogger(__name__)


class DeltaCompressedParameter(nn.Module):
    """Parametrization that returns a parameter after a compress and inflate round trip."""

    def __init__(
        self,
        fxp_arithmetic: FxpArithmetic,
        delta_compression: DeltaCompression,
        quantizer: Quantizer,
    ) -> None:
        super().__init__()
        self.fxp_arithmetic = fxp_arithmetic
        self.delta_compression = delta_compression
        self.quantizer = quantizer

    def _quantize(self, parameter: Tensor) -> Tensor:
        # floor like MathOperations.quantize of elasticai.creator, so the integers
        # match the values the fixed-point modules use in their own forward pass
        integers = torch.floor(parameter * 2**self.fxp_arithmetic.frac_bits)
        return integers.clamp(
            self.fxp_arithmetic.minimum_as_integer,
            self.fxp_arithmetic.maximum_as_integer,
        ).to(torch.int32)

    def forward(self, parameter: Tensor) -> Tensor:
        with torch.no_grad():
            integers = self._quantize(parameter)
            reference, delta = self.delta_compression.compress(integers)
            # wrapped deltas can only be decoded with the value range of the data
            value_range = (
                resolve_value_range(
                    (int(integers.min()), int(integers.max())),
                    self.delta_compression.delta_width,
                )
                if self.quantizer == Quantizer.CUT
                else None
            )
            restored = self.delta_compression.inflate(
                reference, delta, value_range=value_range
            )
            restored = restored.to(parameter.dtype) / 2**self.fxp_arithmetic.frac_bits
        # straight-through estimator: forward uses `restored`, backward sees identity
        return parameter + (restored - parameter).detach()


def apply_delta_compression(
    model: nn.Module,
    fxp_params: FxpParams,
    delta_type: DeltaType,
    quantizer: Quantizer,
    delta_bits: int,
    module_types: tuple[type[nn.Module], ...] = (nn.Linear,),
    parameter_names: tuple[str, ...] = ("weight", "bias"),
) -> nn.Module:
    """Registers delta compression on the parameters of all matching modules.

    The state dict of the model gets the keys `parametrizations.<name>.original`
    instead of `<name>`. Call `remove_delta_compression` to get the plain keys back.

    Raises:
        RuntimeError: If no parameter was found, so no compression would happen.
    """
    fxp_arithmetic = FxpArithmetic(fxp_params)
    delta_compression = build_delta_compression(delta_type, quantizer, delta_bits)

    registered = 0
    for module in model.modules():
        if not isinstance(module, module_types):
            continue
        for name in parameter_names:
            if getattr(module, name, None) is None:
                continue
            parametrize.register_parametrization(
                module,
                name,
                DeltaCompressedParameter(fxp_arithmetic, delta_compression, quantizer),
            )
            registered += 1

    if registered == 0:
        raise RuntimeError(
            f"No parameter {parameter_names} found in modules of type {module_types}"
            " - delta compression would not be simulated."
        )
    logger.info(f"Delta compression registered on {registered} parameters")
    return model


def remove_delta_compression(model: nn.Module, leave_compressed: bool = False) -> None:
    """Removes all delta compression parametrizations from the model.

    If `leave_compressed` is set, each parameter keeps the compressed values.
    Otherwise it keeps the original values.
    """
    for module in model.modules():
        if not parametrize.is_parametrized(module):
            continue
        parametrizations = cast(nn.ModuleDict, module.parametrizations)
        for name in list(parametrizations.keys()):
            parametrize.remove_parametrizations(
                module, name, leave_parametrized=leave_compressed
            )
