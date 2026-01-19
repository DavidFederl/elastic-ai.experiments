import logging
from typing import Literal

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import MathOperations

logger = logging.getLogger(__name__)


class ConsecutiveDeltaCompression:
    """Fixed-Point based consecutive delta-compression.

    Implents fixed-point based consecutive delta calculation for a sequential.
    A compression is accomplished by reducing the total fixed-point bitwidth.

    IMPORTANT: If fixed-point settings are same as standard sequential no compression is archived.

    Args:
        compress_params (FxpParams): Parameters for the compress operation.
        inflate_params (FxpParams): Parameters for the inflate operation.
    """

    math_operations: dict[Literal["compress", "inflate"], MathOperations] = {}

    def __init__(self, compress_params: FxpParams, inflate_params: FxpParams) -> None:
        self.math_operations["compress"] = MathOperations(
            config=FxpArithmetic(fxp_params=compress_params)
        )
        self.math_operations["inflate"] = MathOperations(
            config=FxpArithmetic(fxp_params=inflate_params)
        )

    def compress(self, model: Sequential) -> Sequential:
        quant_params = {}
        for param_name, param in model.named_parameters():
            delta = param.detach().clone().flatten()
            for index in reversed(range(1, len(delta))):
                delta[index] = delta[index] - delta[index - 1]
            delta[1:] = self.math_operations["compress"].quantize(delta[1:])
            quant_params[param_name] = delta.reshape(param.shape)
        model.load_state_dict(quant_params)
        return model

    def inflate(self, model: Sequential) -> Sequential:
        inflated_params = {}
        for param_name, param in model.named_parameters():
            delta = param.detach().clone().flatten()
            for index in range(1, len(delta)):
                delta[index] = delta[index] + delta[index - 1]
            delta[1:] = self.math_operations["inflate"].quantize(delta[1:])
            inflated_params[param_name] = delta.reshape(param.shape)
        model.load_state_dict(inflated_params)
        return model
