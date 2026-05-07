import torch
from elasticai.creator.arithmetic import FxpParams


class ConsecutiveDeltaCompression:
    def __init__(self, bit_width: int, fxp_params: FxpParams) -> None:
        self._fxp_params: FxpParams = fxp_params

        self._bit_width: int = bit_width
        self._min_val: float = float(-(2 ** (self._bit_width - 1)))
        self._max_val: float = float((2 ** (self._bit_width - 1)) - 1)

    def compress(self, data: torch.Tensor) -> torch.Tensor:
        original_shape = data.shape
        flattened = data.flatten()

        # Compute differences without in-place operations
        diff = flattened[1:] - flattened[:-1]
        diff_clamped = torch.clamp(input=diff, min=self._min_val, max=self._max_val)

        # Concatenate first element with clamped differences
        result = torch.cat([flattened[:1], diff_clamped])

        return result.reshape(original_shape)

    def inflate(self, data: torch.Tensor) -> torch.Tensor:
        original_shape = data.shape
        flattened = data.flatten()

        # Compute cumulative sum
        cumsum_result = torch.cumsum(flattened, dim=0)

        # Clamp values without in-place operations
        min_val = float(self._fxp_params.minimum_as_integer)
        max_val = float(
            self._fxp_params.maximum_as_integer
        )  # Fixed: was using minimum for both

        # Create result with first element unchanged, rest clamped
        first_element = cumsum_result[:1]
        clamped_elements = torch.clamp(cumsum_result[1:], min=min_val, max=max_val)
        result = torch.cat([first_element, clamped_elements])

        return result.reshape(original_shape)
