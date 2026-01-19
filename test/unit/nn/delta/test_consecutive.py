import torch
from elasticai.creator.arithmetic import FxpParams

from src.nn.delta import consecutive


class _DummyMathOps:
    def __init__(self, config=None):
        self.config = config
        self.inputs = []

    def quantize(self, values):
        self.inputs.append(values.clone())
        return values + 10


class _DummyModel:
    def __init__(self, params):
        self._params = params
        self.loaded = None

    def named_parameters(self):
        return list(self._params.items())

    def load_state_dict(self, state_dict):
        self.loaded = state_dict


def test_consecutive_delta_compression_init_builds_math_ops(monkeypatch):
    monkeypatch.setattr(consecutive.ConsecutiveDeltaCompression, "math_operations", {})
    monkeypatch.setattr(consecutive, "MathOperations", _DummyMathOps)

    compress_params = FxpParams(total_bits=8, frac_bits=4)
    inflate_params = FxpParams(total_bits=16, frac_bits=8)

    instance = consecutive.ConsecutiveDeltaCompression(
        compress_params=compress_params, inflate_params=inflate_params
    )

    assert "compress" in instance.math_operations
    assert "inflate" in instance.math_operations
    assert isinstance(instance.math_operations["compress"], _DummyMathOps)
    assert isinstance(instance.math_operations["inflate"], _DummyMathOps)


def test_compress_builds_delta_and_quantizes():
    instance = consecutive.ConsecutiveDeltaCompression.__new__(
        consecutive.ConsecutiveDeltaCompression
    )
    instance.math_operations = {"compress": _DummyMathOps(), "inflate": _DummyMathOps()}

    model = _DummyModel({"weight": torch.tensor([1.0, 3.0, 6.0])})

    instance.compress(model)

    expected = torch.tensor([1.0, 12.0, 13.0])
    assert torch.allclose(model.loaded["weight"], expected)
    assert torch.allclose(
        instance.math_operations["compress"].inputs[0],
        torch.tensor([2.0, 3.0]),
    )


def test_inflate_builds_cumsum_and_quantizes():
    instance = consecutive.ConsecutiveDeltaCompression.__new__(
        consecutive.ConsecutiveDeltaCompression
    )
    instance.math_operations = {"compress": _DummyMathOps(), "inflate": _DummyMathOps()}

    model = _DummyModel({"weight": torch.tensor([1.0, 2.0, 3.0])})

    instance.inflate(model)

    expected = torch.tensor([1.0, 13.0, 16.0])
    assert torch.allclose(model.loaded["weight"], expected)
    assert torch.allclose(
        instance.math_operations["inflate"].inputs[0],
        torch.tensor([3.0, 6.0]),
    )
