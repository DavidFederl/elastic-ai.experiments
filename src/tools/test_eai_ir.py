import json
from pathlib import Path

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.ir import IrSerializer
from elasticai.creator.nn.fixed_point import MathOperations
from elasticai.creator.torch2ir import Torch2Ir
from torch import load, nn

from src.nn.model import linear_v1_torch as linear_v1

# load model from disk
_, model = linear_v1(in_features=784, out_features=10)
model.load_state_dict(
    load(Path("logs/").joinpath("eai_Q8.8", "training", "models", "model.pth"))
)


quant_params = FxpParams(total_bits=16, frac_bits=8)
quant_arithmetic = FxpArithmetic(fxp_params=quant_params)
quant_ops = MathOperations(config=quant_arithmetic)

# instantiate converter
t2ir = Torch2Ir()


def hardtanh(module: nn.Hardtanh) -> dict:
    return {}


def linear(module: nn.Linear) -> dict:
    return {
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None,
        # "weights": module.weight.detach().numpy().tolist(),
        # "biases": module.bias.detach().numpy().tolist()
        # if module.bias is not None
        # else [],
        # "weights": (quant_ops.quantize(module.weight.detach())).numpy().tolist(),
        # "biases": quant_ops.quantize(module.bias.detach()).numpy().tolist()
        # if module.bias is not None
        # else [],
        "weights": quant_arithmetic.cut_as_integer(
            (quant_ops.quantize(module.weight.detach())).numpy().tolist()
        ),
        "biases": quant_arithmetic.cut_as_integer(
            quant_ops.quantize(module.bias.detach()).numpy().tolist()
        )
        if module.bias is not None
        else [],
    }


t2ir.register(hardtanh.__name__, hardtanh)
t2ir.register(linear.__name__, linear)


# convert model to IR
ir_graph, ir_registry = t2ir.convert(model)

# instantiate seralizer
serializer = IrSerializer()

data = {}
data["graph"] = serializer.serialize(ir_graph)
for name in ir_registry:
    data[name] = serializer.serialize(ir_registry[name])

with Path("test_quant_int.json").open("w") as f:
    json.dump(data, f, indent=4)
