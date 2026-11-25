import json

import pytest
import torch
from torch import nn

from src.tools.analyze_model_weights import analyze_weights


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(2, 2)
        self.second = nn.Linear(2, 1)


@pytest.fixture
def weights(tmp_path):
    model = TinyModel()
    with torch.no_grad():
        model.first.weight.copy_(torch.tensor([[-1.0, 0.0], [1.0, 2.0]]))
        model.first.bias.zero_()
        model.second.weight.copy_(torch.tensor([[3.0, 3.0]]))
        model.second.bias.fill_(1.0)
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)
    return path


def test_all_weights_are_reported_together(weights, tmp_path):
    output_dir = tmp_path / "out"

    result = analyze_weights(TinyModel, weights, output_dir=output_dir, bins=4)

    assert list(result) == ["all"]
    metrics = result["all"]
    assert metrics["count"] == 9
    assert metrics["min"] == -1.0
    assert metrics["max"] == 3.0
    assert metrics["mean"] == pytest.approx(1.0)
    distribution = metrics["distribution"]
    assert len(distribution) == 4
    assert sum(count for _, _, count in distribution) == 9
    assert distribution[0][0] == -1.0
    assert distribution[-1][1] == 3.0
    assert all(end == next_start for (_, end, _), (next_start, _, _) in zip(distribution, distribution[1:]))
    stored = json.loads((output_dir / "weight_metrics.json").read_text())
    assert stored["all"]["distribution"] == [list(bin) for bin in distribution]
    assert (output_dir / "weight_distribution.png").read_bytes().startswith(
        b"\x89PNG"
    )


def test_per_layer_reports_each_parameter(weights):
    result = analyze_weights(TinyModel, weights, per_layer=True)

    assert set(result) == {
        "all",
        "first.weight",
        "first.bias",
        "second.weight",
        "second.bias",
    }
    assert result["first.weight"]["min"] == -1.0
    assert result["first.weight"]["max"] == 2.0
    assert result["second.weight"]["mean"] == 3.0
    assert result["all"]["count"] == 9
    assert result["all"]["min"] == -1.0
    assert result["all"]["max"] == 3.0


def test_snapshot_file_is_accepted(weights, tmp_path):
    snapshot = tmp_path / "snapshot.pth"
    torch.save({"epoch": 3, "model": torch.load(weights)}, snapshot)

    result = analyze_weights(TinyModel, snapshot)

    assert result["all"]["max"] == 3.0


@pytest.mark.parametrize("per_layer", [False, True])
def test_distribution_plot_is_written_to_the_output_directory(weights, tmp_path, per_layer):
    output_dir = tmp_path / "out"

    analyze_weights(TinyModel, weights, per_layer=per_layer, output_dir=output_dir)

    assert list(json.loads((output_dir / "weight_metrics.json").read_text())) == [
        "all"
    ]
    assert (output_dir / "weight_distribution.png").read_bytes().startswith(
        b"\x89PNG"
    )
    overview = output_dir / "weight_distribution_layers.png"
    assert overview.exists() == per_layer
    if per_layer:
        assert overview.read_bytes().startswith(b"\x89PNG")
        layers = [d for d in output_dir.iterdir() if d.is_dir()]
        assert {d.name for d in layers} == {
            "first.weight",
            "first.bias",
            "second.weight",
            "second.bias",
        }
        for layer in layers:
            assert (layer / "weight_distribution.png").read_bytes().startswith(
                b"\x89PNG"
            )
            assert list(json.loads((layer / "weight_metrics.json").read_text())) == [
                layer.name
            ]
    else:
        assert not [d for d in output_dir.iterdir() if d.is_dir()]


def test_constant_layer_has_a_histogram_with_width(weights):
    result = analyze_weights(TinyModel, weights, per_layer=True)

    distribution = result["first.bias"]["distribution"]
    assert distribution[0][0] < distribution[-1][1]
    assert sum(count for _, _, count in distribution) == 2
    assert result["first.bias"]["min"] == result["first.bias"]["max"] == 0.0
