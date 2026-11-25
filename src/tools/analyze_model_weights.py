import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import matplotlib
import torch
import typer
from torch.nn import Module

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_BINS = 50
METRICS_FILE = "weight_metrics.json"
PLOT_FILE = "weight_distribution.png"
LAYERS_PLOT_FILE = "weight_distribution_layers.png"

app = typer.Typer()


def load_model(model_class: Callable[[], Module], weights: Path) -> Module:
    """Create a model and load the weights of a `.pth` file into it.

    The file holds either a plain state dict (`model.pth`) or a training
    snapshot with the state dict under the key `model`.
    """
    model = model_class()
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    model.load_state_dict(state)
    return model


def weight_metrics(weights: torch.Tensor, bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """Return min, max, mean, median and a histogram of the given values.

    The histogram is a list of `(start, end, count)` tuples, one per bin.
    """
    values = weights.detach().flatten().to(torch.float64)
    low, high = values.min().item(), values.max().item()
    if low == high:  # constant tensor: give the histogram a visible width
        low, high = low - 0.5, high + 0.5
    counts = torch.histc(values, bins=bins, min=low, max=high)
    edges = torch.linspace(low, high, bins + 1)
    return {
        "count": values.numel(),
        "min": values.min().item(),
        "max": values.max().item(),
        "mean": values.mean().item(),
        "median": values.median().item(),
        "distribution": [
            (start, end, int(count))
            for start, end, count in zip(
                edges[:-1].tolist(), edges[1:].tolist(), counts.tolist()
            )
        ],
    }


def plot_distribution(result: dict[str, Any], path: Path, columns: int = 4) -> None:
    """Draw the histogram of each entry of `result` and save it as an image.

    One entry gives a single plot. Several entries give a grid of plots.
    """
    columns = min(columns, len(result))
    rows = -(-len(result) // columns)
    single = len(result) == 1
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8, 5) if single else (4 * columns, 3 * rows),
        squeeze=False,
    )
    for axis, (name, metrics) in zip(axes.flat, result.items()):
        starts, ends, counts = zip(*metrics["distribution"])
        widths = [end - start for start, end in zip(starts, ends)]
        axis.bar(starts, counts, width=widths, align="edge")
        axis.axvline(metrics["mean"], color="red", linestyle="--", label="mean")
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("weight value")
        axis.set_ylabel("count")
        axis.legend(fontsize=8)
    for axis in axes.flat[len(result) :]:
        axis.set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def analyze_weights(
    model_class: Callable[[], Module],
    weights: Path,
    per_layer: bool = False,
    output_dir: Path | None = None,
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Compute weight metrics of a model over all weights, and optionally per layer.

    A layer is one parameter tensor of the model, for example `conv1.weight`.
    The entry `all` combines the weights of all layers. With `per_layer`, the
    result also holds one entry for each layer.

    When `output_dir` is given, the metrics are written as JSON
    (`weight_metrics.json`) and the distribution as an image
    (`weight_distribution.png`). The files of the entry `all` go into
    `output_dir`. With `per_layer`, each layer also gets its own subfolder
    `output_dir/<layer name>` that holds both files. `output_dir` also gets
    `weight_distribution_layers.png`, one image with the histograms of all
    layers side by side.
    """
    model = load_model(model_class, weights)
    parameters = {name: p for name, p in model.named_parameters()}

    combined = torch.cat([p.detach().flatten() for p in parameters.values()])
    result: dict[str, Any] = {"all": weight_metrics(combined, bins)}
    if per_layer:
        for name, parameter in parameters.items():
            result[name] = weight_metrics(parameter, bins)

    if output_dir is not None:
        _write_result({"all": result["all"]}, output_dir)
        if per_layer:
            for name in parameters:
                _write_result({name: result[name]}, output_dir / name)
            layers = {name: result[name] for name in parameters}
            plot_distribution(layers, output_dir / LAYERS_PLOT_FILE)
    return result


def _write_result(result: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / METRICS_FILE).write_text(json.dumps(result, indent=2))
    plot_distribution(result, directory / PLOT_FILE)


def _import_class(path: str) -> Callable[[], Module]:
    module_name, _, class_name = path.partition(":")
    if not class_name:
        raise typer.BadParameter("Use the format 'package.module:ClassName'.")
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as error:
        raise typer.BadParameter(f"Cannot import '{path}': {error}") from error


@app.command()
def main(
    model_class: Annotated[
        Callable[[], Module],
        typer.Argument(
            help="Model class as 'package.module:ClassName'. It needs no arguments.",
            parser=_import_class,
            metavar="MODEL_CLASS",
        ),
    ],
    weights: Annotated[
        Path,
        typer.Argument(
            help="Path to the .pth file (state dict or training snapshot).",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory for the JSON metrics and the distribution plot. With --per-layer, each layer gets a subfolder."
        ),
    ],
    per_layer: Annotated[
        bool, typer.Option(help="Also report each layer on its own.")
    ] = False,
    bins: Annotated[int, typer.Option(min=1, help="Bins of the distribution.")] = (
        DEFAULT_BINS
    ),
) -> None:
    """Report min, max, mean, median and distribution of model weights."""
    result = analyze_weights(model_class, weights, per_layer, output_dir, bins)
    for name, metrics in result.items():
        typer.echo(
            f"{name}: min={metrics['min']:.6g} max={metrics['max']:.6g} "
            f"mean={metrics['mean']:.6g} median={metrics['median']:.6g}"
        )


if __name__ == "__main__":
    app()
