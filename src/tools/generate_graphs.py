#!/usr/bin/env python3
"""Generate graphs from JSON metrics files."""

import json
import logging
import re
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import typer

# Import matplotlib for plotting
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

app = typer.Typer()

logger: logging.Logger


def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging facility.

    Args:
        verbose (bool): Verbose mode.

    Returns:
        Logger: Logger instance.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_json_files(directory: Path) -> List[Dict]:
    """Load all JSON files from a directory in chronological order.

    Args:
        directory (Path): Directory containing JSON files.

    Returns:
        List[Dict]: List of loaded JSON data in chronological order.
    """
    json_files = list(directory.glob("*.json"))

    # Sort files to ensure chronological order
    # This handles both epoch-based files (epoch1.json) and timestamp-based files
    def sort_key(file_path):
        filename = file_path.name
        # Extract epoch number for epoch-based files (e.g., "epoch1.json" -> 1)
        epoch_match = re.search(r"^epoch(\d+)\.json$", filename)
        if epoch_match:
            return (
                0,
                int(epoch_match.group(1)),
            )  # (0, epoch_number) for numeric sorting
        # Extract timestamp for timestamp-based files (e.g., "2026-02-24T10:36:34.442081+00:00")
        timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", filename)
        if timestamp_match:
            return (1, timestamp_match.group(1))  # (1, timestamp) for timestamp sorting
        # Fallback: use filename for sorting
        return (2, filename)  # (2, filename) for fallback sorting

    json_files.sort(key=sort_key)

    data = []

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data.append(json.load(f))
            logger.debug(f"Loaded {json_file}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    return data


def plot_metrics(data: List[Dict], output_dir: Path, metric_name: str = "loss"):
    """Plot metrics from JSON data.

    Args:
        data (List[Dict]): List of metrics data.
        output_dir (Path): Output directory for plots.
        metric_name (str): Name of metric to plot (e.g., 'loss', 'accuracy').
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is not available. Cannot generate plots.")
        return

    output_dir.mkdir(exist_ok=True, parents=True)

    # Extract data for plotting
    epochs = []
    values = []

    for i, metrics in enumerate(data):
        if metric_name in metrics:
            epochs.append(i)
            values.append(metrics[metric_name])

    if not epochs:
        logger.warning(f"No {metric_name} data found in JSON files")
        return

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, marker="o")
    plt.title(f"{metric_name.capitalize()} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.grid(True)

    # Save plot
    output_file = output_dir / f"{metric_name}.png"
    plt.savefig(output_file)
    plt.close()

    logger.info(f"Saved plot to {output_file}")


def plot_comparison(data: List[Dict], output_dir: Path):
    """Plot comparison between original and simulated models.

    Args:
        data (List[Dict]): List of experiment metrics data.
        output_dir (Path): Output directory for plots.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is not available. Cannot generate plots.")
        return

    output_dir.mkdir(exist_ok=True, parents=True)

    # This is for experiment metrics that have "original" and "simulated" sections
    for metrics in data:
        if "original" in metrics and "simulated" in metrics:
            original_metrics = metrics["original"]["metrics"]
            simulated_metrics = metrics["simulated"]["metrics"]

            # Plot loss comparison
            if "loss" in original_metrics and "loss" in simulated_metrics:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(
                    ["Original", "Simulated"],
                    [original_metrics["loss"], simulated_metrics["loss"]],
                )
                ax.set_title("Loss Comparison")
                ax.set_ylabel("Loss")

                output_file = output_dir / "loss_comparison.png"
                plt.savefig(output_file)
                plt.close()
                logger.info(f"Saved loss comparison plot to {output_file}")

            # Plot accuracy comparison
            if "accuracy" in original_metrics and "accuracy" in simulated_metrics:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(
                    ["Original", "Simulated"],
                    [original_metrics["accuracy"], simulated_metrics["accuracy"]],
                )
                ax.set_title("Accuracy Comparison")
                ax.set_ylabel("Accuracy")

                output_file = output_dir / "accuracy_comparison.png"
                plt.savefig(output_file)
                plt.close()
                logger.info(f"Saved accuracy comparison plot to {output_file}")


@app.command()
def generate_graphs(
    input_dir: Annotated[
        Path,
        typer.Option(
            help="Directory containing JSON metrics files.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            help="Output directory for generated graphs.",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = Path("graphs"),
    metrics: Annotated[
        Optional[List[str]],
        typer.Option(
            "--metrics",
            "-m",
            help="Specific metrics to plot (e.g., -m loss -m accuracy). "
            "If not specified, plots all available metrics.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose output.")] = False,
):
    """Generate graphs from JSON metrics files.

    Args:
        input_dir: Directory containing JSON metrics files.
        output_dir: Output directory for generated graphs.
        metrics: Specific metrics to plot.
        verbose: Enable verbose output.
    """
    global logger
    logger = setup_logging(verbose)

    if not MATPLOTLIB_AVAILABLE:
        logger.error(
            "Matplotlib is not installed. Please install it to generate graphs."
        )
        logger.error("You can install it with: pip install matplotlib")
        return

    logger.info(f"Loading JSON files from {input_dir}")
    data = load_json_files(input_dir)

    if not data:
        logger.error("No valid JSON files found in the input directory")
        return

    logger.info(f"Found {len(data)} JSON files")

    # Determine which metrics to plot
    if metrics is None:
        # Auto-detect metrics from the first file
        sample_data = data[0]
        if isinstance(sample_data, dict):
            if "original" in sample_data and "simulated" in sample_data:
                # This is an experiment metrics file
                metrics = ["loss", "accuracy"]
            else:
                # This is a training metrics file
                metrics = list(sample_data.keys())
        else:
            metrics = []

    logger.info(f"Generating plots for metrics: {metrics}")

    # Generate plots
    for metric in metrics:
        plot_metrics(data, output_dir, metric)

    # Check if this is experiment data and generate comparison plots
    if (
        data
        and isinstance(data[0], dict)
        and "original" in data[0]
        and "simulated" in data[0]
    ):
        plot_comparison(data, output_dir)

    logger.info(f"Graphs generated in {output_dir}")


if __name__ == "__main__":
    app()

