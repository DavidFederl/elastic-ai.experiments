"""Report on training runs: CSV tables, statistics and optional plots.

Scans all `metrics/` folders under a logs directory. Writes to the output directory:

    loss.csv       — epoch, <run>_train, <run>_val, ...  (wide format)
    accuracy.csv   — epoch, <run>_train, <run>_val, ...  (wide format)
    statistics.txt — per-epoch mean, std, 95% CI across runs; best/worst run

With `--plots` it also writes, into `<output-dir>/plots/`:

    loss.png, accuracy.png        — all runs: mean and 95% CI band over the runs

With `--run-plots` it also writes the training and validation curve of each run:

    plots/runs/<run>/loss.png, accuracy.png

This writes two images per run, which is slow for hundreds of runs.

`plot_run` draws the curves of one run. Training scripts call it after training.
"""

import csv
import json
import logging
import math
import re
import statistics
from pathlib import Path
from typing import Annotated

import matplotlib
import matplotlib.pyplot as plt
import typer

matplotlib.use("Agg")  # Use non-interactive backend

app = typer.Typer()

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent
_Z95 = 1.96  # z-score for 95% confidence interval
_SPLITS = (("train", "training"), ("val", "validation"))


def epoch_number(path: Path) -> int:
    match = re.search(r"epoch(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_split(metrics_dir: Path, split: str) -> dict[int, dict]:
    split_dir = metrics_dir / split
    if not split_dir.is_dir():
        return {}
    result = {}
    for json_file in split_dir.glob("*.json"):
        epoch = epoch_number(json_file)
        with open(json_file) as f:
            data = json.load(f)
        result[epoch] = data
    return result


# epoch -> {"train": value, "val": value}
EpochData = dict[int, dict[str, float | str]]
# run_name -> EpochData
RunData = dict[str, EpochData]


def load_run(metrics_dir: Path) -> tuple[EpochData, EpochData]:
    """Return the loss and accuracy per epoch of one `metrics/` folder."""
    training = load_split(metrics_dir, "training")
    validation = load_split(metrics_dir, "validation")

    loss: EpochData = {}
    accuracy: EpochData = {}
    for epoch in sorted(set(training) | set(validation)):
        train = training.get(epoch, {})
        val = validation.get(epoch, {})
        loss[epoch] = {"train": train.get("loss", ""), "val": val.get("loss", "")}
        accuracy[epoch] = {
            "train": train.get("accuracy", ""),
            "val": val.get("accuracy", ""),
        }
    return loss, accuracy


def collect(logs_dir: Path) -> tuple[RunData, RunData]:
    loss_data: RunData = {}
    accuracy_data: RunData = {}

    for metrics_dir in sorted(logs_dir.rglob("metrics")):
        if not metrics_dir.is_dir():
            continue

        run_name = str(metrics_dir.parent.relative_to(logs_dir))
        loss_data[run_name], accuracy_data[run_name] = load_run(metrics_dir)

    return loss_data, accuracy_data


def write_wide_csv(path: Path, data: RunData) -> None:
    runs = sorted(data.keys())
    all_epochs = sorted({epoch for run_data in data.values() for epoch in run_data})

    fieldnames = (
        ["epoch"]
        + [f"{run}_{split}" for run in runs for split in ("train", "val")]
        + ["mean_train", "mean_val", "sd_train", "sd_val"]
    )

    rows = []
    for epoch in all_epochs:
        row: dict[str, object] = {"epoch": epoch}
        for run in runs:
            epoch_data = data.get(run, {}).get(epoch, {})
            row[f"{run}_train"] = epoch_data.get("train", "")
            row[f"{run}_val"] = epoch_data.get("val", "")
        train_vals = _values_at_epoch(data, epoch, "train")
        val_vals = _values_at_epoch(data, epoch, "val")
        row["mean_train"] = statistics.mean(train_vals) if train_vals else ""
        row["mean_val"] = statistics.mean(val_vals) if val_vals else ""
        row["sd_train"] = statistics.stdev(train_vals) if len(train_vals) > 1 else ""
        row["sd_val"] = statistics.stdev(val_vals) if len(val_vals) > 1 else ""
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def _values_at_epoch(data: RunData, epoch: int, split: str) -> list[float]:
    values = []
    for run_data in data.values():
        v = run_data.get(epoch, {}).get(split, "")
        if isinstance(v, (int, float)):
            values.append(float(v))
    return values


def _ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, mean
    margin = _Z95 * statistics.stdev(values) / math.sqrt(n)
    return mean - margin, mean + margin


def _best_worst(
    data: RunData, split: str, lower_is_better: bool
) -> tuple[str, float, str, float]:
    """Return (best_run, best_value, worst_run, worst_value) based on the final epoch."""
    candidates: list[tuple[str, float]] = []
    for run_name, epochs in data.items():
        if not epochs:
            continue
        last_epoch = max(epochs)
        v = epochs[last_epoch].get(split, "")
        if isinstance(v, (int, float)):
            candidates.append((run_name, float(v)))
    if not candidates:
        return ("—", float("nan"), "—", float("nan"))
    candidates.sort(key=lambda x: x[1])
    best = candidates[0] if lower_is_better else candidates[-1]
    worst = candidates[-1] if lower_is_better else candidates[0]
    return best[0], best[1], worst[0], worst[1]


def _format_section(data: RunData, metric: str, lower_is_better: bool) -> str:
    runs = sorted(data.keys())
    n_runs = len(runs)
    all_epochs = sorted({epoch for run_data in data.values() for epoch in run_data})

    lines: list[str] = []
    lines.append(f"{metric.capitalize()} Statistics")
    lines.append("=" * (len(metric) + 11))
    lines.append(f"Runs: {n_runs}")
    lines.append("")

    header = f"  {'Epoch':>5}  {'Mean Train':>10}  {'SD Train':>9}  {'95% CI Train':<24}  {'Mean Val':>10}  {'SD Val':>9}  {'95% CI Val':<24}"
    separator = "  " + "-" * (len(header) - 2)
    lines.append(header)
    lines.append(separator)

    for epoch in all_epochs:
        train_vals = _values_at_epoch(data, epoch, "train")
        val_vals = _values_at_epoch(data, epoch, "val")

        if train_vals:
            t_mean = statistics.mean(train_vals)
            t_std = statistics.stdev(train_vals) if len(train_vals) > 1 else 0.0
            t_lo, t_hi = _ci(train_vals)
            t_mean_s = f"{t_mean:.4f}"
            t_std_s = f"{t_std:.4f}"
            t_ci_s = f"[{t_lo:.4f}, {t_hi:.4f}]"
        else:
            t_mean_s = t_std_s = t_ci_s = "—"

        if val_vals:
            v_mean = statistics.mean(val_vals)
            v_std = statistics.stdev(val_vals) if len(val_vals) > 1 else 0.0
            v_lo, v_hi = _ci(val_vals)
            v_mean_s = f"{v_mean:.4f}"
            v_std_s = f"{v_std:.4f}"
            v_ci_s = f"[{v_lo:.4f}, {v_hi:.4f}]"
        else:
            v_mean_s = v_std_s = v_ci_s = "—"

        lines.append(
            f"  {epoch:>5}  {t_mean_s:>10}  {t_std_s:>9}  {t_ci_s:<24}  {v_mean_s:>10}  {v_std_s:>9}  {v_ci_s:<24}"
        )

    epoch_train_means = [
        statistics.mean(vs)
        for epoch in all_epochs
        if (vs := _values_at_epoch(data, epoch, "train"))
    ]
    epoch_val_means = [
        statistics.mean(vs)
        for epoch in all_epochs
        if (vs := _values_at_epoch(data, epoch, "val"))
    ]
    epoch_train_stds = [
        statistics.stdev(vs)
        for epoch in all_epochs
        if len(vs := _values_at_epoch(data, epoch, "train")) > 1
    ]
    epoch_val_stds = [
        statistics.stdev(vs)
        for epoch in all_epochs
        if len(vs := _values_at_epoch(data, epoch, "val")) > 1
    ]
    total_t_std = (
        f"{statistics.stdev(epoch_train_means):.4f}"
        if len(epoch_train_means) > 1
        else "—"
    )
    total_v_std = (
        f"{statistics.stdev(epoch_val_means):.4f}" if len(epoch_val_means) > 1 else "—"
    )
    avg_t_std = f"{statistics.mean(epoch_train_stds):.4f}" if epoch_train_stds else "—"
    avg_v_std = f"{statistics.mean(epoch_val_stds):.4f}" if epoch_val_stds else "—"
    lines.append(separator)
    lines.append(
        f"  {'Total':>5}  {'':>10}  {total_t_std:>9}  {'':24}  {'':>10}  {total_v_std:>9}"
    )
    lines.append(
        f"  {'AvgSD':>5}  {'':>10}  {avg_t_std:>9}  {'':24}  {'':>10}  {avg_v_std:>9}"
    )
    lines.append("")

    best_run, best_val, worst_run, worst_val = _best_worst(data, "val", lower_is_better)
    qualifier = "lowest" if lower_is_better else "highest"
    lines.append(
        f"Best run  ({qualifier} final val {metric}):  {best_run}  →  {best_val:.4f}"
    )
    lines.append(
        f"Worst run ({qualifier} final val {metric}):  {worst_run}  →  {worst_val:.4f}"
    )

    return "\n".join(lines)


_DEFINITIONS = """\
Definitions
===========
SD    — standard deviation
AvgSD — average standard deviation across epochs
CI    — 95% confidence interval  (z = 1.96,  mean ± 1.96 · SD / √n)
Train — training split
Val   — validation split
"""


def write_statistics(path: Path, loss_data: RunData, accuracy_data: RunData) -> None:
    sections = [
        _DEFINITIONS,
        _format_section(loss_data, "loss", lower_is_better=True),
        _format_section(accuracy_data, "accuracy", lower_is_better=False),
    ]
    content = "\n\n\n".join(sections) + "\n"
    path.write_text(content)
    print(f"Wrote statistics to {path}")


def _series(epochs: EpochData, split: str) -> tuple[list[int], list[float]]:
    """Return the epochs and numeric values of one split, skipping gaps."""
    points = [
        (epoch, float(value))
        for epoch, values in sorted(epochs.items())
        if isinstance(value := values.get(split, ""), (int, float))
    ]
    return [epoch for epoch, _ in points], [value for _, value in points]


def _plot_run_data(loss: EpochData, accuracy: EpochData, output_dir: Path) -> None:
    for metric, epochs in (("loss", loss), ("accuracy", accuracy)):
        figure, axis = plt.subplots(figsize=(10, 6))
        for split, label in _SPLITS:
            x, y = _series(epochs, split)
            if x:
                axis.plot(x, y, marker="o", markersize=3, label=label)
        if not axis.lines:
            logger.warning(f"No {metric} data found for {output_dir}")
            plt.close(figure)
            continue
        axis.set_title(f"{metric.capitalize()} over Epochs")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.capitalize())
        axis.grid(True)
        axis.legend()
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / f"{metric}.png")
        plt.close(figure)
        logger.info(f"Saved plot to {output_dir / f'{metric}.png'}")


def plot_run(run_dir: Path, output_dir: Path | None = None) -> None:
    """Plot training and validation loss and accuracy of one run.

    Reads `<run_dir>/metrics` and writes `loss.png` and `accuracy.png` to
    `output_dir` (default: `<run_dir>/graphs`).
    """
    loss, accuracy = load_run(run_dir / "metrics")
    _plot_run_data(loss, accuracy, output_dir or run_dir / "graphs")


def _plot_all_runs(data: RunData, metric: str, path: Path) -> None:
    """Plot the mean and 95% CI band over all runs, with each run behind them."""
    all_epochs = sorted({epoch for epochs in data.values() for epoch in epochs})
    figure, axis = plt.subplots(figsize=(10, 6))
    for index, (split, label) in enumerate(_SPLITS):
        color = f"C{index}"
        for epochs in data.values():
            x, y = _series(epochs, split)
            axis.plot(x, y, color=color, alpha=0.15, linewidth=0.8)
        per_epoch = {
            epoch: values
            for epoch in all_epochs
            if (values := _values_at_epoch(data, epoch, split))
        }
        if not per_epoch:
            continue
        x = list(per_epoch)
        axis.plot(
            x,
            [statistics.mean(v) for v in per_epoch.values()],
            color=color,
            label=f"{label} (mean of {len(data)} runs)",
        )
        if len(data) > 1:
            bounds = [_ci(v) for v in per_epoch.values()]
            axis.fill_between(
                x,
                [low for low, _ in bounds],
                [high for _, high in bounds],
                color=color,
                alpha=0.25,
                label=f"{label} 95% CI",
            )
    axis.set_title(f"{metric.capitalize()} over Epochs")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric.capitalize())
    axis.grid(True)
    axis.legend()
    figure.savefig(path)
    plt.close(figure)
    print(f"Wrote plot to {path}")


def write_plots(
    logs_dir: Path,
    output_dir: Path,
    loss_data: RunData,
    accuracy_data: RunData,
    run_plots: bool = False,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_all_runs(loss_data, "loss", plots_dir / "loss.png")
    _plot_all_runs(accuracy_data, "accuracy", plots_dir / "accuracy.png")
    if not run_plots:
        return
    for run in loss_data:
        name = logs_dir.name if run == "." else run
        _plot_run_data(loss_data[run], accuracy_data[run], plots_dir / "runs" / name)


def write_report(
    logs_dir: Path,
    output_dir: Path,
    plots: bool = False,
    run_plots: bool = False,
) -> None:
    """Write the CSV tables and statistics, and optionally plots, of all runs."""
    loss_data, accuracy_data = collect(logs_dir)
    if not loss_data:
        raise typer.BadParameter(f"No metrics/ folders found under {logs_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_wide_csv(output_dir / "loss.csv", loss_data)
    write_wide_csv(output_dir / "accuracy.csv", accuracy_data)
    write_statistics(output_dir / "statistics.txt", loss_data, accuracy_data)
    if plots or run_plots:
        write_plots(logs_dir, output_dir, loss_data, accuracy_data, run_plots)


@app.command()
def main(
    logs_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory to scan for metrics/ folders.",
            exists=True,
            file_okay=False,
        ),
    ] = _REPO_ROOT / "logs",
    output_dir: Annotated[
        Path | None,
        typer.Argument(help="Directory for the report. Default: LOGS_DIR/report."),
    ] = None,
    plots: Annotated[
        bool, typer.Option(help="Also plot the mean and 95% CI over all runs.")
    ] = False,
    run_plots: Annotated[
        bool, typer.Option(help="Also plot each run (two images per run).")
    ] = False,
) -> None:
    """Report on the training runs below a logs directory."""
    write_report(logs_dir, output_dir or logs_dir / "report", plots, run_plots)


if __name__ == "__main__":
    app()
