"""Collect accuracy and loss from per-epoch JSON files into CSV and statistics files.

Scans all `metrics/` folders under a given logs directory. Writes:

    loss.csv       — epoch, <run>_train, <run>_val, ...  (wide format)
    accuracy.csv   — epoch, <run>_train, <run>_val, ...  (wide format)
    statistics.txt — per-epoch mean, std, 95% CI across runs; best/worst run
"""

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()

_REPO_ROOT = Path(__file__).parent.parent.parent
_Z95 = 1.96  # z-score for 95% confidence interval


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


# run_name -> epoch -> {"train": value, "val": value}
RunData = dict[str, dict[int, dict[str, float | str]]]


def collect(logs_dir: Path) -> tuple[RunData, RunData]:
    loss_data: RunData = {}
    accuracy_data: RunData = {}

    for metrics_dir in sorted(logs_dir.rglob("metrics")):
        if not metrics_dir.is_dir():
            continue

        run_name = str(metrics_dir.parent.relative_to(logs_dir))
        training = load_split(metrics_dir, "training")
        validation = load_split(metrics_dir, "validation")

        loss_data[run_name] = {}
        accuracy_data[run_name] = {}
        for epoch in sorted(set(training) | set(validation)):
            train = training.get(epoch, {})
            val = validation.get(epoch, {})
            loss_data[run_name][epoch] = {
                "train": train.get("loss", ""),
                "val": val.get("loss", ""),
            }
            accuracy_data[run_name][epoch] = {
                "train": train.get("accuracy", ""),
                "val": val.get("accuracy", ""),
            }

    return loss_data, accuracy_data


def write_wide_csv(path: Path, data: RunData) -> None:
    runs = sorted(data.keys())
    all_epochs = sorted({epoch for run_data in data.values() for epoch in run_data})

    fieldnames = (
        ["epoch"]
        + [f"{run}_{split}" for run in runs for split in ("train", "val")]
        + ["mean_train", "mean_val", "std_train", "std_val"]
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
        row["std_train"] = statistics.stdev(train_vals) if len(train_vals) > 1 else ""
        row["std_val"] = statistics.stdev(val_vals) if len(val_vals) > 1 else ""
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

    header = f"  {'Epoch':>5}  {'Mean Train':>10}  {'Std Train':>9}  {'95% CI Train':<24}  {'Mean Val':>10}  {'Std Val':>9}  {'95% CI Val':<24}"
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
    total_t_std = (
        f"{statistics.stdev(epoch_train_means):.4f}"
        if len(epoch_train_means) > 1
        else "—"
    )
    total_v_std = (
        f"{statistics.stdev(epoch_val_means):.4f}" if len(epoch_val_means) > 1 else "—"
    )
    lines.append(separator)
    lines.append(
        f"  {'Total':>5}  {'':>10}  {total_t_std:>9}  {'':24}  {'':>10}  {total_v_std:>9}"
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


def write_statistics(path: Path, loss_data: RunData, accuracy_data: RunData) -> None:
    sections = [
        _format_section(loss_data, "loss", lower_is_better=True),
        _format_section(accuracy_data, "accuracy", lower_is_better=False),
    ]
    content = "\n\n\n".join(sections) + "\n"
    path.write_text(content)
    print(f"Wrote statistics to {path}")


@app.command()
def main(
    logs_dir: Annotated[
        Path,
        typer.Argument(help="Directory to scan for metrics/ folders."),
    ] = _REPO_ROOT / "logs",
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory to write output files."),
    ] = Path(__file__).parent,
) -> None:
    loss_data, accuracy_data = collect(logs_dir)
    write_wide_csv(output_dir / "loss.csv", loss_data)
    write_wide_csv(output_dir / "accuracy.csv", accuracy_data)
    write_statistics(output_dir / "statistics.txt", loss_data, accuracy_data)


if __name__ == "__main__":
    app()
