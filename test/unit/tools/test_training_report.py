import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from src.tools.training_report import app, plot_run, write_report


def _make_run(run_dir: Path, offset: float, epochs: int = 3) -> None:
    for split, loss_base in (("training", 1.0), ("validation", 1.5)):
        directory = run_dir / "metrics" / split
        directory.mkdir(parents=True)
        for epoch in range(epochs):
            metrics = {
                "loss": loss_base + offset - epoch * 0.1,
                "accuracy": 0.5 + offset / 10 + epoch * 0.1,
            }
            (directory / f"epoch{epoch}.json").write_text(json.dumps(metrics))


def _make_logs(tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    _make_run(logs / "a" / "run1", 0.0)
    _make_run(logs / "a" / "run2", 0.2)
    return logs


def test_report_writes_tables_and_statistics_without_plots(tmp_path):
    logs = _make_logs(tmp_path)
    output = tmp_path / "report"

    write_report(logs, output)

    assert sorted(p.name for p in output.iterdir()) == [
        "accuracy.csv",
        "loss.csv",
        "statistics.txt",
    ]
    with open(output / "loss.csv") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3
    assert float(rows[0]["a/run1_train"]) == 1.0
    assert float(rows[0]["mean_val"]) == (1.5 + 1.7) / 2
    assert "Best run" in (output / "statistics.txt").read_text()


def test_report_with_plots_writes_only_the_overview(tmp_path):
    logs = _make_logs(tmp_path)
    output = tmp_path / "report"

    write_report(logs, output, plots=True)

    plots = output / "plots"
    for name in ("loss.png", "accuracy.png"):
        assert (plots / name).read_bytes().startswith(b"\x89PNG")
    assert not (plots / "runs").exists()


def test_report_with_run_plots_writes_a_plot_for_each_run(tmp_path):
    logs = _make_logs(tmp_path)
    output = tmp_path / "report"

    write_report(logs, output, run_plots=True)

    plots = output / "plots"
    for name in ("loss.png", "accuracy.png"):
        assert (plots / name).read_bytes().startswith(b"\x89PNG")
        for run in ("run1", "run2"):
            assert (plots / "runs" / "a" / run / name).read_bytes().startswith(
                b"\x89PNG"
            )


def test_plot_run_writes_graphs_next_to_the_metrics(tmp_path):
    run = tmp_path / "run"
    _make_run(run, 0.0)

    plot_run(run)

    for name in ("loss.png", "accuracy.png"):
        assert (run / "graphs" / name).read_bytes().startswith(b"\x89PNG")


def test_cli_defaults_to_a_report_folder_in_the_logs_directory(tmp_path):
    logs = _make_logs(tmp_path)

    result = CliRunner().invoke(app, [str(logs), "--plots"])

    assert result.exit_code == 0, result.output
    assert (logs / "report" / "statistics.txt").exists()
    assert (logs / "report" / "plots" / "loss.png").exists()


def test_cli_fails_when_no_metrics_are_found(tmp_path):
    result = CliRunner().invoke(app, [str(tmp_path)])

    assert result.exit_code != 0
