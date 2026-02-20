from pathlib import Path

from src.nn.data import Dataset
from src.nn.model import Sequential

from .experiment import Experiment


class ExperimentRunner:
    def __init__(
        self,
        log_dir: Path,
        experiments: list[Experiment],
    ) -> None:
        self.log_dir: Path = log_dir
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.execution_log: Path = self.log_dir.joinpath("experiments_log.txt")
        self.execution_log.touch(exist_ok=True)

        self.experiments: list[Experiment] = experiments

    def _write_log(self, message: str, newline: bool = False) -> None:
        with self.execution_log.open("a") as file:
            file.write(message)
            if newline:
                file.write("\n")

    def run(self, model: Sequential, dataset: Dataset) -> None:
        for experiment in self.experiments:
            self._write_log(f"{experiment.__class__.__name__}: ")
            experiment.run(model=model, dataset=dataset)
            self._write_log("FINISHED", True)
            self._write_log(f"    {experiment.log_dir.absolute()}", True)
