import logging

from src.nn.data import Dataset
from src.nn.model import Sequential

from .experiment import Experiment

logger: logging.Logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        experiments: list[Experiment],
    ) -> None:
        self.experiments: list[Experiment] = experiments

    def run(self, model: Sequential, dataset: Dataset) -> None:
        for idx, experiment in enumerate(self.experiments):
            logger.info(f"Running Experiment {idx}")
            experiment.run(model=model, dataset=dataset)
