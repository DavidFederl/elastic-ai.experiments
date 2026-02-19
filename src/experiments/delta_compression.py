import copy
import logging
from pathlib import Path

from src.config import Configuration
from src.nn.data import Dataset
from src.nn.delta import ConsecutiveDeltaCompression, FxpParams
from src.nn.model import Sequential, save_as_json

from .experiment import Experiment

logger = logging.getLogger(__name__)


class DeltaExperiment01(Experiment):
    def __init__(
        self,
        log_dir: Path,
        config: Configuration,
    ) -> None:
        model_total_bits = config.get("model.parameter.fixed_point_total_bits", 16)
        model_frac_bits = config.get("model.parameter.fixed_point_fraction_bits", 8)
        compress_total_bits = config.get(
            "experiment.parameter.fixed_point_total_bits", 16
        )
        compress_frac_bits = config.get(
            "experiment.parameter.fixed_point_fraction_bits", 8
        )

        self.delta_compression = ConsecutiveDeltaCompression(
            compress_params=FxpParams(compress_total_bits, compress_frac_bits),
            inflate_params=FxpParams(model_total_bits, model_frac_bits),
        )
        self.log_dir = log_dir
        logger.info("Experiment: Delta01")
        logger.debug(f"Compress Params: {compress_total_bits=}; {compress_frac_bits=}")
        logger.debug(f"Inflate Params: {model_total_bits=}; {model_frac_bits=}")

    def run(self, model: Sequential, dataset: Dataset) -> None:
        logger.info("START Experiment")

        model_copy: Sequential = copy.deepcopy(model)
        save_as_json(
            model_copy.state_dict(), self.log_dir.joinpath("model_original.json")
        )
        logger.debug(f"Model Copy: {model_copy}")

        model_copy = self.delta_compression.compress(model_copy)
        save_as_json(
            model_copy.state_dict(), self.log_dir.joinpath("model_compressed.json")
        )
        logger.debug(f"Model Compressed: {model_copy}")

        model_copy = self.delta_compression.inflate(model_copy)
        save_as_json(
            model_copy.state_dict(), self.log_dir.joinpath("model_inflated.json")
        )
        logger.debug(f"Model Inflated: {model_copy}")

        logger.info("END Experiment")
