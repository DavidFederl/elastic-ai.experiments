import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torch import Tensor, cat, no_grad, where
from torch.nn.modules.loss import CrossEntropyLoss, _Loss

from src.nn.data import Dataset
from src.nn.delta import ConsecutiveDeltaCompression, FxpParams
from src.nn.model import Sequential, load_from_json, save_as_json
from src.nn.training import Metrics

from .experiment import Experiment

logger = logging.getLogger(__name__)


class DeltaExperiment01(Experiment):
    def __init__(
        self,
        log_dir: Path,
        model_fixed_point_total_bits: int,
        model_fixed_point_fraction_bits: int,
        compression_fixed_point_total_bits: int,
        compression_fixed_point_fraction_bits: int,
    ) -> None:
        self.delta_compression = ConsecutiveDeltaCompression(
            compress_params=FxpParams(
                compression_fixed_point_total_bits,
                compression_fixed_point_fraction_bits,
            ),
            inflate_params=FxpParams(
                model_fixed_point_total_bits, model_fixed_point_fraction_bits
            ),
        )

        self.log_dir = log_dir.joinpath(f"{datetime.now(timezone.utc).isoformat()}")
        self.log_dir.mkdir(exist_ok=True, parents=True)
        with open(self.log_dir.joinpath("meta.txt"), "w") as mf:
            mf.write(f"EXPERIMEMNT: {self.__class__.__name__}\n")
            mf.write(f"FIXED POINT TOTAL BITS: {compression_fixed_point_total_bits}\n")
            mf.write(
                f"FIXED POINT FRACTION BITS: {compression_fixed_point_fraction_bits}\n"
            )

        self.device = "cpu"
        self.loss_fn = CrossEntropyLoss()

        logger.info("Experiment: Delta01")
        logger.debug(
            f"Compress Params: {compression_fixed_point_total_bits=}; {compression_fixed_point_fraction_bits=}"
        )
        logger.debug(
            f"Inflate Params: {model_fixed_point_total_bits=}; {model_fixed_point_fraction_bits=}"
        )

    def _simulate_compression(self, model: Sequential, dataset: Dataset) -> None:
        save_as_json(model.state_dict(), self.log_dir.joinpath("model_original.json"))
        logger.debug(f"Model Copy: {model}")

        model = self.delta_compression.compress(model)
        save_as_json(model.state_dict(), self.log_dir.joinpath("model_compressed.json"))
        logger.debug(f"Model Compressed: {model}")

        model = self.delta_compression.inflate(model)
        save_as_json(model.state_dict(), self.log_dir.joinpath("model_inflated.json"))
        logger.debug(f"Model Inflated: {model}")

    def _generate_flattend_weights(self, parameter: dict[str, Tensor]) -> Tensor:
        filterd_keys: list[str] = list(
            filter(lambda key: "weight" in key, parameter.keys())
        )
        weight_tensors: list[Tensor] = list(
            parameter[key].flatten() for key in filterd_keys
        )
        return cat(weight_tensors)

    def _analyze_infomration_loss(self) -> dict[str, dict[str, float]]:
        parameter_original: dict[str, Tensor] = load_from_json(
            self.log_dir.joinpath("model_original.json")
        )
        parameter_inflated: dict[str, Tensor] = load_from_json(
            self.log_dir.joinpath("model_inflated.json")
        )

        weights_original: Tensor = self._generate_flattend_weights(parameter_original)
        weights_inflated: Tensor = self._generate_flattend_weights(parameter_inflated)
        weights_difference: Tensor = weights_original - weights_inflated

        results: dict[str, dict[str, int | float]] = {}
        results["original"] = {}
        results["original"]["max"] = weights_original.max().item()
        results["original"]["min"] = weights_original.min().item()
        results["original"]["mean"] = weights_original.abs().mean().item()
        results["simulated"] = {}
        results["simulated"]["max"] = weights_inflated.max().item()
        results["simulated"]["min"] = weights_inflated.min().item()
        results["simulated"]["mean"] = weights_inflated.abs().mean().item()
        results["difference"] = {}
        results["difference"]["max"] = weights_difference.abs().max().item()
        results["difference"]["mean"] = weights_difference.abs().mean().item()
        results["difference"]["num_total"] = weights_difference.numel()
        results["difference"]["num_incorrect"] = (
            where(weights_difference.abs() > 0, 1, 0).sum().item()
        )

        return results

    def _perform_model_evaluation(
        self,
        model: Sequential,
        dataset: Dataset,
        loss_fn: _Loss,
        device: str,
    ) -> dict[str, int | float | list[float]]:
        validation_metrics = Metrics(loss_fn=loss_fn, num_classes=len(dataset.classes))
        validation_metrics.reset()

        with no_grad():
            for inputs, labels in dataset.training_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                validation_metrics.add(labels=labels, outputs=outputs)
            for inputs, labels in dataset.validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                validation_metrics.add(labels=labels, outputs=outputs)

        return validation_metrics.get()  # type: ignore

    def run(self, model: Sequential, dataset: Dataset) -> None:
        logger.info("START Experiment")

        model_copy: Sequential = copy.deepcopy(model)
        self._simulate_compression(model_copy, dataset)

        metrics_information_loss: dict[str, Any] = self._analyze_infomration_loss()

        metrics_information_loss["original"]["metrics"] = (  # type: ignore
            self._perform_model_evaluation(model, dataset, self.loss_fn, self.device)
        )

        metrics_information_loss["simulated"]["metrics"] = (  # type: ignore
            self._perform_model_evaluation(
                model_copy, dataset, self.loss_fn, self.device
            )
        )

        with self.log_dir.joinpath("metrics.json").open("w") as file:
            json.dump(metrics_information_loss, file, indent=4)

        logger.info("END Experiment")
