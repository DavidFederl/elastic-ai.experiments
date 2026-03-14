import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from torch import Tensor, cat, no_grad, where
from torch.nn.modules.loss import CrossEntropyLoss, _Loss

from src.nn.data import Dataset
from src.nn.delta import compress_consecutive, inflate_consecutive
from src.nn.model import Sequential, load_from_json, save_as_json
from src.nn.training import Metrics

from .experiment import Experiment

logger = logging.getLogger(__name__)


class DeltaExperiment01(Experiment):
    ORIGINAL_STATE_DICT_FILE = "model_original.json"
    INT_STATE_DICT_FILE = "model_int.json"
    DELTA_INFLATED_STATE_DICT_FILE = "model_delta_inflated.json"
    DELTA_COMPRESSED_STATE_DICT_FILE = "model_delta_compressed.json"
    SIMULATED_STATE_DICT_FILE = "model_simulated.json"

    def __init__(
        self,
        log_dir: Path,
        model_fixed_point_total_bits: int,
        model_fixed_point_fraction_bits: int,
        delta_bit_width: int,
    ) -> None:
        self.model_fxp_params: FxpParams = FxpParams(
            total_bits=model_fixed_point_total_bits,
            frac_bits=model_fixed_point_fraction_bits,
        )
        self.model_fxp_arithmetic: FxpArithmetic = FxpArithmetic(
            fxp_params=self.model_fxp_params
        )

        self.delta_bit_width: int = delta_bit_width

        self.log_dir: Path = log_dir.joinpath(
            f"{datetime.now(timezone.utc).isoformat()}"
        )
        self.log_dir.mkdir(exist_ok=True, parents=True)
        with open(self.log_dir.joinpath("meta.txt"), "w") as meta_file:
            meta_file.write(f"EXPERIMEMNT: {self.__class__.__name__}\n")
            meta_file.write(
                f"MODEL PARAMS: (fxp_total_bits={model_fixed_point_total_bits}, fxp_frac_bits={model_fixed_point_fraction_bits})\n"
            )
            meta_file.write(f"DELTA PARAMS: (delta_bit_width={delta_bit_width})\n")

        self.device = "cpu"
        self.loss_fn = CrossEntropyLoss()

        logger.info("Experiment: Delta01")
        logger.debug(
            f"Model params: (fxp_total_bits={model_fixed_point_total_bits}, fxp_frac_bits={model_fixed_point_fraction_bits}); Delta params: (delta_bit_width={delta_bit_width})"
        )

    def _get_original_model_state(self, model: Sequential) -> dict[str, Tensor]:
        logger.debug("Retriving original model state")
        state_dict: dict[str, Tensor] = model.state_dict()
        save_as_json(state_dict, self.log_dir.joinpath(self.ORIGINAL_STATE_DICT_FILE))
        return state_dict

    def _convert_weights_to_integer(
        self, state_dict: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        logger.debug("Converting rational weights to integer weights")
        for name, tensor in state_dict.items():
            state_dict[name] = self.model_fxp_arithmetic.cut_as_integer(tensor)
        save_as_json(state_dict, self.log_dir.joinpath(self.INT_STATE_DICT_FILE))
        return state_dict

    def _apply_delta_compression(
        self, state_dict: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        logger.debug("Compressing weights")
        for name, tensor in state_dict.items():
            compressed = compress_consecutive(
                data=tensor, bit_width=self.delta_bit_width
            )
            state_dict[name] = compressed
        save_as_json(
            state_dict, self.log_dir.joinpath(self.DELTA_COMPRESSED_STATE_DICT_FILE)
        )

        logger.debug("Inflating weights")
        for name, tensor in state_dict.items():
            inflated = inflate_consecutive(
                delta=tensor, bit_width=self.model_fxp_params.total_bits
            )
            state_dict[name] = inflated
        save_as_json(
            state_dict, self.log_dir.joinpath(self.DELTA_INFLATED_STATE_DICT_FILE)
        )

        return state_dict

    def _convert_integer_to_weights(
        self, state_dict: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        logger.debug("Converting integer weights to rational weights")
        for name, tensor in state_dict.items():
            state_dict[name] = self.model_fxp_arithmetic.as_rational(tensor)
        save_as_json(state_dict, self.log_dir.joinpath(self.SIMULATED_STATE_DICT_FILE))
        return state_dict

    def _simulate_compression(self, model: Sequential, dataset: Dataset) -> None:
        model_state_dict: dict[str, Tensor] = self._get_original_model_state(model)
        model_state_dict = self._convert_weights_to_integer(model_state_dict)
        model_state_dict = self._apply_delta_compression(model_state_dict)
        model_state_dict = self._convert_integer_to_weights(model_state_dict)

        logger.debug("Loading delta simulated model")
        model.load_state_dict(model_state_dict)

    def _generate_flattend_weights(self, parameter: dict[str, Tensor]) -> Tensor:
        filterd_keys: list[str] = list(
            filter(lambda key: "weight" in key, parameter.keys())
        )
        weight_tensors: list[Tensor] = list(
            parameter[key].flatten() for key in filterd_keys
        )
        return cat(weight_tensors)

    def _analyze_quntized_weights(self) -> dict[str, dict[str, float]]:
        parameter_original: dict[str, Tensor] = load_from_json(
            self.log_dir.joinpath(self.INT_STATE_DICT_FILE)
        )
        parameter_simulated: dict[str, Tensor] = load_from_json(
            self.log_dir.joinpath(self.DELTA_INFLATED_STATE_DICT_FILE)
        )

        weights_original: Tensor = self._generate_flattend_weights(parameter_original)
        weights_inflated: Tensor = self._generate_flattend_weights(parameter_simulated)
        weights_difference: Tensor = weights_original - weights_inflated

        results: dict[str, dict[str, int | float]] = {}
        results["original_quant"] = {}
        results["original_quant"]["max"] = weights_original.max().item()
        results["original_quant"]["min"] = weights_original.min().item()
        results["simulated_quant"] = {}
        results["simulated_quant"]["max"] = weights_inflated.max().item()
        results["simulated_quant"]["min"] = weights_inflated.min().item()
        results["difference_quant"] = {}
        results["difference_quant"]["max"] = weights_difference.max().item()
        results["difference_quant"]["min"] = weights_difference.min().item()
        results["difference_quant"]["num_total"] = weights_difference.numel()
        results["difference_quant"]["num_incorrect"] = (
            where(weights_difference.abs() > 0, 1, 0).sum().item()
        )

        return results

    def _analyze_weights(self) -> dict[str, dict[str, float]]:
        parameter_original: dict[str, Tensor] = load_from_json(
            self.log_dir.joinpath(self.ORIGINAL_STATE_DICT_FILE)
        )
        parameter_inflated: dict[str, Tensor] = load_from_json(
            self.log_dir.joinpath(self.SIMULATED_STATE_DICT_FILE)
        )

        weights_original: Tensor = self._generate_flattend_weights(parameter_original)
        weights_inflated: Tensor = self._generate_flattend_weights(parameter_inflated)
        weights_difference: Tensor = weights_original - weights_inflated

        results: dict[str, dict[str, int | float]] = {}
        results["original"] = {}
        results["original"]["max"] = weights_original.max().item()
        results["original"]["min"] = weights_original.min().item()
        results["simulated"] = {}
        results["simulated"]["max"] = weights_inflated.max().item()
        results["simulated"]["min"] = weights_inflated.min().item()
        results["difference"] = {}
        results["difference"]["max"] = weights_difference.max().item()
        results["difference"]["min"] = weights_difference.min().item()
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
        logger.info(f"START {self.__class__.__name__}")

        model_copy: Sequential = copy.deepcopy(model)
        self._simulate_compression(model_copy, dataset)

        metrics_information_loss: dict[str, Any] = {}
        metrics_information_loss.update(self._analyze_quntized_weights())
        metrics_information_loss.update(self._analyze_weights())

        metrics_information_loss["original"]["metrics"] = (
            self._perform_model_evaluation(model, dataset, self.loss_fn, self.device)
        )

        metrics_information_loss["simulated"]["metrics"] = (
            self._perform_model_evaluation(
                model_copy, dataset, self.loss_fn, self.device
            )
        )

        with self.log_dir.joinpath("metrics.json").open("w") as file:
            json.dump(metrics_information_loss, file, indent=4)

        logger.info(f"FINISHED {self.__class__.__name__}")
