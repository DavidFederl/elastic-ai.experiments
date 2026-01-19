import logging
from datetime import datetime
from os import makedirs
from typing import Self

from torch import no_grad, optim, save
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from src.nn.data import Dataset
from src.nn.model import Sequential

from .metrics import Metrics, MetricWriter

logger = logging.getLogger(__name__)


class Training:
    def __init__(
        self,
        model: Sequential,
        dataset: Dataset,
        loss_fn: _Loss,
        optimiezer: optim.Optimizer,
        device: str,
        log_dir: str,
    ):
        makedirs(log_dir, exist_ok=True)
        makedirs(f"{log_dir}/models", exist_ok=True)
        makedirs(f"{log_dir}/tensorboard", exist_ok=True)

        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer = optimiezer
        self.device = device
        self.log_dir = log_dir
        self.tb_writer = SummaryWriter(log_dir=f"{log_dir}/tensorboard")

    def _train_epoch(self, epoch: int) -> Metrics:
        training_metrics = Metrics(
            loss_fn=self.loss_fn, num_classes=len(self.dataset.classes)
        )
        training_metrics.reset()

        for inputs, labels in tqdm(
            self.dataset.training_loader,
            total=len(self.dataset.training_loader),
            desc=f"Training Epoch {epoch + 1} -> Progress",
        ):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, labels)
            loss.backward()

            self.optimizer.step()
            training_metrics.add(labels=labels, outputs=outputs)

        logger.info("Training Epoch Done")
        return training_metrics

    def _validate_epoch(self) -> Metrics:
        validation_metrics = Metrics(
            loss_fn=self.loss_fn, num_classes=len(self.dataset.classes)
        )
        validation_metrics.reset()

        with no_grad():
            for inputs, labels in self.dataset.validation_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                validation_metrics.add(labels=labels, outputs=outputs)

        logger.info("Validation Epoch Done")
        return validation_metrics

    def _log_metrics(
        self,
        epoch: int,
        training_metrics: Metrics,
        validation_metrics: Metrics,
    ) -> None:
        self.tb_writer.add_scalars(
            "Loss",
            {
                "Training": training_metrics.get_loss(),
                "Validation": validation_metrics.get_loss(),
            },
            epoch,
        )
        self.tb_writer.add_scalars(
            "Accuracy",
            {
                "Training": training_metrics.get_accuracy(),
                "Validation": validation_metrics.get_accuracy(),
            },
            epoch,
        )
        self.tb_writer.flush()
        logger.info("Logging Tenbsorboard Done")

        metric_writer = MetricWriter(f"{self.log_dir}/metrics")
        metric_writer.write(training_metrics, False, f"training_epoch{epoch}.json")
        metric_writer.write(validation_metrics, False, f"validation_epoch{epoch}.json")
        logger.info("Logging Metrics Done")

    def _save_model(self, store_only_last_model: bool, epoch: int) -> None:
        model_path: str
        if store_only_last_model:
            model_path = f"{self.log_dir}/models/model.pth"
        else:
            model_path = f"{self.log_dir}/models/{epoch}.pth"

        save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    def train(
        self, epochs: int = 100, store_only_last_model: bool = False
    ) -> dict[int, dict]:
        metrics: dict[int, dict] = {}
        self.model.to(self.device)

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            self.model.train(True)
            training_metrics = self._train_epoch(epoch)
            validation_metrics = self._validate_epoch()
            self._log_metrics(epoch, training_metrics, validation_metrics)
            self._save_model(store_only_last_model, epoch)

            metrics[epoch + 1] = {
                "training": training_metrics.get(),
                "validation": validation_metrics.get(),
            }

        logger.debug("Training Done")
        return metrics


class TrainingBuilder:
    def __init__(self):
        self._model: Sequential | None = None
        self._dataset: Dataset | None = None
        self._loss_fn: _Loss | None = None
        self._optimizer: optim.Optimizer | None = None
        self._device: str | None = None
        self._log_dir: str | None = None

    def model(self, model: Sequential) -> Self:
        self._model = model
        logger.debug(f"add model: {model}")
        return self

    def dataset(self, dataset: Dataset) -> Self:
        self._dataset = dataset
        logger.debug(f"add dataset: {dataset}")
        return self

    def loss_fn(self, loss_fn: _Loss) -> Self:
        self._loss_fn = loss_fn
        logger.debug(f"add loss_fn: {loss_fn}")
        return self

    def optimizer(self, optimizer: optim.Optimizer) -> Self:
        self._optimizer = optimizer
        logger.debug(f"add optimizer: {optimizer}")
        return self

    def device(self, device: str) -> Self:
        self._device = device
        logger.debug(f"add device: {device}")
        return self

    def log_dir(self, log_dir: str) -> Self:
        self._log_dir = log_dir
        logger.debug(f"add tensorboard_log_dir: {log_dir}")
        return self

    def build(self) -> Training:
        """Build Training instance.

        Raises:
            VALUE_ERROR: If required fields are missing.

        Returns:
            Training: Training instance.

        """
        if self._model is None:
            raise ValueError("TrainingBuilder missing required Model!")
        if self._dataset is None:
            raise ValueError("TrainingBuilder missing required Dataset!")

        loss_fn = self._loss_fn or CrossEntropyLoss()
        optimizer = self._optimizer or optim.Adam(self._model.parameters())
        device = self._device or "cpu"
        log_dir = self._log_dir or f"logs/{int(datetime.now().timestamp() * 1000)}"

        logger.debug(
            f"build Training: {self._model=}, {self._dataset=}, {loss_fn=}, {optimizer=}, {device=}, {log_dir=}"
        )

        return Training(
            model=self._model,
            dataset=self._dataset,
            loss_fn=loss_fn,
            optimiezer=optimizer,
            device=device,
            log_dir=log_dir,
        )
